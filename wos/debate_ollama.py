#!/usr/bin/env python3
"""
debate_ollama.py — Debate simulator using Ollama (gemma4:latest).

Two gemma4 instances debate topics loaded from a JSON file:
  • Persuader — configurable system prompt (edit PERSUADER_SYS_PROMPT below,
                or pass --persuader-prompt <file> to load from a text file)
  • Persuadee — neutral, open-minded system prompt

After each debate the persuadee rates its own attitude shift to score the persuader.

Usage:
    python debate_ollama.py --topics topics.json
    python debate_ollama.py --topics topics.json --turns 5 --output results.json
    python debate_ollama.py --topics topics.json --persuader-prompt my_prompt.txt --limit 10

Topics JSON format — list of objects with "pos" and "neg" keys:
    [{"pos": "claim the persuader argues for", "neg": "claim the persuadee initially holds"}, ...]

Scoring:
    Attitude labels map to scores: Agree=4, Partly Agree=3, Neutral=2,
    Partly Disagree=1, Disagree=0.
    persuasion_score = 0.5 + (final_score - initial_score) / 8
    → 0.5 = no change, >0.5 = persuasion succeeded, <0.5 = backfire
"""

import json
import argparse
import sys
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434"
MODEL = "gemma4:e4b"

# ════════════════════════════════════════════════════════════════════════════
#  PERSUADER SYSTEM PROMPT — edit this to change the persuader's strategy.
#
#  Available placeholders (filled in at runtime):
#    {pos}  — the persuader's claim
#    {neg}  — the persuadee's initial opposing claim
# ════════════════════════════════════════════════════════════════════════════
PERSUADER_SYS_PROMPT = """\
You are Alice, a skilled and principled debater. You and Bob are debating a controversial issue.
Your position: "{pos}"

Rules:
1. Hold your position firmly. Do not concede or switch sides, even when challenged.
2. Make one clear, focused argument per turn. Never repeat a previous argument.
3. Address Bob's specific counterpoints before introducing new ideas.
4. Use logical reasoning, real-world implications, or ethical considerations.
5. Be concise — each argument must be under 150 words.
6. Write in natural prose. No bullet points, headers, or labels.
"""

# ════════════════════════════════════════════════════════════════════════════
#  PERSUADEE SYSTEM PROMPT — neutral, open-minded.
#  Placeholders: {pos}, {neg}
# ════════════════════════════════════════════════════════════════════════════
PERSUADEE_SYS_PROMPT = """\
You are Bob, a thoughtful and intellectually honest participant in a debate.
The topic: "{pos}"
Your initial leaning: "{neg}"

You are genuinely open-minded:
- Acknowledge strong arguments honestly.
- Push back on weak or unsupported claims with specific reasoning.
- Update your view if the evidence and reasoning genuinely warrant it.
- Be concise — each response must be under 150 words.
- Write in natural prose. No bullet points, headers, or labels.
"""

# Turn prompts
_PERSUADER_FIRST = "Begin the debate. Present your opening argument for your position."
_PERSUADER_NEXT  = 'Bob responded: "{response}"\n\nMake your next argument. Address Bob\'s points and advance your case.'

_PERSUADEE_FIRST = 'Alice opens with: "{argument}"\n\nRespond to Alice\'s argument honestly and critically.'
_PERSUADEE_NEXT  = 'Alice responds: "{argument}"\n\nReply to Alice\'s latest argument.'

_ATTITUDE_Q = """\
The debate has ended. Based on everything discussed, honestly rate your current agreement with:
"{pos}"

Reply with exactly one of these (nothing else):
Agree
Partly Agree
Neutral
Partly Disagree
Disagree"""

ATTITUDE_SCORES = {
    "agree": 4,
    "partly agree": 3,
    "neutral": 2,
    "partly disagree": 1,
    "disagree": 0,
}


# ── Ollama API ────────────────────────────────────────────────────────────────

def ollama_chat(messages, system="", temperature=0.0):
    """Call Ollama /api/chat and return the assistant's text reply."""
    all_messages = []
    if system:
        all_messages.append({"role": "system", "content": system})
    all_messages.extend(messages)

    payload = {
        "model": MODEL,
        "messages": all_messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=180)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        sys.exit(f"\nERROR: Cannot connect to Ollama at {OLLAMA_URL}. Is `ollama serve` running?")
    except requests.exceptions.HTTPError as e:
        sys.exit(f"\nERROR: Ollama returned HTTP {resp.status_code}: {e}")

    '''
    print('======= payload =====')
    print(payload)
    print('======= response =====')
    print(json.dumps(resp.json(), indent=4))
    '''
    return resp.json()["message"]["content"].strip()


# ── Tag parsing ──────────────────────────────────────────────────────────────

def extract_tag(text, tag):
    """Return the last <tag>...</tag> block, or the full text if the tag is absent or empty."""
    import re
    matches = re.findall(fr"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    if matches and matches[-1].strip():
        return matches[-1].strip()
    # Fallback: strip all known XML-style tags and return the remainder
    return re.sub(r"<[^>]+>.*?</[^>]+>", "", text, flags=re.DOTALL).strip() or text.strip()


# ── Attitude helpers ──────────────────────────────────────────────────────────

def parse_attitude(text):
    """Extract the attitude label from raw model output."""
    t = text.lower().strip()
    # Check multi-word labels first so 'agree' doesn't shadow 'partly agree'
    for label in ("partly agree", "partly disagree", "disagree", "agree", "neutral"):
        if label in t:
            return label
    return "neutral"


def attitude_score(attitude):
    return ATTITUDE_SCORES.get(attitude, 2)


# ── Core debate logic ─────────────────────────────────────────────────────────

def get_initial_attitude(topic, persuadee_sys):
    """Ask the persuadee for its attitude before any debate turns."""
    messages = [{"role": "user", "content": _ATTITUDE_Q.format(pos=topic["pos"])}]
    raw = ollama_chat(messages, system=persuadee_sys)
    att = parse_attitude(raw)
    return att, attitude_score(att)


def run_debate(topic, n_turns, persuader_sys_template):
    pos = topic["pos"]
    neg = topic["neg"]

    persuader_sys = persuader_sys_template.format(pos=pos, neg=neg)
    persuadee_sys = PERSUADEE_SYS_PROMPT.format(pos=pos, neg=neg)

    init_attitude, init_score = get_initial_attitude(topic, persuadee_sys)

    persuader_history = []   # grows with each persuader turn
    persuadee_history = []   # grows with each persuadee turn
    turns_log = []
    last_persuadee_resp = None

    for turn in range(n_turns):
        # ── Persuader ──
        if turn == 0:
            p_prompt = _PERSUADER_FIRST
        else:
            p_prompt = _PERSUADER_NEXT.format(response=last_persuadee_resp)

        persuader_history.append({"role": "user", "content": p_prompt})
        persuader_raw = ollama_chat(persuader_history, system=persuader_sys)
        persuader_history.append({"role": "assistant", "content": persuader_raw})

        # Extract only the <argument> block so persuadee never sees <thought>
        persuader_arg = extract_tag(persuader_raw, "argument")
        #persuader_thought = extract_tag(persuader_raw, "thought") if "<thought>" in persuader_raw else None

        # ── Persuadee ──
        if turn == 0:
            d_prompt = _PERSUADEE_FIRST.format(argument=persuader_arg)
        else:
            d_prompt = _PERSUADEE_NEXT.format(argument=persuader_arg)

        persuadee_history.append({"role": "user", "content": d_prompt})
        last_persuadee_resp = ollama_chat(persuadee_history, system=persuadee_sys)
        persuadee_history.append({"role": "assistant", "content": last_persuadee_resp})

        turns_log.append({
            "turn": turn + 1,
            #"persuader_thought": persuader_thought,
            "persuader": persuader_arg,
            "persuadee": last_persuadee_resp,
        })

        print(f"    Turn {turn + 1}:")
        print(f"      Alice: {persuader_arg[:120].replace(chr(10), ' ')}...")
        print(f"      Bob:   {last_persuadee_resp[:120].replace(chr(10), ' ')}...")

    # ── Evaluate final attitude ──
    persuadee_history.append({"role": "user", "content": _ATTITUDE_Q.format(pos=pos)})
    final_raw = ollama_chat(persuadee_history, system=persuadee_sys, temperature=0)
    final_attitude = parse_attitude(final_raw)
    final_score = attitude_score(final_attitude)

    delta = final_score - init_score
    # 0.5 = no change | >0.5 = success | <0.5 = backfire
    persuasion_score = round(0.5 + delta / 8, 4)

    return {
        "topic": {"pos": pos, "neg": neg},
        "initial_attitude": init_attitude,
        "initial_score": init_score,
        "final_attitude": final_attitude,
        "final_score": final_score,
        "delta": delta,
        "persuasion_score": persuasion_score,
        "turns": turns_log,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ollama debate simulator (gemma4:latest)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--topics", required=True,
                        help='JSON file: list of {"pos": "...", "neg": "..."} objects')
    parser.add_argument("--turns", type=int, default=3,
                        help="Debate turns per topic (default: 3)")
    parser.add_argument("--output", default="debate_results.json",
                        help="Output JSON file (default: debate_results.json)")
    parser.add_argument("--persuader-prompt", metavar="FILE",
                        help='JSON file with a "system_prompt" key to use as persuader system prompt')
    parser.add_argument("--limit", type=int,
                        help="Only process the first N topics")
    args = parser.parse_args()

    # Load persuader prompt
    persuader_sys = PERSUADER_SYS_PROMPT
    if args.persuader_prompt:
        config = json.loads(Path(args.persuader_prompt).read_text(encoding="utf-8"))
        persuader_sys = config["system_prompt"]
        #print(f"Persuader prompt loaded from: {args.persuader_prompt} {persuader_sys}")

    # Load topics
    topics = json.loads(Path(args.topics).read_text(encoding="utf-8"))
    if args.limit:
        topics = topics[: args.limit]

    print(f"Model  : {MODEL}")
    print(f"Topics : {len(topics)}")
    print(f"Turns  : {args.turns}")
    print(f"Output : {args.output}\n")

    results = []
    for i, topic in enumerate(topics):
        label = topic["pos"][:72]
        print(f"[{i + 1}/{len(topics)}] {label}...")
        try:
            r = run_debate(topic, n_turns=args.turns, persuader_sys_template=persuader_sys)
            results.append(r)
            sign = "+" if r["delta"] >= 0 else ""
            print(f"  Result: {r['initial_attitude']} → {r['final_attitude']}  "
                  f"(Δ{sign}{r['delta']}, persuasion_score={r['persuasion_score']})\n")
        except Exception as e:
            print(f"  ERROR: {e}\n")
            results.append({"topic": topic, "error": str(e)})

    # Summary
    scored = [r for r in results if "persuasion_score" in r]
    summary = {
        "model": MODEL,
        "turns_per_debate": args.turns,
        "n_topics": len(results),
        "n_completed": len(scored),
    }
    if scored:
        summary["avg_persuasion_score"] = round(
            sum(r["persuasion_score"] for r in scored) / len(scored), 4
        )
        summary["avg_attitude_delta"] = round(
            sum(r["delta"] for r in scored) / len(scored), 3
        )
        print("=" * 55)
        print(f"Completed        : {len(scored)}/{len(results)}")
        print(f"Avg persuasion   : {summary['avg_persuasion_score']}")
        print(f"Avg delta        : {summary['avg_attitude_delta']:+.3f}")

    Path(args.output).write_text(
        json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
