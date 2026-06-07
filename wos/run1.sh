#!/usr/bin/env bash
# Sweep every persuader config across all topic sets.
# Results land in output1/<topicset>/<config>.

set -e

TURNS=3
TEMP=0.7
REPEATS=3

for topicset in topics topics2 topics3; do
	odir="output1/$topicset"
	mkdir -p "$odir"
	for cfg in persuader_config*.json; do
		echo "=== $topicset / $cfg ==="
		python3 debate_ollama.py \
			--topics "$topicset.json" \
			--persuader-prompt "$cfg" \
			--turns "$TURNS" \
			--temperature "$TEMP" \
			--repeats "$REPEATS" \
			--output "$odir/$cfg"
	done
done
