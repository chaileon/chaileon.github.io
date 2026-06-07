#!/usr/bin/env bash
# Sweep every persuader config across all topic sets, with and without
# native thinking mode. Results land in output/<think|nothink>/<topicset>/<config>.

set -e

TURNS=3
TEMP=0.7
REPEATS=5

for mode in nothink think; do
	think_flag=""
	[ "$mode" = "think" ] && think_flag="--think"

	for topicset in topics topics2 topics3; do
		odir="output/$mode/$topicset"
		mkdir -p "$odir"
		for cfg in persuader_config*.json; do
			echo "=== [$mode] $topicset / $cfg ==="
			python3 debate_ollama.py \
				--topics "$topicset.json" \
				--persuader-prompt "$cfg" \
				--turns "$TURNS" \
				--temperature "$TEMP" \
				--repeats "$REPEATS" \
				$think_flag \
				--output "$odir/$cfg"
		done
	done
done
