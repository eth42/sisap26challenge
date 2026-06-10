#!/bin/bash

set -euo pipefail

source pyenv/bin/activate

# --- Command blocks ---
run_1() {
	tira-cli code-submission \
		--path . \
		--command 'python /app/runner.py --input $inputDataset/*.h5 --task-description $inputDataset/config.json --output $outputDir' \
		--task sisap-2026 \
		--dataset task-1-spot-check-20260528-training \
		--dry-run
}
run_2() {
	tira-cli code-submission \
		--path . \
		--command 'ls -hal /app; python /app/runner.py --input $inputDataset/*.h5 --task-description $inputDataset/config.json --output $outputDir' \
		--task sisap-2026 \
		--dataset task-2-spot-check-20260528-training \
		--dry-run --verbose
}
run_3() {
	tira-cli code-submission \
		--path . \
		--command 'python /app/runner.py --input $inputDataset/*.h5 --task-description $inputDataset/config.json --output $outputDir' \
		--task sisap-2026 \
		--dataset task-3-spot-check-20260529-training \
		--dry-run
}

# --- Argument handling ---
if [[ $# -gt 1 ]]; then
	echo "Error: At most one argument allowed (1, 2, or 3)." >&2
	exit 1
fi

if [[ $# -eq 1 ]]; then
	arg="$1"
	# Validate: must be exactly 1, 2, or 3
	if [[ ! "$arg" =~ ^[1-3]$ ]]; then
		echo "Error: Argument must be one of 1, 2, or 3." >&2
		exit 1
	fi
	case "$arg" in
		1) run_1 ;;
		2) run_2 ;;
		3) run_3 ;;
	esac
else
	# No argument: run all
	run_1
	run_2
	run_3
fi

