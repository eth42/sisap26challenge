#!/bin/bash

# Adapted from
# https://github.com/sisap-challenges/sisap26-python-baseline/blob/main/run_search.sh
# License:
# MIT License

# Copyright (c) 2026 SISAP Challenges

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Ensure that the container is built
# ./docker_build.sh

# Parse options to swap to small datasets
tasks="1 2 3"
small=false
for arg in "$@"; do
	if [[ "$arg" == "--small" ]];
	then small=true
	else tasks=$arg
	fi
done

for task in $tasks; do
	echo Running Task $task
	if [[ "$small" == false ]]; then
		# Big datasets
		case $task in
			1) dataset="wikipedia" ;;
			2) dataset="llama-dev" ;;
			3) dataset="nq" ;;
			*) echo "Task $task not recognized."; exit 1;
		esac
	else
		# small datasets
		case $task in
			1) dataset="wikipedia-small" ;;
			2) dataset="llama-dev" ;;
			3) dataset="fiqa-dev" ;;
			*) echo "Task $task not recognized."; exit 1;
		esac
	fi
	mkdir -p results/$dataset
	chmod a+rwx results/$dataset
	h5name="$(basename $(ls data/$dataset/*.h5))"
	docker run \
		--rm \
		--cpus=8 \
		--memory=24g \
		--memory-swap=24g \
		--memory-swappiness 0 \
		--volume $(pwd)/data:/app/data:ro \
		--volume $(pwd)/results:/app/results:rw \
		sisap26 python /app/runner.py \
			--input /app/data/$dataset/$h5name \
			--task-description /app/data/$dataset/config.json \
			--output /app/results/$dataset/
done
