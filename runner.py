#!/bin/python

def get_args():
	import argparse
	import json
	parser = argparse.ArgumentParser()
	# --input $inputDataset/*.h5 --task-description $inputDataset/config.json --output $outputDir
	parser.add_argument('-i', '--input')
	parser.add_argument('-o', '--output')
	parser.add_argument('-t', '--task-description')
	args = parser.parse_args()
	# https://github.com/sisap-challenges/sisap26-python-baseline#task-configuration-format-configjson
	print(args)
	with open(args.task_description, "rt") as td:
		args.task_description = json.loads(td.read())
	print(args)
	return args

args = get_args()
_task = args.task_description["task"]
if _task == "task1":
	from task1 import run
	run(args)
elif _task == "task2":
	from task2 import run
	run(args)
elif _task == "task3":
	from task3 import run
	run(args)
else:
	raise ValueError(f"Unknown task '{_task}'")

