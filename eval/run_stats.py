import os, sys

def print_usage_and_exit():
	print("Usage: `python run_stats.py <directory_or_tira_hash>`\nwhere the argument is either a directory containing result *.h5 files or the hash returned by a TIRA dry run referencing the directory `/tmp/tira-<hash>`.")
	sys.exit(1)

if len(sys.argv) != 2: print_usage_and_exit()

base_path = sys.argv[1]
if not os.path.isdir(base_path): base_path = f"/tmp/tira-{base_path}"
if not os.path.isdir(base_path):
	print("Argument does not specify a directory.")
	print_usage_and_exit()
try:
	files = [f for f in os.listdir(base_path) if f.endswith(".h5")]
	get_n_from_file = lambda s: int(s.split("=")[1].split(".")[0])
	files = sorted(files, key=get_n_from_file)
	ns = list(map(get_n_from_file, files))
except Exception as e:
	print(e)
	print_usage_and_exit()

import h5py
import numpy as np
import pandas as pd
columns = None
data = []
for n,f in zip(ns,files):
	with h5py.File(base_path+"/"+f) as hf:
		if columns is None: columns = list(hf.attrs.keys())
		data.append([n, *[hf.attrs[col] for col in columns]])
		# print(f"{n:>3} {hf.attrs['querytime']:>7.03f} {hf.attrs['recall']:>7.03f}")
if columns is None: print_usage_and_exit()
print(pd.DataFrame(data, columns=["n"]+columns))
