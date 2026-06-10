# Task 2
import os
import numpy as np
import h5py
from tqdm.auto import tqdm

import graphidxbaselines as gib
import hiob

from util import *

_ALGO = "HaNSWurst"
_TASK = "task2"

def make_output(args, n, build_time, query_time, qresult, true_nn_ids):
	outpath = os.path.join(args.output,f"{_ALGO}_ef={n}.h5")
	os.makedirs(os.path.dirname(outpath),exist_ok=True)
	with h5py.File(outpath,"w") as f_out:
		# Write arrays
		f_out["knns"] = qresult[0]+1
		f_out["dists"] = qresult[1]
		# Write attributes
		f_out.attrs["algo"] = _ALGO
		f_out.attrs["task"] = _TASK
		f_out.attrs["buildtime"] = build_time
		f_out.attrs["querytime"] = query_time
		f_out.attrs["params"] = f"ef={n}"
		f_out.attrs["recall"] = recall(qresult[0], true_nn_ids)
		f_out.attrs["dataset"] = args.task_description["dataset_name"]
def run(args):
	print(args)
	if 1: # Primitive type selection
		_int, _uint, _float, _floatint, _floatuint = get_gib_primitives()
		assert _float == np.float16 and _uint == np.uint32
		float_size = np.dtype(_float).itemsize
		int_size = np.dtype(_int).itemsize
		print(f"Running with f{float_size*8}/u{int_size*8}")
	if 1: # Load dataset
		f = h5py.File(args.input,"r")
		X = np.array(f[args.task_description["data"]]).astype(_float)
		Q = np.array(f[args.task_description["queries"]]).astype(_float)
		true_nn_ids = np.array(f[args.task_description["gt_I"]])-1
		k = args.task_description["k"]
		true_nn_ids = true_nn_ids[:,:k]
	if 1: # Hyperparameters
		dist_build = lambda: gib.DotProdSurrogateMix(-0.85)
		dist_query = lambda: gib.NegDotProduct()
		degree = 200
		cap_const = 10
		cap_query = 50
		ns = np.round(np.exp(np.linspace(np.log(80),np.log(500),15))).astype(int)
	if 1: # Run benchmark
		with tqdm(total=1+len(ns)) as pbar:
			with Timer("Build") as build_timer:
				_dist = dist_build()
				_X = append_norm(X)
				# Build HNSW
				idx = gib.PyHNSW(
					_X,
					higher_max_degree=degree//2,
					lowest_max_degree=degree,
					distance=_dist.to_enum(),
					max_build_frontier_size=cap_const,
					max_frontier_size=cap_query,
					insert_heuristic=False,
				)
				idx.with_distance_and_data(dist_query().to_enum(), X)
			build_time = build_timer.delta_time
			pbar.update(1)
			for n in ns:
				with Timer("Query", print=False) as query_timer:
					qresult = idx.knn_query_batch(Q,k,n)
				query_time = query_timer.delta_time
				make_output(args, n, build_time, query_time, qresult, true_nn_ids)
				pbar.update(1)

