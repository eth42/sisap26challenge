# Task 1
import os
if os.path.basename(os.path.abspath(".")) == "_sisap2026": os.chdir("..")

import numpy as np
import h5py
from tqdm.auto import tqdm
import multiprocessing
import sys

import graphidxbaselines as gib
import hiob

from util import *

_ALGO = "HotSwap"
_TASK = "task1"


def make_output(args, n, build_time, query_time, qresult, true_nn_ids, **additional_attrs):
	outpath = os.path.join(args.output,f"{_ALGO}_ef={n}.h5")
	# outpath = os.path.join(args.output,_TASK,f"HotSwap_ef={n}.h5")
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
		for k,v in additional_attrs.items(): f_out.attrs[k] = v
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
		_true_knn_path = args.task_description["gt_I"]
		if type(_true_knn_path) == list: _true_knn_path = "/".join(_true_knn_path)
		true_nn_ids = np.array(f[_true_knn_path])-1
		k = args.task_description["k"]
		true_nn_ids = true_nn_ids[:,:k]
		assert true_nn_ids.shape[0] == f[args.task_description["data"]].shape[0]
		assert true_nn_ids.shape[1] == k
	if 1: # Hyperparameters
		# HIOB
		n_bits = 512
		run_total = 10_000
		sample_size = 10_000
		its_per_sample = 10_000
		scale = .15
		run_batch = 1_000
		# HNSW
		dist_build = lambda: gib.HammingDistance()
		dist_query = lambda: gib.NegDotProduct()
		degree = 50
		cap_const = None
		cap_query = 20
		# Queries
		ns = np.round(np.exp(np.linspace(np.log(40),np.log(150),15))).astype(int)
	if 1: # Parameter sets
		# Arguments for the sketcher holding the dataset during training
		base_sketcher_kwargs = dict(
			sample_size=sample_size,
			its_per_sample=its_per_sample,
			n_bits=n_bits,
			scale=scale,
			output_type=np.uint64,
			init_greedy=False,
			init_itq=False,
			init_ransac=True,
		)
		# Arguments for the sketcher not holding any data post training
		derived_sketcher_kwargs = dict(
			X=np.zeros((1,f[args.task_description["data"]].shape[1]),dtype=_float),
			sample_size=1,its_per_sample=1,
			n_bits=n_bits,
			output_type=_floatuint,
			init_greedy=False,
			init_itq=False,
			init_ransac=False,
		)
	if 1: # Helper functions
		def batch_sketch(sketcher, X, chunk_size=50_000):
			n_bits = sketcher.n_bits
			n_bins = n_bits // (float_size*8)
			bits = np.empty((X.shape[0],n_bins),dtype=_float)
			for start in tqdm(range(0,X.shape[0],chunk_size),leave=False,desc="Sketching"):
				end = min(start+chunk_size,X.shape[0])
				if end-start <= 0: continue
				bits[start:end] = sketcher.binarize(X[start:end]).view(_float)
			return bits
		def batch_self_join(X, hnsw, k, heap_size=None, chunk_size=50_000):
			_ids = np.empty((X.shape[0],k), dtype=_int)
			_dists = np.empty((X.shape[0],k), dtype=_float)
			for start in tqdm(range(0,X.shape[0],chunk_size),leave=False,desc="Self Join"):
				end = min(start+chunk_size,X.shape[0])
				if end-start <= 0: continue
				_ids[start:end], _dists[start:end] = hnsw.self_join_query_local_arr(k, heap_size or 2*k, (start,end))
			return _ids, _dists
		def produce_sketches(queue):
			try:
				with Timer("Loading dataset") as timer_load_data:
					X = np.empty(f[args.task_description["data"]].shape, dtype=_float)
					f[args.task_description["data"]].read_direct(X)
				with Timer("Sketcher initialization") as timer_sketch_init:
					sketcher = hiob.StochasticHIOB.from_ndarray(X=X,**base_sketcher_kwargs)
				with Timer("Sketcher training") as timer_sketch_train:
					with tqdm(total=run_total) as pbar:
						for i in range(run_total//run_batch):
							sketcher.run(run_batch)
							pbar.desc=f"Training sketcher: {np.mean(sketcher.sim_mat):.5f}"
							pbar.update(run_batch)
				if 1: # Move trained hyperplanes to new sketcher that does not know the data
					trained_sketcher = hiob.StochasticHIOB.from_ndarray(
						centers=sketcher.centers.copy(),
						**derived_sketcher_kwargs,
					)
					del sketcher
				with Timer("Sketching") as timer_sketch:
					Xsketch = batch_sketch(trained_sketcher, X)
				queue.put((
					Xsketch,
					timer_load_data.delta_time,
					timer_sketch_init.delta_time,
					timer_sketch_train.delta_time,
					timer_sketch.delta_time,
				))
			except Exception as e:
				print(e, flush=True)
				queue.put(None)
			if 1: # Forcefully flush the dataset from memory
				X.resize((1,X.shape[1]), refcheck=False)
				del X
		def async_produce_sketches():
			queue = multiprocessing.Queue()
			proc = multiprocessing.Process(target=produce_sketches, args=(queue,))
			proc.start()
			result = queue.get()
			queue.close()
			queue.join_thread()
			proc.join()
			assert result is not None, "Sketching failed"
			return result
	if 1: # Run benchmark
		with tqdm(total=1+len(ns)) as pbar:
			with Timer("Build") as build_timer:
				Xsketch, t_load_data, t_sketch_init, t_sketch_train, t_sketch = async_produce_sketches()
				# Build HNSW
				with Timer("Build HNSW") as timer_build_hnsw:
					idx = gib.PyHNSW(
						Xsketch,
						higher_max_degree=degree//2,
						lowest_max_degree=degree,
						distance=dist_build().to_enum(),
						max_build_frontier_size=cap_const,
						max_frontier_size=cap_query,
						# insert_heuristic=False,
					)
				with Timer("Reload dataset") as timer_reload_data:
					X = np.empty(f[args.task_description["data"]].shape, dtype=_float)
					f[args.task_description["data"]].read_direct(X)
					idx.with_distance_and_data(dist_query().to_enum(), X)
					del Xsketch
			build_time = build_timer.delta_time
			t_build_hnsw = timer_build_hnsw.delta_time
			t_reload_data = timer_reload_data.delta_time
			pbar.update(1)
			for n in ns:
				with Timer("Query", print=False) as query_timer:
					qresult = batch_self_join(X, idx, k, n)
				query_time = query_timer.delta_time
				make_output(
					args, n, build_time, query_time, qresult, true_nn_ids,
					loadtime=t_load_data,
					reloadtime=t_reload_data,
					sketchinittime=t_sketch_init,
					sketchtraintime=t_sketch_train,
					sketchingtime=t_sketch,
					hnswconstructiontime=t_build_hnsw,
				)
				del qresult
				pbar.update(1)

