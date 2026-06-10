# Task 3
import os
if os.path.basename(os.path.abspath(".")) == "_sisap2026": os.chdir("..")

import numpy as np
import scipy
from scipy.sparse import csr_matrix
import h5py
from tqdm.auto import tqdm

import graphidxbaselines as gib
import hiob

from util import *

_ALGO = "HotSwap"
_TASK = "task3"
_N_DIMS = 30_522

def make_output(args, n, build_time, query_time, qresult, true_nn_ids):
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
		def read_csr(v):
			def read_field(w, t):
				ret = np.empty(w.shape, dtype=t)
				w.read_direct(ret)
				return ret
			return [
				read_field(v[k],t)
				for k,t in [
					("data", _float),
					("indices", _floatint),
					("indptr", _int),
				]
			]
		Xd, Xin, Xip = read_csr(f[args.task_description["data"]])
		Qd, Qin, Qip = read_csr(f[args.task_description["queries"]])
		true_nn_ids = np.array(f[args.task_description["gt_I"]])-1
		k = args.task_description["k"]
		true_nn_ids = true_nn_ids[:,:k]
	if 1: # Hyperparameters
		dist = lambda: gib.SparseNegDotProduct()
		degree = 200
		cap_const = 100
		cap_query = 50
		ns = np.round(np.exp(np.linspace(np.log(80),np.log(700),15))).astype(int)
	if 1: # Run benchmark
		with tqdm(total=1+len(ns)) as pbar:
			with Timer("Build") as build_timer:
				_dist = dist()
				# Build HNSW
				idx = gib.SparsePyHNSW(
					Xd, Xin, Xip,
					higher_max_degree=degree//2,
					lowest_max_degree=degree,
					distance=_dist.to_enum(),
					max_build_frontier_size=cap_const,
					max_frontier_size=cap_query,
					insert_heuristic=False,
				)
			build_time = build_timer.delta_time
			pbar.update(1)
			for n in ns:
				with Timer("Query", print=False) as query_timer:
					qresult = idx.knn_query_batch(Qd,Qin,Qip,k,n)
				query_time = query_timer.delta_time
				make_output(args, n, build_time, query_time, qresult, true_nn_ids)
				pbar.update(1)


if 0:
	if 1: # Primitive type selection
		assert gib.ref_bits() in [32,64]
		assert gib.prec_bits() in [16,32,64]
		if gib.ref_bits() == 32:
			_int = np.int32
			_uint = np.uint32
		else:
			_int = np.int64
			_uint = np.uint64
		if gib.prec_bits() == 16:
			_float = np.float16
			_floatint = np.int16
			_floatuint = np.uint16
		elif gib.prec_bits() == 32:
			_float = np.float32
			_floatint = np.int32
			_floatuint = np.uint32
		else:
			_float = np.float64
			_floatint = np.int64
			_floatuint = np.uint64
		float_size = np.dtype(_float).itemsize
		int_size = np.dtype(_uint).itemsize
		print(f"Running experiments with f{float_size*8}/u{int_size*8}")

	if 0: # fiqa dataset (sparse)
		from scipy.sparse import csr_matrix
		f = h5py.File("data/fiqa-dev.h5")
		read_csr = lambda v, rows: csr_matrix((v["data"][:], v["indices"][:], v["indptr"][:]), shape=(rows,30522))
		X = read_csr(f["train"], 57638)
		Q = read_csr(f["otest"]["queries"], 6648)
		true_nn_ids = np.array(f["otest"]["knns"])-1
		k = 30
		true_nn_ids = true_nn_ids[:,:k]
		data_variant = "fiqa"
	else: # nq dataset (sparse)
		from scipy.sparse import csr_matrix
		f = h5py.File("data/nq.h5")
		read_csr = lambda v: csr_matrix((v["data"][:], v["indices"][:], v["indptr"][:]), shape=(v["indptr"].shape[0]-1,30522))
		X = read_csr(f["train"])
		Q = read_csr(f["otest"]["queries"])
		true_nn_ids = np.array(f["otest"]["knns"])-1
		k = 30
		true_nn_ids = true_nn_ids[:,:k]
		data_variant = "nq"

	if 0: # Ensure datatypes fit graphidxbaselines compilation
		assert X.data.dtype == _float
		assert X.indices.dtype == _floatint
		assert X.indptr.dtype == _floatint
		assert Q.data.dtype == _float
		assert Q.indices.dtype == _float
		assert Q.indptr.dtype == _float

	if 0: # Preselection of distance to use
		if 1: # Hyperparams
			# -0.2 seems to work best on fiqa
			# -0.05 seems to work best on nq
			dot_surrogate_mix_factors = np.linspace(-.15,.1,17)
			dot_surrogate_mix_factors = np.array([-.05 if X.shape[0] > 2.5e6 else -.2])
			# dist_build, dist_build_args, norm_build
			parameterizations = [
				(gib.SparseSquaredEuclideanDistance, (), False),
				(gib.SparseSquaredEuclideanDistance, (), True),
				(gib.SparseNegDotProduct, (), False),
				(gib.SparseNegDotProduct, (), True),
				*[
					(gib.SparseDotProdSurrogateMix, (f,), False)
					for f in dot_surrogate_mix_factors
				],
			]
		if 1: # Experiment params
			n_idxs = 5
			ns = np.round(k * np.exp(np.log(1.1) * np.arange(-4,31))).astype(int)
			query_repetitions = 5
		if 1: # Init storage
			_full_shape = (len(parameterizations), n_idxs, len(ns), query_repetitions)
			build_times = np.zeros(_full_shape[:2])
			query_times = np.zeros(_full_shape)
			recalls = np.zeros(_full_shape)

		if 0: # Compute k@n curves
			with tqdm(total=len(parameterizations)*n_idxs*(1+len(ns)*query_repetitions)) as pbar:
				for i_param, params in enumerate(parameterizations):
					for i_idx in range(n_idxs):
						dist_build, dist_build_args, norm_build = params
						with Timer("Build", print=False) as build_timer:
							_dist = dist_build(*dist_build_args)
							_X = append_norm(X) if dist_build == gib.SparseDotProdSurrogateMix else normalize(X) if norm_build else X
							# Build HNSW
							_Xd, _Xin, _Xip = [v.astype(t, copy=False) for v,t in [
								[_X.data, _float],
								[_X.indices, _floatint],
								[_X.indptr, _int],
							]]
							idx = gib.SparsePyHNSW(
								_Xd, _Xin, _Xip,
								distance=_dist.to_enum(),
								insert_heuristic=False,
								# higher_max_degree=25,
								# lowest_max_degree=50,
								# max_build_frontier_size=10,
								# max_frontier_size=5,
							)
							Xd, Xin, Xip = [v.astype(t, copy=False) for v,t in [
								[X.data, _float],
								[X.indices, _floatint],
								[X.indptr, _int],
							]] if norm_build else [_Xd, _Xin, _Xip]
							idx.with_distance_and_data(gib.SparseNegDotProduct().to_enum(), Xd, Xin, Xip)
							del _X, _Xd, _Xin, _Xip
						build_times[i_param,i_idx] = build_timer.delta_time
						pbar.update(1)
						for i_n, n in enumerate(ns):
							for i_rep in range(query_repetitions):
								with Timer("Query", print=False) as query_timer:
									Qd, Qin, Qip = [v.astype(t, copy=False) for v,t in [
										[Q.data, _float],
										[Q.indices, _floatint],
										[Q.indptr, _int],
									]]
									qresult = idx.knn_query_batch(Qd, Qin, Qip,k,n)
									del Qd, Qin, Qip
								query_times[i_param,i_idx,i_n,i_rep] = query_timer.delta_time
								recalls[i_param,i_idx,i_n,i_rep] = recall(true_nn_ids,qresult[0])
								pbar.update(1)
						del idx
						np.save(f"_sisap2026/_task3_build_times_{data_variant}.npy", build_times)
						np.save(f"_sisap2026/_task3_query_times_{data_variant}.npy", query_times)
						np.save(f"_sisap2026/_task3_recalls_{data_variant}.npy", recalls)
		else:
			build_times = np.load(f"_sisap2026/_task3_build_times_{data_variant}.npy")
			query_times = np.load(f"_sisap2026/_task3_query_times_{data_variant}.npy")
			recalls = np.load(f"_sisap2026/_task3_recalls_{data_variant}.npy")

	if 1: # Tuning HNSW parameters
		variant = ["degree","buildcap","querycap"][2]
		print(f"Running HNSW tuning variant '{variant}'")
		if 1: # Hyperparams
			# ef_const, cap_const
			if variant == "degree":
				parameterizations = [
					(ef_const, None)
					for ef_const in [25, 50, 100, 150, 200, 250]
				]
			elif variant == "buildcap":
				parameterizations = [
					(ef_const, cap_const)
					for ef_const in [200]
					for cap_const in [None, 10, 20, 50, 100]
				]
			elif variant == "querycap":
				parameterizations = [
					# TODO: Repeat with [200, None], that was slightly faster
					[200, 100],
				]
			else: raise ValueError(f"Unknown variant '{variant}'")
		if 1: # Experiment params
			n_idxs = 5
			query_repetitions = 5
			if variant == "degree":
				ns = np.round(k * np.exp(np.log(1.1) * np.linspace(0,30,15))).astype(int)
				cap_sizes = [None]
			elif variant == "buildcap":
				ns = np.round(k * np.exp(np.log(1.1) * np.linspace(0,30,15))).astype(int)
				cap_sizes = [None]
			elif variant == "querycap":
				ns = np.round(k * np.exp(np.log(1.1) * np.linspace(0,30,15))).astype(int)
				cap_sizes = [None, 5, 10, 20, 30, 40, 50, 100]
			else: raise ValueError(f"Unknown variant '{variant}'")
		if 1: # Init storage
			_full_shape = (len(parameterizations), n_idxs, len(ns), len(cap_sizes), query_repetitions)
			build_times = np.zeros(_full_shape[:2])
			query_times = np.zeros(_full_shape)
			recalls = np.zeros(_full_shape)

		if 1: # Compute k@n curves
			with tqdm(total=len(parameterizations)*n_idxs*(1+len(ns)*len(cap_sizes)*query_repetitions)) as pbar:
				for i_param, params in enumerate(parameterizations):
					for i_idx in range(n_idxs):
						ef_const, cap_const = params
						with Timer("Build", print=False) as build_timer:
							# Build HNSW
							_dist = gib.SparseNegDotProduct()
							Xd, Xin, Xip = [v.astype(t, copy=False) for v,t in [
								[X.data, _float],
								[X.indices, _floatint],
								[X.indptr, _int],
							]]
							idx = gib.SparsePyHNSW(
								Xd, Xin, Xip,
								higher_max_degree=ef_const//2,
								lowest_max_degree=ef_const,
								distance=_dist.to_enum(),
								max_build_frontier_size=cap_const,
								insert_heuristic=False,
							)
							del Xd, Xin, Xip
						build_times[i_param,i_idx] = build_timer.delta_time
						pbar.update(1)
						for i_n, n in enumerate(ns):
							for i_cap_size, cap_size in enumerate(cap_sizes):
								idx.max_frontier_size = cap_size
								for i_rep in range(query_repetitions):
									with Timer("Query", print=False) as query_timer:
										Qd, Qin, Qip = [v.astype(t, copy=False) for v,t in [
											[Q.data, _float],
											[Q.indices, _floatint],
											[Q.indptr, _int],
										]]
										qresult = idx.knn_query_batch(Qd, Qin, Qip,k,n)
										del Qd, Qin, Qip
									query_times[i_param,i_idx,i_n,i_cap_size,i_rep] = query_timer.delta_time
									recalls[i_param,i_idx,i_n,i_cap_size,i_rep] = recall(true_nn_ids,qresult[0])
									pbar.update(1)
						del idx
						np.save(f"_sisap2026/_task3_hnswtune_{variant}_build_times_{data_variant}.npy", build_times)
						np.save(f"_sisap2026/_task3_hnswtune_{variant}_query_times_{data_variant}.npy", query_times)
						np.save(f"_sisap2026/_task3_hnswtune_{variant}_recalls_{data_variant}.npy", recalls)
		else:
			build_times = np.load(f"_sisap2026/_task3_hnswtune_{variant}_build_times_{data_variant}.npy")
			query_times = np.load(f"_sisap2026/_task3_hnswtune_{variant}_query_times_{data_variant}.npy")
			recalls = np.load(f"_sisap2026/_task3_hnswtune_{variant}_recalls_{data_variant}.npy")
