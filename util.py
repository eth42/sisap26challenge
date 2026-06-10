import time
# import psutil
import numpy as np
import scipy
import os

import graphidxbaselines as gib
import hiob

class Timer():
	def __init__(self, label=None, print=True):
		self.label=label
		self.delta_proc = None
		self.delta_time = None
		self.print = print
	def _format_time(self, t):
		if t < 1: return f"{int(t*1000):d}ms"
		elif t < 60: return f"{t:.03f}s"
		elif t < 3600: return f"{int(t/60):d}:{t%60:06.03f}"
		else: return f"{int(t/3600):d}:{int(t/60)%60:02d}:{t%60:06.03f}"
	def __enter__(self, *args, **kwargs):
		self.proctime = time.process_time()
		self.time = time.time()
		return self
	def __exit__(self, *args, **kwargs):
		delta_proc = time.process_time() - self.proctime
		delta_time = time.time() - self.time
		self.delta_proc = delta_proc
		self.delta_time = delta_time
		parts = []
		if self.label is not None: parts.append(self.label+":")
		parts.append(self._format_time(delta_time))
		parts.append(f"({self._format_time(delta_proc)})")
		if self.print: print(*parts, flush=True)

# def get_current_ram_gb():
# 	process = psutil.Process()
# 	return process.memory_info().rss / (1<<30)
def recall(true,pred):
	k = min(true.shape[1],pred.shape[1])
	true,pred = [v[:,:k].astype(np.uint64) for v in [true,pred]]
	return hiob.RawBinarizationEvaluator().k_at_n_recall_prec_all(true,pred)
def normalize(X):
	if type(X) == np.ndarray:
		return X/np.linalg.norm(X,axis=-1,keepdims=1)
	else:
		return (X/np.sqrt(X.multiply(X).sum(1))).tocsr()
def append_norm(X):
	if type(X) == np.ndarray:
		return np.concatenate([X,np.linalg.norm(X,axis=-1,keepdims=1)],axis=-1)
	else:
		return scipy.sparse.hstack([
			X,
			np.sqrt(X.multiply(X).sum(1))
		]).tocsr()
def append_zero(X):
	if type(X) == np.ndarray:
		return np.concatenate([X,np.zeros((X.shape[0],1),dtype=X.dtype)],axis=-1)
	else:
		return X.tocsr()

def get_gib_primitives():
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
	return _int, _uint, _float, _floatint, _floatuint
