"""Offline Needle 3 native ABI baseline. Never loads the SDK or sends telemetry.

Usage: python tools/native_baseline.py LIBRARY MODEL
Run only a trusted, pinned native library on its matching architecture.
"""
import ctypes
import hashlib
import json
import pathlib
import sys
import time

from live_e2e import ROOT, TOOL, MODEL_SHA

library, model = sys.argv[1:]
weights = pathlib.Path(model).read_bytes()
assert hashlib.sha256(weights).hexdigest() == MODEL_SHA
lib = ctypes.CDLL(str(pathlib.Path(library).resolve()))
lib.needle_load.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
lib.needle_load.restype = ctypes.c_int
lib.needle_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
lib.needle_init.restype = ctypes.c_int
lib.needle_complete.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
                               ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
lib.needle_complete.restype = ctypes.c_int
lib.needle_reset.argtypes = []
lib.needle_reset.restype = None
assert lib.needle_load(weights, len(weights)) >= 0
assert lib.needle_init(b"", json.dumps([TOOL]).encode(), None) >= 0
output = ctypes.create_string_buffer(65536)
for fixture in json.loads((ROOT / "tests/e2e/cases/tool_calls.json").read_text()):
    lib.needle_reset()
    start = time.perf_counter()
    status = lib.needle_complete(fixture["input"].encode(), None, 0, 128, output, len(output))
    assert status >= 0, status
    value = json.loads(output.value)
    print(json.dumps({"id": fixture["id"], "seconds": time.perf_counter()-start,
                      "response": value}, ensure_ascii=False), flush=True)
print(json.dumps({"library_sha256": hashlib.sha256(pathlib.Path(library).read_bytes()).hexdigest(),
                  "model_sha256": MODEL_SHA}), flush=True)
