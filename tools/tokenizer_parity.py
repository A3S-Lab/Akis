"""Exact tokenizer differential against unmodified pinned upstream RefTokenizer.

Usage: python tools/tokenizer_parity.py MODEL REFERENCE_DIRECTORY AKIS_BINARY
"""
import hashlib
import importlib
import json
import pathlib
import random
import struct
import subprocess
import sys
import time
import types

model, reference, binary = sys.argv[1:]
package = types.ModuleType("needle_reference")
package.__path__ = [str(pathlib.Path(reference).resolve())]
sys.modules[package.__name__] = package
export = importlib.import_module("needle_reference.export")
data = pathlib.Path(model).read_bytes()
header = struct.unpack_from("<48If", data)
record = struct.unpack_from("<BBHIIIIQQII", data, 196 + header[2]*4 + (header[1]-1)*44)
assert record[0] == 4
tok = export.RefTokenizer(export.parse_tokenizer_blob(data[record[7]:record[7]+record[8]]))
randomizer = random.Random(20261013)
atoms = ["hello", "你好", "🦀", "e\u0301", "é", "\n", "\t", " ", "▁", "<tool_call>",
         "</tool_call>", "<think>", "</think>", "<|im_start|>", "\u0001", "مرحبا", "123", "\\", '"']
cases = ["", *atoms]
cases.extend("".join(randomizer.choices(atoms, k=randomizer.randrange(1,20))) for _ in range(100))
result = []
for index, text in enumerate(cases):
    actual = json.loads(subprocess.check_output([binary,"tokenize",model,text],text=True,timeout=10))
    ids = tok.encode(text)
    assert actual["ids"] == ids, (index,text,actual["ids"],ids)
    assert actual["text"] == tok.decode(ids), (index,text,actual["text"],tok.decode(ids))
    result.append({"case":index,"tokens":len(ids),"ids_sha256":hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
print(json.dumps({"passed":len(result),"comparisons":result}),flush=True)

# A bounded latency regression, including process startup and archive validation.
# The previous quadratic implementation takes about 10 seconds for this input.
start = time.perf_counter()
try:
    actual = json.loads(subprocess.check_output([binary,"tokenize",model,"hello "*4000],text=True,timeout=2))
except subprocess.TimeoutExpired:
    raise SystemExit("hello-times-4000 exceeded the 2-second regression budget") from None
print(json.dumps({"performance_case":"hello-times-4000","seconds":time.perf_counter()-start,
                  "tokens":len(actual["ids"]),"ids_sha256":hashlib.sha256(json.dumps(actual["ids"]).encode()).hexdigest()}),flush=True)
