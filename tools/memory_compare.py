"""Compare cold CLI peak RSS for identical real-model logits requests on macOS.

Usage: python tools/memory_compare.py MODEL BASELINE_BINARY CURRENT_BINARY
The baseline and current binaries must be distinct, immutable snapshots.
"""
import argparse
import hashlib
import json
import math
import os
import pathlib
import signal
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model",type=pathlib.Path)
    parser.add_argument("baseline",type=pathlib.Path)
    parser.add_argument("current",type=pathlib.Path)
    parser.add_argument("--tokens",type=int,default=1025)
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("this measurement uses macOS /usr/bin/time -l")
    if not 1 <= args.tokens <= 8192:
        parser.error("tokens must be 1..8192")
    binaries = [args.baseline.resolve(),args.current.resolve()]
    hashes = [sha(p) for p in binaries]
    assert hashes[0] != hashes[1], "refuse to compare the same executable twice"
    model_hash = sha(args.model)
    text = "<think>" * (args.tokens-1)
    rows, values = [],[]
    for name,binary,identity in zip(["baseline","current"],binaries,hashes):
        start = time.perf_counter()
        process = subprocess.Popen(["/usr/bin/time","-l",str(binary),"logits",str(args.model.resolve()),text],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
        try:
            stdout,stderr = process.communicate(timeout=120)
        except BaseException:
            try:
                os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise
        assert process.returncode == 0,stderr
        assert sha(binary) == identity, "executable changed during measurement"
        value = json.loads(stdout)
        assert len(value["ids"]) == args.tokens
        assert all(math.isfinite(x) for x in value["logits"])
        rss = [int(line.split()[0]) for line in stderr.splitlines()
               if line.strip().endswith("maximum resident set size")]
        assert len(rss) == 1,stderr
        row = {"binary":name,"binary_sha256":identity,"tokens":args.tokens,
               "seconds":time.perf_counter()-start,"maximum_resident_bytes":rss[0],"raw_time":stderr}
        rows.append(row);values.append(value)
        print(json.dumps(row),flush=True)
    assert values[0]["ids"] == values[1]["ids"]
    a,b = values[0]["logits"],values[1]["logits"]
    cosine = sum(x*y for x,y in zip(a,b))/math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
    assert cosine > 0.999
    assert max(range(len(a)),key=a.__getitem__) == max(range(len(b)),key=b.__getitem__)
    assert sha(args.model) == model_hash
    print(json.dumps({"model_sha256":model_hash,"cosine":cosine,
        "peak_reduction_bytes":rows[0]["maximum_resident_bytes"]-rows[1]["maximum_resident_bytes"],
        "peak_reduction_fraction":1-rows[1]["maximum_resident_bytes"]/rows[0]["maximum_resident_bytes"]}),flush=True)


if __name__ == "__main__":
    main()
