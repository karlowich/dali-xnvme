#!/usr/bin/env python3

# Modified version of script found here: https://github.com/safl/xallib

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from joblib import Parallel, delayed
import re


def get_extents(path: Path):
    """The given path is expected to be absolute"""

    regex = (
        r"^\d+:\s+"
        r"\[\d+\.\.\d+\]:\s+"
        r"(?P<startblock>\d+)\.\.(?P<endblock>\d+)$"
    )

    cmd = ["xfs_bmap", path]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    if proc.returncode:
        print("Failed running: %s" % " ".join(cmd))
        return proc.returncode

    for line in proc.stdout.splitlines():
        match = re.match(regex, line.strip())
        if match:
            return (str(path), {k: int(v) for k,v in match.groupdict().items()})

    return ("", {})


def main(args):
    with args.output.open("w") as f:
        results = Parallel(n_jobs=24)(delayed(get_extents)(Path(root) / name) for root, _, files in os.walk(args.mountpoint) for name in files)
        result = {key : val for key, val in results}

        f.write(json.dumps(result, indent=2) + "\n")

    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract file extent-info from mounted XFS file-system"
    )

    parser.add_argument(
        "--mountpoint", type=Path, required=True, help="Path to the mountpoint."
    )
    parser.add_argument(
        "--output",
        default=Path.cwd() / "bmap.json",
        type=Path,
        help="Path to the output file or directory.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main(parse_args()))
