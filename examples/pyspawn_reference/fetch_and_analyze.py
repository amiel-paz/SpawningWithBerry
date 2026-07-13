"""Fetch the public PySpawn SI archive and reproduce its numerical observables."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from aims_berry.analysis import analyze_pyspawn_reference

URL = "https://ndownloader.figshare.com/files/24165942"
MD5 = "b0e99a061dcbdca57421d4f9d826b6d6"


def verify_summary(actual_path: Path) -> None:
    expected_path = Path(__file__).with_name("expected_metrics.json")
    expected = json.loads(expected_path.read_text())
    actual = json.loads(actual_path.read_text())
    failures = []
    for key, target in expected.items():
        value = actual[key]
        if isinstance(target, dict):
            for subkey, subtarget in target.items():
                if abs(value[subkey] - subtarget) > 1.0e-8:
                    failures.append(f"{key}.{subkey}: {value} != {subtarget}")
        elif abs(value - target) > 1.0e-8:
            failures.append(f"{key}: {value} != {target}")
    if failures:
        raise RuntimeError("reference metrics changed:\n" + "\n".join(failures))


def main(output: str = "reference-output") -> None:
    target = Path(output).resolve()
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "ct0c00575_si_001.zip"
    if not archive.exists():
        with urllib.request.urlopen(URL) as response, archive.open("wb") as stream:
            shutil.copyfileobj(response, stream)
    digest = hashlib.md5(archive.read_bytes()).hexdigest()  # noqa: S324 - published checksum
    if digest != MD5:
        raise RuntimeError(f"supporting archive checksum mismatch: {digest}")
    with zipfile.ZipFile(archive) as zipped:
        zipped.extract("sim.hdf5", target)
    for product in analyze_pyspawn_reference(target / "sim.hdf5", target / "analysis"):
        print(product)
    verify_summary(target / "analysis" / "reference_summary.json")
    print("published supporting-data metrics verified")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "reference-output")
