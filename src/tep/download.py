"""Download the public TEP simulation dataset and convert it to Parquet.

Source: Rieth, Amsel, Tran and Cook (2017), "Additional Tennessee Eastman
Process Simulation Data for Anomaly Detection Evaluation", Harvard
Dataverse, https://doi.org/10.7910/DVN/6C3JR1

File IDs and checksums are read from the Dataverse API at run time, every
download is verified against its published MD5, and any failure stops the
script. Nothing is substituted for missing data.

    python -m tep.download
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from . import config

API = "https://dataverse.harvard.edu/api"
USER_AGENT = "tep-fault-detection/0.1"
FILES = {
    "TEP_FaultFree_Training.RData": "fault_free_training",
    "TEP_Faulty_Training.RData": "faulty_training",
    "TEP_FaultFree_Testing.RData": "fault_free_testing",
    "TEP_Faulty_Testing.RData": "faulty_testing",
}
EXPECTED_ROWS = {
    "fault_free_training": 500 * 500,
    "faulty_training": 20 * 500 * 500,
    "fault_free_testing": 500 * 960,
    "faulty_testing": 20 * 500 * 960,
}


class DownloadError(RuntimeError):
    """The dataset could not be fetched or did not match what was published."""


def _open(url: str, timeout: int = 60):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=timeout)


def list_remote_files() -> dict[str, dict]:
    """File name -> {id, md5, size} for the dataset, as published on Dataverse."""
    url = f"{API}/datasets/:persistentId/?persistentId={config.DATASET_DOI}"
    try:
        with _open(url) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise DownloadError(f"could not read the dataset listing from {url}: {error}") from error
    if payload.get("status") != "OK":
        raise DownloadError(f"Dataverse returned status {payload.get('status')!r} for {config.DATASET_DOI}")
    listing = {}
    for entry in payload["data"]["latestVersion"]["files"]:
        data_file = entry["dataFile"]
        listing[data_file["filename"]] = {
            "id": data_file["id"], "md5": data_file.get("md5"), "size": data_file.get("filesize")}
    missing = sorted(set(FILES) - set(listing))
    if missing:
        raise DownloadError(f"dataset {config.DATASET_DOI} no longer lists: {', '.join(missing)}")
    return {name: listing[name] for name in FILES}


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name: str, remote: dict, raw_dir: Path) -> Path:
    """Download one file unless a verified copy is already on disk."""
    target = raw_dir / name
    if target.exists() and md5_of(target) == remote["md5"]:
        print(f"  {name}: already downloaded, checksum ok")
        return target

    url = f"{API}/access/datafile/{remote['id']}"
    partial = target.with_suffix(target.suffix + ".part")
    size_mb = (remote["size"] or 0) / 1e6
    print(f"  {name}: downloading {size_mb:,.0f} MB")
    try:
        with _open(url, timeout=120) as response, partial.open("wb") as handle:
            done, next_report = 0, 100 << 20
            while chunk := response.read(1 << 20):
                handle.write(chunk)
                done += len(chunk)
                if done >= next_report:
                    print(f"    {done / 1e6:,.0f} / {size_mb:,.0f} MB", flush=True)
                    next_report += 100 << 20
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
        partial.unlink(missing_ok=True)
        raise DownloadError(f"download of {name} from {url} failed: {error}") from error

    actual = md5_of(partial)
    if remote["md5"] and actual != remote["md5"]:
        partial.unlink()
        raise DownloadError(f"{name}: checksum mismatch (expected {remote['md5']}, got {actual})")
    partial.replace(target)
    return target


def convert(raw_path: Path, table: str, processed_dir: Path) -> Path:
    """RData -> Parquet with compact dtypes, in the file's original row order."""
    import pyreadr

    target = processed_dir / f"{table}.parquet"
    if target.exists():
        print(f"  {target.name}: already converted")
        return target
    print(f"  {raw_path.name} -> {target.name}")
    frames = pyreadr.read_r(str(raw_path))
    if len(frames) != 1:
        raise DownloadError(f"{raw_path.name}: expected one table, found {list(frames)}")
    frame = next(iter(frames.values()))

    expected_columns = config.RUN_KEYS + ["sample"] + config.SENSORS
    if list(frame.columns) != expected_columns:
        raise DownloadError(f"{raw_path.name}: unexpected columns {list(frame.columns)[:6]}...")
    if len(frame) != EXPECTED_ROWS[table]:
        raise DownloadError(f"{raw_path.name}: expected {EXPECTED_ROWS[table]:,} rows, found {len(frame):,}")

    for column in config.RUN_KEYS + ["sample"]:
        frame[column] = frame[column].astype("int16")
    for column in config.SENSORS:
        frame[column] = frame[column].astype("float32")
    # small row groups let the loader skip runs it does not need
    frame.to_parquet(target, index=False, row_group_size=48_000)
    return target


def download_all(data_dir: Path | None = None) -> list[Path]:
    data_dir = Path(data_dir or config.DATA_DIR)
    raw_dir, processed_dir = data_dir / "raw", data_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading file listing for {config.DATASET_DOI}")
    remote = list_remote_files()
    print("Downloading")
    raw_paths = {name: fetch(name, remote[name], raw_dir) for name in FILES}
    print("Converting to Parquet")
    return [convert(raw_paths[name], table, processed_dir) for name, table in FILES.items()]


def main() -> int:
    try:
        paths = download_all()
    except DownloadError as error:
        print(f"\nDOWNLOAD FAILED: {error}", file=sys.stderr)
        print("No data was substituted. Fix the problem and run `python -m tep.download` again.", file=sys.stderr)
        return 1
    print("\nReady:")
    for path in paths:
        print(f"  {path}  ({path.stat().st_size / 1e6:,.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
