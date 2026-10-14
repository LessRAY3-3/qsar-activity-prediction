"""Package a completed returned study, deduplicating identical bytes with tar links.

The archive contains the returned study, not the graph/split input bundle.
This checks completion and file integrity; model replay is a separate audit.
Uses a new output file and never changes the study directory.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile


def package(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("Archive must be outside the returned study")
    status = json.loads((root / "study_status.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    jobs = list(root.glob("*/lane*/results/time_campaign/*/status.json"))
    if (status["state"] != "training_complete_pending_independent_audit"
            or len(jobs) != protocol["jobs"]
            or any(json.loads(p.read_text())["state"] != "success" for p in jobs)
            or list(root.rglob(".time_campaign_*.lock"))):
        raise ValueError("Study is incomplete or still locked")
    paths = sorted(p for p in root.rglob("*") if not p.is_dir())
    hashes, identical, links = {}, {}, 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as raw, gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as archive:
            for path in paths:
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"Nonregular study file: {path}")
                name = "study/" + path.relative_to(root).as_posix()
                data = path.read_bytes()
                sha = hashlib.sha256(data).hexdigest()
                hashes[name] = sha
                info = tarfile.TarInfo(name)
                info.mode = 0o644
                if sha in identical:
                    info.type, info.linkname = tarfile.LNKTYPE, identical[sha]
                    archive.addfile(info)
                    links += 1
                else:
                    import io
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
                    identical[sha] = name
    return {"archive": output.name, "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "bytes": output.stat().st_size, "files": len(paths), "hardlinks": links,
            "member_sha256": hashes,
            "note": "Tar hardlinks reuse identical canonical/snapshot bytes. Extract with a data-only tar filter. Input graphs/splits are supplied separately."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = package(args.study_root, args.output)
    manifest = args.output.with_suffix(args.output.suffix + ".manifest.json")
    with manifest.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "member_sha256"}))


if __name__ == "__main__":
    main()
