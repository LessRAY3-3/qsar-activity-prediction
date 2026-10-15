"""Package the complete fixed-parameter RF cohort, inputs and logs for replay."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    archive = args.archive.resolve()
    if archive.exists() or archive.is_relative_to(root):
        parser.error("archive must be new and outside run directory")
    inputs = json.loads((root / "input_manifest.json").read_text())
    summary = json.loads((root / "output/summary.json").read_text())
    if summary["status"] != "complete" or len(summary["jobs"]) != 16:
        raise ValueError("incomplete RF cohort")
    names = {**inputs["files"]}
    for name in ("input_manifest.json", "training.log", "launch.json", "output/summary.json", "output/environment.json"):
        names[name] = sha(root / name)
    for row in summary["jobs"]:
        folder = root / "output" / row["job_id"]
        rel = folder.relative_to(root).as_posix()
        if json.loads((folder / "receipt.json").read_text()) != row:
            raise ValueError("receipt/summary mismatch")
        names[rel + "/receipt.json"] = sha(folder / "receipt.json")
        names.update({rel + "/" + name: digest for name, digest in row["output_sha256"].items()})
    for name, digest in names.items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root) or sha(path) != digest:
            raise ValueError(f"unsafe or changed artifact: {name}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "x:gz", compresslevel=3) as tar:
        for name in sorted(names):
            tar.add(root / name, arcname=name, recursive=False)
    manifest = {"sha256": sha(archive), "bytes": archive.stat().st_size,
                "packer_sha256": sha(Path(__file__)),
                "files": {name: {"sha256": digest, "bytes": (root/name).stat().st_size}
                          for name, digest in sorted(names.items())}}
    # Recheck after packaging so a changing log/model cannot get accepted.
    if any(sha(root / name) != digest for name, digest in names.items()):
        raise ValueError("artifact changed while packaging")
    path = Path(str(archive)+".manifest.json")
    with path.open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"archive": str(archive), "sha256": manifest["sha256"],
                      "bytes": manifest["bytes"], "files": len(names)}))


if __name__ == "__main__":
    main()
