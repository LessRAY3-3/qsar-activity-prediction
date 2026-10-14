"""Remove ten acceptance protections in isolated copies and check real artifacts.

These are disclosed, targeted mutations, not the unpublished sixteen mutants
mentioned in PR #8. A mutant is caught only if the real-artifact negative test
reports DID NOT RAISE (the damaged artifact was accepted). Fixture, import and
training failures are inconclusive, never counted as caught mutations.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

CASES = {
    "checkpoint_label": ("label", ("GIN checkpoint label_id",), 1),
    "checkpoint_arguments": ("seed", (), 1),
    "finite_weights": ("nan", ("GIN checkpoint has empty or invalid weights",), 1),
    "architecture_loading": ("shape", (), 1),
    "normalization": ("bad_mean", ("GIN checkpoint has invalid", "GIN checkpoint label scaling",
                                   "GIN checkpoint normalization"), 3),
    "prediction_label": ("prediction_version", (), 1),
    "metric_recomputation": ("metric", ("Stored ",), 1),
    "runtime_device": ("runtime", ("GIN checkpoint runtime differs",), 2),
    "metric_label": ("metric_label", ("Metric label_id differs",), 1),
    "rf_label": (None, ("RF checkpoint label_id or seed differs",), 1),
}
TEST = "tests/test_time_campaign_real.py"
WORKER = "scripts/run_time_campaign.py"


def mutate(source, case):
    """Delete only the named protection in validate_outputs, refusing drift."""
    _, messages, expected = CASES[case]

    class Remove(ast.NodeTransformer):
        count = 0

        def visit_FunctionDef(self, node):
            return self.generic_visit(node) if node.name == "validate_outputs" else node

        def drop(self, node):
            self.count += 1
            return ast.copy_location(ast.Pass(), node)

        def visit_If(self, node):
            for statement in node.body:
                if isinstance(statement, ast.Raise):
                    strings = [n.value for n in ast.walk(statement)
                               if isinstance(n, ast.Constant) and isinstance(n.value, str)]
                    if any(text.startswith(prefix) for text in strings for prefix in messages):
                        return self.drop(node)
            return self.generic_visit(node)

        def visit_For(self, node):
            if case == "checkpoint_arguments" and ast.unparse(node.iter) == "expected_args.items()":
                return self.drop(node)
            return self.generic_visit(node)

        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call):
                name = ast.unparse(node.value.func)
                if ((case == "architecture_loading" and name == "model.load_state_dict") or
                        (case == "prediction_label" and name == "require_prediction_version")):
                    return self.drop(node)
            return self.generic_visit(node)

    editor = Remove()
    tree = editor.visit(ast.parse(source))
    if editor.count != expected:
        raise ValueError(f"{case}: expected {expected} guard edits, found {editor.count}; review source drift")
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = (root / WORKER).read_text()
    # Prepare every mutation before running tests; a stale anchor fails clearly.
    mutations = {name: mutate(source, name) for name in CASES}
    rows = []
    for name, modified in [("baseline", source), *mutations.items()]:
        with tempfile.TemporaryDirectory(prefix="qsar-tooling-mutant-") as temporary:
            work = Path(temporary)
            shutil.copytree(root / "scripts", work / "scripts",
                            ignore=shutil.ignore_patterns("__pycache__", "campaign_history"))
            (work / "tests").mkdir()
            for relative in (TEST, "tests/conftest.py", "pyproject.toml"):
                shutil.copy2(root / relative, work / relative)
            (work / WORKER).write_text(modified)
            test = TEST
            if name != "baseline":
                corruption = CASES[name][0]
                test += ("::test_real_fitted_rf_is_checked_and_wrong_generation_rejected" if corruption is None
                         else f"::test_real_gin_checkpoint_corruption_rejected[{corruption}]")
            junit = output / f"{name}.xml"
            with (output / f"{name}.log").open("w") as log:
                result = subprocess.run([sys.executable, "-m", "pytest", "-q", test,
                                         f"--junitxml={junit}"], cwd=work, stdout=log,
                                        stderr=subprocess.STDOUT, check=False)
            xml = ET.parse(junit)
            failures, errors = xml.findall(".//failure"), xml.findall(".//error")
            killed = result.returncode == 1 and len(failures) == 1 and not errors and "DID NOT RAISE" in failures[0].get("message", "")
            status = ("passed" if result.returncode == 0 else "failed") if name == "baseline" else (
                "caught" if killed else "survived" if result.returncode == 0 else "inconclusive")
            rows.append({"case": name, "status": status, "returncode": result.returncode,
                         "mutated_source_sha256": hashlib.sha256(modified.encode()).hexdigest(),
                         "test": test})
            print(f"{name}: {status}", flush=True)
            if name == "baseline" and status != "passed":
                break
    report = {"scope": "Ten specified acceptance mutations; not a reproduction of the original undisclosed sixteen",
              "worker_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "test_sha256": hashlib.sha256((root / TEST).read_bytes()).hexdigest(),
              "rows": rows, "all_caught": len(rows) == 11 and rows[0]["status"] == "passed"
              and all(row["status"] == "caught" for row in rows[1:])}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.root, args.out_dir)
    raise SystemExit(0 if report["all_caught"] else 1)


if __name__ == "__main__":
    main()
