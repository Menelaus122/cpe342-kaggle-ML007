"""Assemble CPE342_Master_Pipeline.ipynb from the cell definitions in nb_part*.py.

    python scripts/build_notebook.py            # write the notebook
    jupyter nbconvert --to notebook --execute --inplace CPE342_Master_Pipeline.ipynb
"""
import importlib.util
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load_cells(part):
    spec = importlib.util.spec_from_file_location(part, HERE / f"{part}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CELLS


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["nb_part1", "nb_part2"])
    ap.add_argument("--out", default=str(ROOT / "CPE342_Master_Pipeline.ipynb"))
    ap.add_argument("--reuse-outputs", action="store_true",
                    help="keep outputs of code cells whose source is unchanged in the existing --out notebook "
                         "(lets markdown edits land without re-executing)")
    args = ap.parse_args()
    old_outputs = {}
    if args.reuse_outputs and Path(args.out).exists():
        old = nbformat.read(args.out, as_version=4)
        old_outputs = {c.source: (c.outputs, c.execution_count) for c in old.cells if c.cell_type == "code"}
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "cpe342", "display_name": "Python (cpe342 venv)", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    for part in args.parts:
        for kind, src in load_cells(part):
            if kind == "markdown":
                nb.cells.append(nbformat.v4.new_markdown_cell(src))
                continue
            cell = nbformat.v4.new_code_cell(src)
            if src in old_outputs:
                cell.outputs, cell.execution_count = old_outputs[src]
            elif args.reuse_outputs:
                print("WARNING: code changed, no outputs for cell starting:", src[:60].replace("\n", " "))
            nb.cells.append(cell)
    out = Path(args.out)
    nbformat.write(nb, out)
    print(f"wrote {out} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
