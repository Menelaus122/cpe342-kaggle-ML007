# experiments/

Personal sandbox. Each teammate works in their own sub-folder (`experiments/<github-username>/`), so two people
never edit the same scratch notebook.

* Anything goes here: quick notebooks, scripts, ablations, plots.
* Code and notebooks are committed; generated data (`*.csv`, `*.npy`, `*.npz`, `outputs/`) is git-ignored.
* Nothing in the main pipeline imports from `experiments/`.
* When an idea works, move it into `src/` (and the notebook cells in `scripts/nb_part*.py`) through a pull request,
  with its out-of-fold score in the PR description.

To make the shared code importable from a notebook in `experiments/<you>/`:

```python
import os, sys
from pathlib import Path
ROOT = Path.cwd().parents[1]          # experiments/<you>/ -> project root
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src import config as C, task1    # etc.
```
