# User guide — how to use everything in this repository

This guide assumes no prior knowledge of the code. It covers installation, every command, what
each one prints when it works, how to publish the repository, and what to do when something fails.

## 1. What you need

| Item | Detail |
| --- | --- |
| Python | Version 3.13 (the version the results were produced with) |
| Memory and time | 4 GB of RAM is ample (measured peak under 1 GB); about 9 minutes on 2 CPU cores for the main command. No GPU |
| Disk | About 1 GB free (the data bundle is about 400 MB, the libraries about 500 MB) |
| The data bundle | The gated match data from the organisers. It is **not** in this repository and must never be added to it |
| Git | Only needed to download the training matches (section 6) and to publish (section 8) |

## 2. Install

Open a terminal in the repository folder.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` pins the four libraries the code uses (numpy, pandas, scipy, lightgbm) to exact
versions. Do not upgrade them: different versions can change the output.

## 3. Put the data in place

Create a folder named `data` inside the repository and unpack the bundle into it, keeping the
organisers' layout:

```
data/
├── tactical/
│   ├── BAR-ATH_tactical_atd.txt            required
│   ├── BAR-ATH_tactical_atd.xml            required
│   └── BAR-ATH_tactical_smart_tagging.xml  required for the kick-off rule
└── broadcast/
    ├── BAR-ATH_broadcast_atd.txt           required to reproduce the submitted file
    └── BAR-ATH_broadcast_atd.xml           required to reproduce the submitted file
```

The homography files and the videos are not read; they may be present or absent.

If the bundle lives somewhere else, point to it instead of copying it:

```bash
export MTAH_DATA_DIR=/path/to/bundle          # macOS / Linux
$env:MTAH_DATA_DIR = "D:\path\to\bundle"      # Windows PowerShell
```

The `data` folder is listed in `.gitignore`, so Git will not upload it.

## 4. Regenerate the submission — `run.py`

```bash
python run.py
```

This is the command that produces the submitted file. It reads the bundle, aligns the two cameras,
labels every frame with the model in `possession_lgbm.txt`, and writes
`possession.csv`.

What it prints when it works:

```
aligning the broadcast tracking to the tactical timeline ...
  lag candidate 4012 -> refined 4014 (median player distance 0.29 m, 29 blocks)
  lag candidate 26922 -> refined 26920 (median player distance 0.30 m, 29 blocks)
  broadcast frames with a ball: 83121, accepted: 76552, used to fill tactical gaps: 5981
half 1: frames 11185..90651, 0 frames set to dead ball by the opening kick-off rules
half 2: frames 110808..190197, 1044 frames set to dead ball by the opening kick-off rules
wrote .../possession.csv
sha256 97cb9940a32764f55465d97ad92d0007b21601a61eac8ba0283067d134d21acc
graded frames 158857: H 0.329  A 0.205  D 0.467  X 0.000
frames outside the two halves: 40462, of which X: 40462
half 1: 147 possessions (H 76, A 71), median 6.2 s, shortest 0.56 s, longest 59.4 s
half 2: 179 possessions (H 84, A 95), median 4.8 s, shortest 0.32 s, longest 57.0 s
```

**The line to check is `sha256`.** It is a fingerprint of the file. If it reads
`97cb9940…d21acc`, the file is identical, byte for byte, to the one submitted.

Options:

| Option | Meaning | Default |
| --- | --- | --- |
| `--data-dir PATH` | Where the bundle is | `./data`, or `MTAH_DATA_DIR` if set |
| `--out PATH` | Where to write the CSV | `possession.csv` |
| `--model PATH` | Which model file to use | `possession_lgbm.txt` |
| `--cache-dir PATH` | Folder to keep parsed tracking between runs; a second run then takes about 2 minutes instead of 9 | none |

If you use `--cache-dir`, choose a folder named `cache` (already git-ignored) or one outside the
repository: the cache contains tracking data and must not be published.

## 5. Check a submission file — `check_submission.py` and `diagnostics.py`

**Format check.** Confirms 199,319 rows, frames in order, labels H/A/D/X, and prints the counts.

```bash
python check_submission.py possession.csv
```

**Score against a ground-truth file.** If you hold a `frame,label` ground truth (for example the
organisers' public 5-minute window), add `--gt`:

```bash
python check_submission.py possession.csv --gt path/to/ground_truth.csv
```

It prints spellF1 and macroF1 computed with this repository's re-implementation of the metrics.
The organisers' `evaluation/run_evaluation.py` is the authority; use theirs for anything official.
The rules allow the public window to be used this way only — to confirm the pipeline is aligned —
never to tune or train.

**Sanity checks without any ground truth.** Compares the labels with the automatic data: player
speed in each state, relation to the ball track, agreement with the Smart Tagging.

```bash
python diagnostics.py possession.csv
```

Expected: players move at about 2.3 m/s when the ball is labelled in play and 1.1 m/s when dead;
agreement with "BUILD UP" windows about 95% and with "PROGRESSION" windows about 89%.

## 6. Rebuild the model — `train.py` (optional)

You do not need this to regenerate the submission: the trained model (`possession_lgbm.txt`) is already in the repository. Run
it to verify that the model comes from the public training matches and from nothing else.

```bash
bash get_sample_data.sh      # downloads the three training matches into external/
python train.py                      # about 10 minutes
```

On Windows without Bash, the script is equivalent to:

```powershell
git clone https://github.com/metrica-sports/sample-data.git external/sample-data
git -C external/sample-data checkout e706dd506b360d69d9d123d5b8026e7294b13996
```

`train.py` overwrites `possession_lgbm.txt`. On the machine the results were produced on,
the rebuilt file is identical to the committed one (sha256 `e90a1833…33297a`). On a different
operating system or LightGBM build the trees can differ slightly; that is why the model file is
committed. To compare without overwriting, write to another path:

```bash
python train.py --model /tmp/model_check.txt
```

## 7. Reproduce the validation numbers — `validate.py` (optional)

```bash
python validate.py                   # about 20 minutes; needs the training matches from section 6
```

For each of the three training matches it trains on the other two, tests on the third, and prints
the scores. Expected last line:

```
mean decoded spellF1 0.7564  macroF1 0.9272
```

This uses no data from the challenge match.

## 8. Publish the repository

1. Create an empty **public** repository on GitHub (no README, no licence, so it starts empty).
2. In the repository folder:

```bash
git init
git add .
git status          # read this list before going on
git commit -m "Possession challenge submission"
git branch -M main
git remote add origin https://github.com/<account>/<repository>.git
git push -u origin main
```

**Before `git commit`, read the output of `git status`.** Nothing under `data/`, `cache/` or
`external/` may appear in it. Expect 28 files, all in one folder; the two largest are the model (about 10 MB)
and the submission (about 1.7 MB). If a tracking file, a video or a homography file is listed,
stop and remove it with `git rm --cached <file>`.

Without Git, in the browser: on the repository page click **Add file**, then **Upload files**,
drag in every file of this folder, and click **Commit changes**. All files go at the top level;
the code does not need sub-folders. A file uploaded under an existing name replaces it.

3. Put the repository's URL in the first lines of `TECHNICAL_DOCUMENT.md` and in the
   submission form.

## 9. If something goes wrong

| What you see | Cause | What to do |
| --- | --- | --- |
| `FileNotFoundError` naming a `BAR-ATH_…` file | The bundle is not where the code looks | Check the folder layout in section 3, or set `MTAH_DATA_DIR` |
| `ModuleNotFoundError: No module named 'lightgbm'` (or numpy, pandas, scipy) | The libraries are not installed in the active environment | Activate `.venv` and run `pip install -r requirements.txt` again |
| `AssertionError: feature set does not match the model` | The code in `features.py` was changed after the model was trained | Restore the file, or run `train.py` to rebuild the model |
| The `sha256` line differs from `97cb9940…` | Different library versions, a different or incomplete bundle, a missing broadcast folder, or a retrained model | Check `pip list` against `requirements.txt`; check that both `tactical/` and `broadcast/` are present; restore `possession_lgbm.txt` from Git |
| The run is killed or the computer freezes | Not enough memory | Close other programs. The main command peaks under 1 GB; training and validation need more and were run on a machine with 7 GB |
| `Permission denied` on `get_sample_data.sh` | The file is not marked executable | Run it as `bash get_sample_data.sh` |

## 10. What each file is

| Path | Role |
| --- | --- |
| `run.py` | Regenerates the submission |
| `train.py` | Rebuilds the model from the training matches |
| `validate.py` | Leave-one-game-out validation |
| `check_submission.py` | Format check and optional scoring |
| `diagnostics.py` | Sanity checks without ground truth |
| `requirements.txt` | Pinned library versions |
| `get_sample_data.sh` | Downloads the training matches at a fixed revision |
| `possession_lgbm.txt` | The trained classifier |
| `possession.csv` | The submitted file |
| `TECHNICAL_DOCUMENT.md`, `.pdf` | The technical document |
| `USER_GUIDE.md`, `.pdf` | This guide |
| `pipeline.py` | Runs every step in order |
| `match.py` | Match constants and graded frames |
| `io_bundle.py` | Reads the challenge files |
| `align.py` | Aligns the two cameras and merges the ball tracks |
| `features.py` | Builds the 342 features per frame |
| `model.py` | Classifier settings and prediction |
| `decode.py` | Turns frame probabilities into possessions |
| `kickoff.py` | Opening kick-off rules |
| `io_metrica.py` | Reads the training matches and derives their labels |
| `simulate.py` | Makes the training matches resemble automatic tracking |
| `dataset.py` | Builds the training table |
| `metrics.py` | macroF1 and spellF1, re-implemented from EVALUATION.md |
| `check.py` | Helpers used by the two checking commands |
