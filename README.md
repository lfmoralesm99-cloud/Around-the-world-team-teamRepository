# Ball possession from automatic tracking — BAR-ATH challenge

One label per frame of the tactical video — `H` home in possession, `A` away in possession,
`D` dead ball, `X` no answer — produced by an algorithm from the provided automatic tracking.

| | |
| --- | --- |
| Submission | [`possession.csv`](possession.csv) — 199,319 rows, `frame,label` |
| Regenerate it | `python run.py` (byte-identical; sha256 `97cb9940…d21acc`) |
| Validation (other matches, simulated tracking noise) | spellF1 **0.756**, macroF1 **0.927** (leave-one-game-out, 3 games) |
| Technical document | [`TECHNICAL_DOCUMENT.md`](TECHNICAL_DOCUMENT.md) |
| Step-by-step guide to every command | [`USER_GUIDE.md`](USER_GUIDE.md) |

## How it works

1. **Read** both tracking sources (FIFA EPTS). Row *i* of the tactical file is frame *i* of the tactical video.
2. **Align and fuse** (`align.py`). The broadcast tracking is delivered unaligned; the lag to the
   tactical timeline is recovered from the data (players seen by both cameras coincide at the right lag),
   replays are rejected by the same test, and the broadcast ball fills holes of the tactical ball.
3. **Features** (`features.py`). 342 per frame, identity-free: ball position, speed and margin to
   the lines; holes in the ball track; evidence that play stopped; stretches the tracker bridged with a
   straight line; distance from the ball to each team; *kicks* (sharp changes of ball
   velocity) and who made the last and next ones; team shape and speed; all with context windows before and
   after the frame. Each frame is seen twice, once from each team's side, so the model is symmetric.
4. **Classifier** (`model.py`). LightGBM, 3 classes (us / them / dead), trained **only on other
   matches**: the three open [Metrica Sports sample games](https://github.com/metrica-sports/sample-data),
   whose event data gives per-frame labels by code (`io_metrica.py`).
5. **Noise simulation** (`simulate.py`). The sample games are corrected data; the challenge
   tracking is automatic. Before training, each game is degraded until label-free statistics of its ball
   track match the challenge tracking: long holes, ball visible during stoppages, a low-pass-filtered track
   that sits about a metre off the player, gaps bridged by straight lines, a ball carried around during
   stoppages, a ball in the air seen metres from where it is, fragmented player tracks.
6. **Decoder** (`decode.py`). Viterbi with a price on every change of state, then a clean-up of
   spells too short to be controlled possession and stoppages too short to be stoppages. The ranking metric
   scores possessions as objects; this step is worth +0.20 spellF1 in validation.
7. **Opening kick-off** (`kickoff.py`). The graded block of the second half starts two minutes
   before the kick-off. Frames before the automatic "Kick Off (Start)" tag are dead ball, and a possession
   that is over while both teams are still lined up in their own halves is removed.

Frames outside the two graded blocks are `X`. Inside them the system always answers.

## Run it

```bash
pip install -r requirements.txt               # tested with Python 3.13 on Linux

# the match bundle is gated and is NOT in this repository: unpack it into ./data
#   data/tactical/BAR-ATH_tactical_atd.{txt,xml}, ..._smart_tagging.xml
#   data/broadcast/BAR-ATH_broadcast_atd.{txt,xml}        (optional but used)
# or point MTAH_DATA_DIR at it.

python run.py                                  # ~9 min on 2 cores -> possession.csv
python check_submission.py possession.csv
python diagnostics.py possession.csv   # label-free sanity checks
```

To rebuild the model and the validation numbers from scratch (no challenge data involved):

```bash
bash get_sample_data.sh                # clones the open sample games at a pinned revision
python train.py                                # ~10 min -> possession_lgbm.txt (same bytes)
python validate.py                             # ~20 min -> the table below
```

## Validation

Leave-one-game-out on the sample games; the held-out game is degraded with an unseen seed. Metrics are
re-implemented from EVALUATION.md in `metrics.py`; the organisers' scorer is the authority.

| Held-out game | Frame argmax spellF1 / macroF1 | Decoded spellF1 / macroF1 | Possessions predicted / true |
| --- | --- | --- | --- |
| Sample Game 1 | 0.615 / 0.899 | 0.771 / 0.898 | 314 / 350 |
| Sample Game 2 | 0.539 / 0.939 | 0.752 / 0.940 | 299 / 324 |
| Sample Game 3 | 0.526 / 0.943 | 0.746 / 0.943 | 347 / 381 |
| **Mean** | 0.560 / 0.927 | **0.756 / 0.927** | |

These are scores on other matches with simulated noise, not a forecast of the leaderboard.

Label-free checks on the challenge match (`diagnostics.py`): H / A / D = 32.9 / 20.5 / 46.7 % of graded
frames; 326 possessions (3.1 per minute); players move at 2.27 m/s when the ball is labelled in play and
1.05 m/s when dead; the team in possession agrees with the automatic Smart Tagging in 95 % of "BUILD UP" and
89 % of "PROGRESSION" window time; the first possession of each half starts at video frames 11185 and 113741.

## Rules

- Labels are produced by code; nothing on the challenge match was labelled or corrected by hand.
- No ground truth of the challenge match (private or public) and no human-produced data about it were used to
  train, tune or select anything. The public 5-minute window may be used with `check_submission.py --gt` to
  confirm the pipeline is aligned — never to tune.
- The only hand-made labels are Metrica's public event data on three **other** matches (pinned revision
  `e706dd5`), fetched by `get_sample_data.sh`.
- No frame-specific constants: the only frame numbers in the code are the two graded blocks, computed from
  the organisers' `games.py`.
- The match data is not in this repository (`data/`, `cache/`, `external/` are git-ignored), and no derived
  data that could reconstruct it is published.

## Layout

All files sit in one folder; nothing depends on sub-folders.

```
run.py                 regenerate the submission
train.py               rebuild the model from the open sample games
validate.py            leave-one-game-out validation
check_submission.py    format check, counts, optional scoring against a frame,label file
diagnostics.py         label-free checks against the automatic data
pipeline.py            runs every step in order
match.py, io_bundle.py, align.py, features.py, model.py, decode.py, kickoff.py
                       the steps, in the order the data flows through them
io_metrica.py, simulate.py, dataset.py
                       training data: reading the open matches, noise simulation, training table
metrics.py, check.py   scoring and submission checks
get_sample_data.sh     downloads the open training matches
requirements.txt       pinned library versions
possession_lgbm.txt    the trained classifier
possession.csv         the submission
TECHNICAL_DOCUMENT     method (.md and .pdf)
USER_GUIDE             how to run everything (.md and .pdf)
```

## Acknowledgement

Training data: Metrica Sports sample data, https://github.com/metrica-sports/sample-data.
