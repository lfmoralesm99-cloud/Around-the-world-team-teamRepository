# Ball possession from automatic tracking — technical document

Challenge match: BAR-ATH · Team: **Around the world**

Members: Maria Cachim, Ana Barroso, Stephane Cruz, Dan Arscott, Luis Felipe Morales

Repository: <https://github.com/lfmoralesm99-cloud/Around-the-world-team-teamRepository>

## Submission file fingerprint

File: `possession.csv`

sha256: `97cb9940a32764f55465d97ad92d0007b21601a61eac8ba0283067d134d21acc`

The rules require that the repository regenerates the submitted file exactly, not approximately.
This code identifies that exact file: changing a single label would change it. `python run.py`
prints the sha256 of the file it writes, so a matching code confirms the reproduction without
comparing 199,319 rows.

## 1. Approach

**What we built.** A supervised frame classifier trained only on other matches, followed by a
sequence decoder that outputs whole possessions.

1. Read the tactical and broadcast tracking. Row *i* of the tactical file is frame *i* of the video.
2. Put the broadcast tracking on the tactical timeline and use its ball where the tactical ball is missing.
3. Describe every graded frame with 342 identity-free features: ball position, speed and margin to
   the lines; holes in the ball track; evidence that play stopped (ball left the pitch, ball came to
   rest); stretches where the ball moves without any change of velocity; distance from the ball to
   each team; kicks (sharp changes of ball velocity) and which team made the last and next ones;
   team shape and player speed; each with context windows of 0.5 to 10 s before and after the frame.
4. Classify each frame as us / them / dead with gradient-boosted trees (LightGBM, 500 trees). Each
   frame is seen once from each team's side and the two views are averaged, so the model cannot
   favour home or away.
5. Decode with Viterbi, charging a price for every change of state, then merge spells and stoppages
   too short to be real.
6. Apply two kick-off rules (section 4) and write H, A or D for graded frames, X elsewhere.

**Why this rather than something else.**

- *Why not a nearest-player rule.* In this tracking the ball sits a median 1.8 m from the nearest
  player and its track is low-pass filtered. Proximity flickers between teams, which is why the
  published baselines reach macroF1 0.65 but spellF1 0.05 to 0.24.
- *Why a decoder.* The ranking metric scores possessions as objects. One wrong flip inside a long
  possession costs two possessions. In validation the decoder raised spellF1 from 0.560 to 0.756
  with no change in macroF1.
- *Why train on other games.* The rules forbid any hand label on the provided game, and the public
  window is not training data. Other matches with event data are the only legitimate supervision.
- *Why simulate noise.* The other matches are corrected tracking; the challenge data is automatic.
  A model trained on clean data learns "a ball position exists, so the ball is in play" and "the
  ball is next to a player, so his team has it". Both are false here. We degrade the training games
  until label-free statistics of their ball track match the challenge tracking.

| Label-free statistic of the ball track | Challenge match | Training game, as published | Training game, degraded |
| --- | --- | --- | --- |
| Kicks detected per minute of track | 27.6 | 31 to 36 | 28 to 32 |
| Velocity change over 0.4 s, 90th / 99th percentile (m/s) | 6.9 / 13.2 | 11.3 to 11.9 / 23 to 26 | 7.4 to 7.5 / 13.8 to 14.7 |
| Distance to nearest player, 5th / 25th / 50th percentile (m) | 0.26 / 0.84 / 1.83 | 0.03 / 0.10 / 0.28 | 0.26 / 0.67 / 1.3 |
| Stretches of 1 s or more with no change of velocity, per 10 min | 14.7 | about 90 (straight lines between touches) | 8 to 11 |
| Tracks that leave the pitch over a touchline: come back unbroken / end outside | 25 / 13 | 1 / 15 to 20 | 23 / 12 to 14 |
| Frames with a ball position | 63.0% | 59 to 68% (in play only) | 64 to 72% |

**Validation.** Leave-one-game-out on the three training matches, held-out game degraded with an
unseen seed, scored with our re-implementation of the metrics from EVALUATION.md.

| Held-out game | Decoded spellF1 | Decoded macroF1 | Possessions predicted / true |
| --- | --- | --- | --- |
| Sample Game 1 | 0.771 | 0.898 | 314 / 350 |
| Sample Game 2 | 0.752 | 0.940 | 299 / 324 |
| Sample Game 3 | 0.746 | 0.943 | 347 / 381 |
| Mean | 0.756 | 0.927 | |

These are scores on other matches with simulated noise. They are not an estimate of the score on
the challenge match.

On the challenge match the file has 326 possessions (147 + 179); H 32.9%, A 20.5%, D 46.7% of
graded frames. Two label-free checks: players move at 2.27 m/s when the ball is labelled in play
and 1.05 m/s when dead, and the team in possession agrees with the automatic Smart Tagging
"BUILD UP" tags 95% and "PROGRESSION" tags 89% of the time.

## 2. Assets used and dropped

| Asset | Used | How |
| --- | --- | --- |
| Tactical tracking (EPTS txt + xml) | Yes | Player positions by team and ball; the basis of every feature |
| Broadcast tracking (EPTS txt + xml) | Yes | Aligned to the tactical timeline from the data; its ball fills 5,928 graded frames (3.7%) where the tactical ball is missing. The disagreement between the two cameras' ball positions (median 0.7 m on a slow ball, 2.7 m on a fast one) also calibrates how far a ball in the air is displaced |
| Smart Tagging | Partly | Only the "Kick Off (Start)" tag of each half, for the kick-off rule. The team tags (build-up, progression) were used as a label-free check. The set-piece tags were tested as extra evidence of stoppages and dropped: they mark stoppages the model already finds |
| Homography (both cameras) | No | The tracking is already in pitch coordinates |
| Videos | No | Not used; no detection or tracking of our own |
| Public 5-minute ground truth | Checking only | Scored with the organisers' `run_evaluation.py` to confirm the file is valid and aligned, and looked at to understand how the reference labels stoppages and restarts, as the challenge text intends. Never used to train, to set a constant, or to choose between versions (section 6) |

**Broadcast alignment.** The broadcast file is delivered "not aligned". For the right lag, players
seen by both cameras coincide. The lag is constant per half: tactical frame = broadcast frame + 4014
(first half) and + 26920 (second half), with a median player disagreement of 0.3 m. A broadcast
frame is used only if its players agree with the tactical players, which also rejects replays.

**Tried and dropped.**

- Training on clean data with light noise: scored 0.81 in simulation, but the simulation did not
  resemble the challenge tracking. Replaced by the calibrated simulator.
- A second model over the first model's probabilities (stacking): 0.764 against 0.769. Dropped.
- Finding the kick-off from the player line-up alone: false detections during the warm-up. Replaced
  by the tag window plus a line-up rule.
- Smart Tagging team labels as a prior in the decoder: could not be validated, because the training
  games have no such tags. Not used.
- A rule that a ball beyond a line is dead (Laws of the Game), forced before decoding. Tested at
  16 settings (0.3 to 3 m beyond the line, for 3 to 25 frames) on the training matches: spellF1
  0.742 to 0.755, against 0.756 without it. On the challenge match the ball is beyond a line while
  labelled in play in 14 places; about half are the ball crossing the line at the end of a play,
  where we switch to dead ball 0.3 to 1.2 s later, and most of the rest are throw-ins being taken.
  Not used.
- A rule that a stoppage predicted inside a hole of the ball track needs a reason (ball last seen
  near a line or at rest, or a long hole). Tested at 18 settings: spellF1 0.717 to 0.754. In the
  training matches most such stoppages are real fouls in midfield, and the rule deletes them. Not used.
- A fixed delay on restarts (section 5, restart timing). Not used: its size could only be estimated
  from three restarts in the public window.

## 3. Training data

One labelled dataset, none of it from the provided game.

| | |
| --- | --- |
| Dataset | Metrica Sports sample data, three anonymised matches |
| Source | https://github.com/metrica-sports/sample-data, revision `e706dd506b360d69d9d123d5b8026e7294b13996` |
| Labelled by | Metrica Sports, for their public release. We labelled nothing ourselves |
| What is labelled | Event data: passes, carries, shots, recoveries, set pieces, balls lost, challenges, with team and frame; plus corrected tracking with a ball position only while the ball is in play |
| How much | 429,923 frames at 25 fps (about 4.8 hours): 145,006 + 141,156 + 143,761. 1,055 possessions after conversion |
| How it is used | Per-frame H/A/D labels are derived from the events by code (`io_metrica.py`, rules in section 4). Three degraded copies per game, two views per frame, every fifth frame: 515,908 training rows |

No label on the challenge match exists anywhere in the pipeline: no hand labels, no pseudo-labels,
no ground truth, no event data about the game.

## 4. Possession rules implemented

The rules live in two places: the conversion of the training events into labels (what the
classifier learns) and the decoder (what the output is allowed to look like).

| Situation | Rule |
| --- | --- |
| Gaining possession | Only a pass, carry, shot, recovery or set piece by a team gives it possession (training labels) |
| Loose or contested ball | Possession stays with the last team that had it. A ball lost never changes the owner; the owner changes when the other team makes a controlled action |
| Challenges and aerial duels | A challenge on its own never grants possession. In the training data the winner of a duel is credited with a recovery on the same frame, so possession changes at the duel when the winner keeps the ball |
| Simple touches and deflections | A spell shorter than 8 frames (0.32 s) next to a spell of the other team is merged into it, in the training labels and again in the output. The reference keeps possessions as short as 0.76 s |
| Changing team | The decoder charges 4 log-probability units for H to A or back, so a change needs more than one frame of evidence |
| Dead ball | In training, D is where the corrected tracking has no ball. The classifier learns the cues that survive in automatic data: ball beyond a line, ball at rest, ball carried at walking pace to the restart, a long hole in the ball track while players walk |
| Ball tracked during a stoppage | Still D. A ball that is carried or passed between team-mates while play is stopped does not start a possession; the possession starts when the restart is taken |
| Short holes | A stoppage shorter than 12 frames (0.5 s) is absorbed by the possession around it; it is a tracking hole, not a stoppage |
| Restarts | Possession starts on the frame the ball is played. Who "owns" a dead ball is never predicted: a dead ball is D |
| Opening kick-off | Frames of a half before the automatic kick-off tag are D. Inside the tag window, a possession that ends while 90% of each team's tracked players are still in their own half is set to D |
| Halves | Each half is decoded on its own; a possession cannot span the break |
| Abstention | X only outside the two graded blocks. Inside them the system always answers, because on spellF1 an X is filled with the previous label anyway and on macroF1 it is a miss |

On the challenge match the kick-off rule changes 1,044 frames, all before the second-half kick-off,
where the graded block starts 117 s early. The first possession of each half then starts at video
frames 11185 and 113741; the kick-off anchors in the organisers' `games.py` are 11186 and 113741.

## 5. Limitations

**Where it fails.**

- *Stoppages are the main error on this match.* In the public window most errors are dead balls,
  missed or invented. In validation the picture is the reverse: 87% of stoppages of a second or more are found and 8% of predicted stoppages are
  invented, and team attribution costs more (spellF1 0.80 with perfect stoppages, 0.92 with perfect
  teams). So stoppages on the real match are harder than in our simulation.
- *Quick restarts with the ball in view.* A throw-in taken within about 5 s, with the ball tracked
  along the touchline and players still moving, looks like play down the wing. These are
  usually missed.
- *Slow play with no ball in the tracking.* When the ball is untracked for several seconds and
  players slow down, the system calls a stoppage. It cannot tell this from a foul.
- *Bridged ball tracks.* After some headers the tracker draws a straight line for several seconds
  to where it next finds the ball. The model now recognises these stretches, but it still gives
  possession to the team the line passes near, and changes late.
- *Short possessions.* About 9% fewer possessions are predicted than exist in validation. Touches
  of about a second where both teams are within 2 m of the ball are usually missed.
- *Restart timing.* At restarts our possession starts on the frame the ball first appears in the
  tracking, already moving; in the public window the reference starts it 10 to 13 frames later in
  all three such cases. On the training matches the model is on time (median error 0 frames at
  every kind of edge), so this is a difference of convention between the reference and our
  training labels, not a learned bias. We did not correct it.
- *Small training set.* Three matches, none with these teams or this camera system.
- *Not measured on the full match.* The only labels we can score against are the 17 possessions of
  the public window, too few to measure the system (section 6).

**With more time.**

1. Re-detect the ball in the video for frames where the tracking has none or has a bridged line.
2. Add labelled matches from other open datasets that carry per-frame ball status and possession.
3. Self-train on the challenge match with pseudo-labels produced by the algorithm.

## 6. Reproduction

```bash
pip install -r requirements.txt          # tested with Python 3.13 on Linux; four pinned libraries

# Unpack the gated bundle into ./data (or set MTAH_DATA_DIR). It is not in the repository.
#   data/tactical/BAR-ATH_tactical_atd.txt, _atd.xml, _smart_tagging.xml
#   data/broadcast/BAR-ATH_broadcast_atd.txt, _atd.xml

python run.py                            # about 9 min on 2 cores -> possession.csv
python check_submission.py possession.csv
```

`run.py` uses the committed model `possession_lgbm.txt` and prints the sha256 of the file it
writes. Two independent runs produced the same file, byte for byte.

To rebuild the model and the validation table without any challenge data:

```bash
bash get_sample_data.sh          # clones the training matches at the pinned revision
python train.py                          # about 10 min; two runs gave the same model file
python validate.py                       # about 20 min; the table in section 1
```

The model file is committed because LightGBM does not guarantee identical trees across platforms
or builds. The match data is excluded by `.gitignore` (`data/`, `cache/`, `external/`).

**Checking the file.** The organisers' `evaluation/run_evaluation.py` validates the file and
finds the frames aligned: shifting our labels by up to 12 frames later leaves the score unchanged,
and 6 or more earlier lowers it. There are no abstentions on graded frames. Its own bootstrap
interval for spellF1 on the 17 public possessions is 0.43 to 0.77, so the window cannot rank
alternatives, and we did not use it to.

A step-by-step guide to every command, with expected output and troubleshooting, is in
`USER_GUIDE.md`.
