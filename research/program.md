# nanochat autoresearch

An experiment to have an LLM autonomously improve nanochat's own training.
The agent edits the model/optimizer/training code, trains for a fixed time
budget, checks whether the validation metric improved, keeps or discards, and
repeats — indefinitely.

This mirrors the standalone `autoresearch` repo, but runs against the **real
nanochat source tree** so improvements you discover can be ported directly.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `jun24`).
   The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files** (the repo is large; only these matter here):
   - `research/harness.py` — fixed data, tokenizer, eval (`val_bpb`), sequence
     length, and the 5-minute time budget. **Do not modify.**
   - `research/train.py` — the training driver you edit (hyperparameters,
     LR schedule, training loop).
   - `nanochat/gpt.py` — the GPT model. Fair game to edit.
   - `nanochat/optim.py` — the Muon/AdamW optimizer. Fair game to edit.
4. **Verify data exists**: check that `~/.cache/nanochat/base_data/` has train
   and val shards and that a tokenizer is present. If not, tell the human to run
   `uv run python -m nanochat.dataset -n 10` and `uv run python -m scripts.tok_train`
   (or reuse the already-downloaded shards).
5. **Confirm and go**: run the baseline first (see below), record it, then loop.

## Experimentation

Each experiment runs on the single AMD R9700 GPU. Training runs for a **fixed
5-minute budget** (`harness.TIME_BUDGET = 300` s of training wall-clock,
excluding startup/compile — step 0 absorbs torch.compile and is untimed). You
launch it as:

```bash
uv run python -m research.train > run.log 2>&1
```

**What you CAN do:**
- Edit `research/train.py` — architecture knobs, optimizer LRs, batch size, LR
  schedule, the training loop. Everything here is fair game.
- Edit `nanochat/gpt.py` and `nanochat/optim.py` — the actual model and
  optimizer. Architecture changes, new layers, optimizer tweaks, etc.

**What you CANNOT do:**
- Modify `research/harness.py`. It is read-only: the fixed eval (`evaluate`),
  data loaders, tokenizer, sequence length, time budget, and the `report()`
  summary format. The val_bpb it computes is the ground-truth metric.
- Install new packages or add dependencies — only what's already installed.
- Change the metric, the eval token count, or the time budget.

**The goal is simple: get the lowest `val_bpb`.** Since the time budget is
fixed, you don't optimize for speed directly — but a faster step means more
tokens trained in 5 minutes, which usually lowers val_bpb. Everything is fair
game: architecture, optimizer, hyperparameters, batch size, model size. The
only constraints are that the code runs without crashing and finishes within
the budget.

**VRAM** is a soft constraint (32 GB on the R9700). Some increase is fine for
meaningful val_bpb gains, but don't blow it up dramatically.

**ROCm note**: this box runs ROCm 7.2 on gfx1201 (RDNA4). `torch.compile` of
the model is correct and fast here, but compiling the *fused optimizer steps*
miscompiles (NaN) — `nanochat/optim.py` already gates that off on HIP. Keep
`--window-pattern`/`WINDOW_PATTERN = "L"` to hit the fast AOTriton flash path;
sliding-window masks fall back to a slow math path.

**Simplicity criterion**: all else equal, simpler is better. A tiny val_bpb
gain that adds ugly complexity isn't worth it. Removing code for equal-or-better
results is a great outcome. Weigh complexity cost against improvement magnitude.

**The first run**: your very first run establishes the baseline — run
`research/train.py` unmodified.

## Output format

When the script finishes it prints a fixed summary block:

```
---
val_bpb:          1.234567
training_seconds: 300.1
total_seconds:    332.4
peak_vram_mb:     16320.5
mfu_percent:      50.40
total_tokens_M:   98.6
num_steps:        1504
num_params_M:     74.2
depth:            8
```

Extract the key numbers with:

```bash
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

If that grep is empty, the run crashed — `tail -n 50 run.log` for the traceback.

## Logging results

When an experiment is done, append a row to `research/results.tsv`
(tab-separated, NOT comma-separated). **Do not commit `results.tsv`** — leave it
untracked. The 5 columns are:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, `.1f` (peak_vram_mb / 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	1.234567	15.9	keep	baseline (depth 8, L windows, batch 2**16)
b2c3d4e	1.221000	16.1	keep	raise matrix LR to 0.03
c3d4e5f	1.240000	15.9	discard	switch MLP to GeLU (worse)
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/jun24`).

LOOP FOREVER:

1. Look at the git state: the branch/commit we're on.
2. Tune `research/train.py` (and/or `nanochat/gpt.py`, `nanochat/optim.py`) with
   one experimental idea by directly editing the code.
3. `git commit` the change.
4. Run: `uv run python -m research.train > run.log 2>&1` (redirect everything —
   do NOT use tee or let output flood your context).
5. Read results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`.
6. If grep is empty the run crashed. `tail -n 50 run.log`, attempt a fix. If you
   can't get it working after a few attempts, give up on that idea.
7. Record the result in `research/results.tsv` (keep it untracked).
8. If val_bpb improved (lower), "advance" — keep the commit.
9. If val_bpb is equal or worse, `git reset --hard` back to where you started.

You are an autonomous researcher: if an idea works, keep it; if not, discard.
Advance the branch so you can iterate.

**Timeout**: each experiment should take ~5 min training + a minute or two of
startup/compile/eval. If a run exceeds ~10 minutes, kill it, treat it as a
failure (discard and revert).

**Crashes**: if a run crashes (OOM, bug, typo), use judgment: trivial fix →
fix and re-run; fundamentally broken idea → log `crash` and move on.

**NEVER STOP**: once the loop has begun (after setup), do NOT pause to ask the
human whether to continue. The human may be asleep and expects you to keep
working until manually stopped. If you run out of ideas, think harder — re-read
the in-scope files, read papers referenced in the code, combine previous
near-misses, try more radical architectural changes. The loop runs until
interrupted.
