# Restore state after the 2026-08-19 maintenance shutdown

Server goes down **09:00 Central = 14:00 UTC, Wed 2026-08-19**.
Written 05:10 UTC while Run 2 was training.

---

## TL;DR — the one command you probably need

```bash
cd /home/jarugula/survey-invariant-lensnn
ROOT=/deepskieslab/stronglensing/survey-invariant-lensnn

# 1. Did Run 2 finish before the shutdown?
grep -c "" $ROOT/runs/run2_full/version_0/metrics.csv     # 161 lines == 80 epochs done
grep "best checkpoint" $ROOT/runs/run2_full_stdout.log    # present == it completed

# 2a. If it FINISHED -> go to "Evaluate Run 2" below.
# 2b. If it did NOT finish, resume from where it stopped:
nohup env PYTHONPATH=src python -m survey_lens.train \
  --name run2_full --epochs 80 --batch-size 8 \
  --n-train 10000 --n-val 2000 --survey-val \
  --loss-mode joint --seed 0 \
  --cache-dir $ROOT/data --log-root $ROOT/runs --accelerator cuda \
  --resume auto \
  >> $ROOT/runs/run2_full_stdout.log 2>&1 &
```

**`--epochs 80` must stay 80 on resume.** The LR schedule is
`CosineAnnealingLR(T_max=max_epochs)`; passing a different number silently
changes the schedule and invalidates the comparison against Run 1.

⚠ **Lightning's CSVLogger DELETES an existing `metrics.csv`** when it reuses a
version directory, then writes only the epochs from the new process — so a
naive resume destroys the pre-interruption training curve. (Verified by
actually resuming a run and watching the file vanish.) `train.py` now snapshots
it to `metrics.pre_resume_<timestamp>.csv` automatically before the logger
starts; concatenate the pieces when plotting the curve. Belt and braces:

```bash
cp $ROOT/runs/run2_full/version_0/metrics.csv \
   $ROOT/runs/run2_full/version_0/metrics.manual_backup.csv
```

`--resume auto` picks up `.../version_0/checkpoints/last.ckpt`, which is
rewritten every epoch, so at most one epoch is lost. It restores model +
optimizer + LR scheduler + epoch counter. `--version 0` is pinned so the
resumed run appends to the *same* `metrics.csv` rather than starting a
`version_1` and splitting the training curve across two files.

---

## Everything durable is on CephFS (survives the shutdown)

`$ROOT = /deepskieslab/stronglensing/survey-invariant-lensnn` — network mount, persists.
**`/tmp/claude-*/**` is scratch and will be LOST.** Nothing needed lives there.

| path | what | size |
|---|---|---|
| `$ROOT/data/` | frozen dataset: train 10k, val 2k, test_set_1 200x5 | 3.0 GB |
| `$ROOT/runs/run2_full/version_0/checkpoints/` | Run 2: `last.ckpt` + top-3 by `val/l1` | ~88 MB |
| `$ROOT/runs/run2_full/version_0/metrics.csv` | Run 2 training curve | |
| `$ROOT/runs/run1_full/version_0/` | Run 1 (superseded, keep for provenance) | |
| `$ROOT/figures/run1_ts1_hf_metrics.csv` | **Run 1's top-octave numbers — the baseline Run 2 is judged against** | |
| `$ROOT/figures/run1_summary.md` | Run 1 write-up + retraction banner | |
| `$ROOT/figures/gate_ts1_metrics.csv` | 3-epoch gate evaluation | |

⚠ **`/deepskieslab` is 99% full (64 GB free).** Checkpoints are 22 MB each and
`save_top_k=3` caps it, but do not start anything that writes tens of GB
without checking `df -h /deepskieslab` first.

**Do NOT rebuild the dataset.** It is frozen and Run 1 vs Run 2 is a controlled
comparison only because both train on byte-identical bytes. Rebuilding costs
~7 min but breaks that. `cache_datasets.py` exists if you ever genuinely need to.

---

## Where the work stands

All code is committed to `main` in `/home/jarugula/survey-invariant-lensnn`
(**not pushed** — the branch is ahead of `origin/main`; push when convenient).

| commit | what |
|---|---|
| `88cb13d` | Run-1 summary committed verbatim, for provenance |
| `facc7e9` | `metrics.py` (top-octave metric + Wiener baseline) + wiring |
| `f928456` | Retraction banner on the Run-1 conclusions |
| `92192c0` | The Run-2 model / loss / training fixes |

**Run 1's checkpoint no longer loads against current `model.py`** (the decoder
changed shape). Its top-octave numbers are preserved in
`$ROOT/figures/run1_ts1_hf_metrics.csv`; to re-derive them, `git checkout facc7e9`.

### Why Run 2 exists

Run 1's reconstruction was adjudicated with masked L1/SSIM — the metrics
`CLAUDE.md:187-189` says cannot decide deconvolution. With the correct
top-octave metric, `identity` (returning the blurred input unchanged) beat Run 1
in **all five** surveys and HST/Euclid scored *worse than emitting nothing*.
Root causes and fixes are in the commit message of `92192c0` and in
`figures/run1_summary.md`'s correction banner.

### Gate evidence (already validated, do not redo)

3 epochs of Run 2 vs 60 epochs of Run 1, top-octave complex error
(`f_hat=0` scores exactly 1.0, so >1.0 is worse than emitting nothing):

| survey | Run 1 (60 ep) | gate (3 ep) | identity | Wiener |
|---|---|---|---|---|
| HST | 2.705 | 0.653 | 0.593 | 0.025 |
| Euclid | 1.354 | 0.341 | 0.412 | 0.038 |
| Roman | 0.949 | 0.132 | 0.168 | 0.093 |
| Rubin | 0.940 | 0.869 | 0.962 | 0.321 |
| DES | 0.937 | 0.880 | 0.960 | 0.339 |

Verdicts are unchanged when patch-grid harmonics are excluded, so the gain is
signal and not decoder artifact.

---

## Evaluate Run 2 (once training completes)

```bash
CK=$(ls -t $ROOT/runs/run2_full/version_0/checkpoints/epoch=*.ckpt | head -1)  # or pick best val/l1
PYTHONPATH=src python -m survey_lens.test --ckpt "$CK" \
  --test-set 1 --n-lenses 200 --device cuda --workers 16 \
  --out-prefix $ROOT/figures/run2_ts1
PYTHONPATH=src python -m survey_lens.test --ckpt "$CK" \
  --test-set 2 --n-lenses 200 --device cuda --workers 16 \
  --out-prefix $ROOT/figures/run2_ts2
```

`test.py` **silently falls back to random weights if `--ckpt` is missing** —
confirm `loaded checkpoint:` appears in the output before trusting any number.
`--device` defaults to `cpu`; always pass `--device cuda`.

The headline column is `hf_complex_err` (plus `hf_coherence`). **Ignore
`hf_power_ratio` on its own** — it is phase-blind and Run 1 scored a
healthy-looking 0.86 at Euclid while being worse than emitting nothing.

Test Set 2 is **evaluation-only**: same checkpoint, no retraining, no noise
augmentation, and nothing may be tuned on it.

---

## Environment gotchas (re-verify after maintenance)

These bit us before and are patched in `src/survey_lens/env_setup.py`, which
every entry point imports **first**:

- `os.cpu_count()` reports the host's 127 cores but the cgroup quota is 32
  (`/sys/fs/cgroup/cpu.max`). Unpinned, PyTorch oversubscribes ~7x.
- Memory cgroup limit ~45 GB (`/sys/fs/cgroup/memory.max`), well under host RAM.
- `torchvision` is ABI-broken against the installed torch, which breaks
  `import lightning`. `env_setup` stubs it **only if the real import fails**.

Sanity check after restart:
```bash
PYTHONPATH=src python -c "
import survey_lens.env_setup as e, torch
print('cgroup cores:', e._n_cpu, '| torch threads:', torch.get_num_threads())
print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
PYTHONPATH=src python -m pytest tests/ -q | tail -3    # expect 78 passing
```
GPU is an A100 80GB MIG 2g.20gb slice (20.9 GB). Measured peak 10.7 GB at the
training prior edge, so batch 8 fits with headroom.

---

## Next steps after Run 2 evaluates

1. Compare Run 2 vs Run 1 vs `identity` vs Wiener on `hf_complex_err`, TS1 **and** TS2.
2. **The open question:** Wiener beats the network 3-10x on *noise-free* data but
   blows up under noise (top-octave ratio 1977x on noisy HST). Whether Run 2
   closes that gap decides the paper's deconvolution framing — deliberately
   deferred until these numbers exist.
3. Still outstanding: cache Test Set 2 + add `--cache-dir` to `test.py`/`diagnose.py`
   (every eval currently re-simulates); write `make_paper_figures.py` (Step 10,
   does not exist yet).
4. Known-open and untouched: CLAUDE.md bug 3 (Euclid/Roman PSF stamps
   under-sampled, which is why Wiener over-shoots at Roman).
