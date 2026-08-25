# Writing the two papers — start here

Everything needed to draft either paper. Two submissions, **same checkpoint, same
numbers**, differing in framing and in one figure.

| | **Sim2Science** | **Representations** |
|---|---|---|
| length | 5 pp | 4 pp |
| deadline | Aug 29 2026 | Aug 29 2026 |
| theme | leveraging *imperfect* simulators against real data | symmetry / geometry in learned representations |

## The three documents

| file | what it is |
|---|---|
| **`RESULTS_SUMMARY.md`** | **Read this first.** Every number, with the caveat attached to it. Setup facts, metric definitions, results, both ablations, per-venue framing, limitations. |
| `plan_sim2science.md` | Per-figure spec: what each figure claims, its panels, and the slug ↔ figure-number mapping for LaTeX. |
| `CLAUDE.md` | Project conventions, known bugs, hardware. Background, not writing material. |

## The figures

All in `figures/run3/paper/`. **Three per paper, no appendix.**

| | Sim2Science | Representations |
|---|---|---|
| Fig 1 | `fig1_hero_grid` | `fig1_hero_grid` |
| Fig 2 | `fig3_survey_invariance_noise` | `fig3_survey_invariance` |
| Fig 3 | `fig7_cross_attention` | `fig7_cross_attention` |

Each ships `.pdf` (use this in LaTeX), `.png` (quick viewing) and `<slug>_data.csv`
holding **exactly** the plotted numbers.

Generated but not allocated to either paper: `fig2_architecture`,
`fig6_loss_ablation`, `figA1_hero_grid_appendix`.

### Two rules that matter when writing

**Quote the CSV, never the plot.** Every figure has one, and the papers are expected to
cite values that match it exactly.

**Slug digits are historical and match neither venue's numbering.** `fig3_*` is Fig 2;
`fig7_*` is Fig 3. Filenames are frozen provenance — resolve numbering with
`\label`/`\ref`, never from a filename.

## Claims that appear in NO figure

These were cut for length. Each is measured and quotable, and at least two are things a
referee will ask about — see `RESULTS_SUMMARY.md` §6.3.

* **Reblur consistency** (the Sim2Science mechanism) — flags HST at 0.92× its own noise
  floor on noisy input, **without using ground truth**.
* **Top-octave coherence** — the answer to "is this just sharpening artifacts?"
  (0.993 vs 0.900 at HST; 0.975 vs 0.615 at Rubin).
* **θ_E is unaffected by removing cross-attention** — the control for the capacity
  confound. Fig 3 alone does not answer that objection.
* **The architecture** — no diagram is allocated, so the method must be fully described
  in prose.

## Regenerating

See `RESULTS_SUMMARY.md` §8. `--fov 12.0` is mandatory — the checkpoint does not record
its field of view.
