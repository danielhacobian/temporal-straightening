# Wall Baseline Reproduction

**Sources:** `baseline_artifacts/results/wall_baseline.json`, per-seed/per-chunk logs under `baseline_artifacts/plans/wall_reproduction/seed_{100,200,300}/chunk_{00..40}/logs.json`. Added to the repo via PRs #7/#8 (commits `df7ef9e`, `05a6674`, `908631c`). This is the **proper** Wall reproduction; it supersedes the broken Wall numbers in [The Reproduction Study](The%20Reproduction%20Study.md) and [Results Analysis](Results%20Analysis.md).

## The headline

A clean 3-seed Wall reproduction of the **DINO-patch baseline** (no straightening):

| | Config | Success (Wall, open-loop GD) |
|---|---|---|
| **Ours** (3 seeds, 150 evals) | DINOv2 patch, frozen, no projector, straightening **OFF** | **61.3 ± 8.1%** |
| Paper (Table 5) | same baseline | 73.33 ± 3.4% |

We **under-reproduce by ~12 points**, and the error bars barely fail to overlap (ours ≈53–69%, paper ≈70–77%). So this is a *partial* reproduction — the right ballpark and clearly a working pipeline, but not a bullseye.

## The single most important takeaway

**This supersedes the old Wall evidence.** The earlier `modal_evidence` Wall runs in [The Reproduction Study](The%20Reproduction%20Study.md) reported **0.02 success** for both variants. This proper run gets **61%**. The 0.02 result was a broken/underpowered run (that report even noted Wall required import shims and image-shape fixes and rendered no media). The correct Wall baseline is ~61%; the 0.02 figure should be disregarded.

## Per-seed results

| Seed | Success | State dist | Visual dist | Proprio dist | Visual emb div |
|---|---|---|---|---|---|
| 100 | 0.68 | 11.32 | 0.640 | 6.96 | 509.1 |
| 200 | 0.50 | 13.70 | 0.802 | 8.06 | 531.1 |
| 300 | 0.66 | 12.88 | 0.647 | 7.60 | 489.0 |
| **Mean** | **0.613 ± 0.081** | 12.63 | 0.696 | 7.54 | 509.7 |

Note the **wide spread** — seed 200 (0.50) sits far from seeds 100/300 (0.68/0.66). And within a single seed the 10-eval chunks swing from 0.4 to 0.8. Wall is a high-variance task; even 50 evals/seed leaves a lot of noise, which is exactly why the 3-seed protocol matters here.

## How it was run: chunked planning

Unlike the UMaze run ([UMaze Baseline Reproduction](UMaze%20Baseline%20Reproduction.md)), Wall planning was split into **five 10-eval chunks per seed** (`chunk_00`…`chunk_40`, offsets 0/10/20/30/40) and aggregated by equal-weighted mean (divergence norms combined by root-sum-square). This is the deterministic chunked-evaluation path added in commit `466382d` ("Preserve dataset samples in chunked planning") — a memory-management device so 50 goal-reaching rollouts don't have to fit at once. The 15 distinct `plan_target` hashes confirm each seed×chunk got its own goal set.

Planner: open-loop GD, horizon 25, Adam lr 0.1, 100 opt steps, zero action init, `decode_for_viz` on (150 decoded final videos were produced). See [The Planners - GD, CEM, MPC](The%20Planners%20-%20GD,%20CEM,%20MPC.md).

## Config detail worth noting

The checkpoint is `wall_False_agg32_projnone_dim384_hw14_sgTrue_lr1e-05` — **frozen DINO patch, no projector, straightening off, full 384-dim features**. This differs from the UMaze run, which used a trainable **channel projector to dim 8 with straightening ON**. So the two new baselines are not directly comparable: UMaze tests the *method* (straightening + adapter), Wall tests the *plain DINO-patch control*. Consistent with that frozen-patch setup, `mean_div_visual_emb` here is high (~510) — the same ill-conditioned-planning-space signature seen in the frozen Medium runs ([Results Analysis](Results%20Analysis.md)), and the opposite of UMaze's well-behaved ~17.

## Caveats

1. **Straightening-OFF only.** There is still **no proper straightening-ON Wall run** — the only ON Wall result is the broken 0.02 from `modal_evidence`. So this folder tells us the *baseline* reproduces (partially), not whether straightening helps on Wall.
2. **Under-reproduces the paper** (61 vs 73%). Could be dataset scale, training length (20 epochs), or Wall's inherent variance; not diagnosed here.
3. **High variance** across seeds and chunks — treat the 61% as a wide estimate, not a precise number.
4. **Checkpoint not committed** (size); identified by `checkpoint_sha256 = fba13171…`.

## Bottom line

The `baseline_artifacts/` folder is the **real Wall baseline**, and it does two things: (a) it retires the bogus 0.02 Wall result, replacing it with a credible ~61% DINO-patch baseline, and (b) it adds proper 3-seed, chunked, deterministic evaluation. It partially reproduces the paper's Table 5 baseline (61 vs 73%). The natural next step — mirroring the open question from [UMaze Baseline Reproduction](UMaze%20Baseline%20Reproduction.md) — is a **matching straightening-ON Wall run on the same trainable-adapter config**, so Wall finally gets a real ON-vs-OFF comparison instead of a broken one. Artifact index in [Evidence Packs](Evidence%20Packs.md).
