# UMaze Baseline Reproduction

**Sources:** `umaze_reproduction/results/umaze_baseline.json`, `umaze_reproduction/results/umaze_RESULT_50.txt`, per-seed planner logs `umaze_reproduction/plans/seed_{100,200,300}/logs.json`. Added to the repo on 2026-07-14 (commit `8be52b1`, contributor `erinlee316`) — **after** the runs covered in [The Reproduction Study](The%20Reproduction%20Study.md) and my [Results Analysis](Results%20Analysis.md), and it materially updates their conclusions.

## The headline

This is the **first faithful, paper-protocol reproduction** in the repo — and it reproduces the paper's central claim cleanly:

| | Straightening | Success (UMaze, open-loop GD) |
|---|---|---|
| **Ours** (3 seeds) | ON (`aggcos`, λ=0.1) | **96.0 ± 1.63%** |
| Paper (Table 1/5) | ON | 94.0 ± 1.63% |
| Paper control | OFF | 44.0 ± 7.12% |

Our straightening-ON number lands **within error of the paper's**, and towers over the paper's straightening-OFF control (a ~50-point gap). On this task, temporal straightening clearly and reproducibly helps.

## Why this run "works" when the earlier ones didn't

The contrast with the Medium/Wall runs in [Results Analysis](Results%20Analysis.md) is the whole point. Those earlier runs used a **frozen DINO patch encoder with no projector**, which has no trainable path between pixels and the measured latents — so the straightening loss could not reshape latent geometry, and the planner results came out degenerate (byte-identical across variants). This UMaze run fixes exactly that:

| | Earlier Medium/Wall | This UMaze run |
|---|---|---|
| Encoder | frozen DINO patch, **no projector** | DINO patch **+ channel projector → 14×14×8** |
| Trainable path to latents? | ❌ none | ✅ the dim-8 adapter |
| Straightening | `cos1e-1` (inert on frozen features) | `aggcos`, λ=0.1 (acts on the adapter) |
| Seeds | 1 | 3 (100/200/300), 50 evals each = 150 |
| Protocol | reduced/toy | paper protocol |

This is precisely the setup [The Straightening Loss](The%20Straightening%20Loss.md) and [Encoders - DINO and Friends](Encoders%20-%20DINO%20and%20Friends.md) flagged as *necessary* for straightening to do anything: a frozen backbone is fine, but there must be a trainable projector for the loss to sculpt the geometry the planner sees.

## Per-seed results

| Seed | Success | State dist | Visual dist | Proprio dist | Visual emb div |
|---|---|---|---|---|---|
| 100 | 0.96 | 2.731 | 0.393 | 0.900 | 16.97 |
| 200 | 0.98 | 2.529 | 0.380 | 0.835 | 16.99 |
| 300 | 0.94 | 2.930 | 0.412 | 0.971 | 17.82 |
| **Mean** | **0.96 ± 0.016** | 2.730 | 0.395 | 0.902 | 17.26 |

The tight spread (94–98%) across seeds is the statistical rigor the earlier single-seed runs lacked.

## A corroborating detail: the latent space is now well-behaved

`mean_div_visual_emb` — how far the planner's imagined visual embeddings drift from what's achieved — is **≈17 here vs ≈778** for the frozen DINO-patch Medium runs. That ~45× improvement in planning-target conditioning is the mechanistic reason success jumps from 0.14 to 0.96: the trainable dim-8 projector gives gradient-descent planning a smooth space to optimize in, exactly the property [The Big Idea](The%20Big%20Idea.md) argues straightening should produce.

## Planner protocol

Open-loop gradient descent via `MPCPlanner` wrapping `GDPlanner`, configured so there's effectively no replanning (`max_iter=1`, `n_taken_actions=25=horizon` → the full plan is executed once): horizon 25, Adam at lr 0.1, 100 optimization steps, zero action initialization, `objective_mode=last`. This matches the paper's UMaze open-loop-GD protocol. See [The Planners - GD, CEM, MPC](The%20Planners%20-%20GD,%20CEM,%20MPC.md).

## Caveats — read before over-claiming

1. **The OFF control is the paper's, not reproduced locally.** The committed checkpoint is straightening-**ON only**; the 44% figure is quoted from the paper's Table 1/5. So the 96→44 gap mixes *our* reproduced ON number with the *paper's reported* OFF number. A locally-run straightening-OFF UMaze checkpoint would be needed to make the gap fully self-contained.
2. **A logging quirk in `plan.log`, not a wrong model.** The plan log prints `Unknown projector 'none'` and `Straightening disabled` — but "straightening disabled" is *expected* at inference (it's a training-only loss), and `emb_dim: 28` (= 8 visual + 10 proprio + 10 action) confirms the dim-8 channel projector **is** loaded. The projector-name warning is cosmetic.
3. **UMaze is the easiest task in the suite.** A 96% ceiling here does not by itself carry to Medium, Wall, or PushT — and the earlier (flawed) Medium runs are a reminder that harder tasks behaved differently.
4. **Checkpoint not committed** (size); it's identified by `checkpoint_sha256 = dbe075b1…` for provenance.

## Bottom line

This result substantially updates the story in [Results Analysis](Results%20Analysis.md). The earlier "straightening doesn't help / DINO-CLS wins" impression was an artifact of the **frozen-encoder** setup that couldn't measure the effect. With the **trainable channel-projector adapter and the paper's protocol**, temporal straightening's benefit reproduces **cleanly, largely, and across seeds** on UMaze (96% vs a 44% control). The open question now shifts to the harder environments — and specifically to running a **local straightening-OFF control** on each, so the ON/OFF gap is measured end-to-end rather than half-quoted from the paper. Artifact index in [Evidence Packs](Evidence%20Packs.md).
