# Medium Adapter Ablations - Partial Evidence

These runs were launched to address the caveat in the frozen-DINO speed ablation. The variants use `encoder=dino_channel`, so the DINO backbone stays frozen but the projected latent path includes trainable projector and aggregation parameters.

Status: incomplete. Modal rejected new app creation with:

```text
App creation failed: workspace billing cycle spend limit reached
```

The raw logs recovered from the Modal volume are in `raw_volume/`. The parsed partial summary is `partial_adapter_results.json`.

No final planner success rates or latent diagnostics should be inferred from this folder. It only records training progress and planner progress before the spend-limit stop.

