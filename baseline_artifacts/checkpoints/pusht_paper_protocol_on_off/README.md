# PushT paper-protocol checkpoints

This directory is the repository-backed checkpoint bundle for the standalone
PushT layerwise probe notebook. The notebook compares the frozen final
straightening-OFF and straightening-ON models on the same 18,500 public PushT
trajectories.

GitHub rejects ordinary Git objects larger than 100 MB, so each checkpoint is
stored as three numbered parts. Run `restore_checkpoints.sh` from the repository
root to concatenate the parts and verify both reconstructed files.

Expected SHA-256 checksums:

- OFF `model_latest.pth`: `a03cd7e514223db0f3543ce00748036f80df9397dde560684555801e41c936a5`
- ON `model_latest.pth`: `de31f8345d5274cb0dbd68bdaa38e8bab601eb52c4c0f7f83ea0d85a8c20af4c`

The original notebook recorded the following run provenance: two epochs,
seed 0, frame skip 5, three history frames, and one predicted frame. OFF used
encoder learning rate `1e-6` with straightening disabled; ON used encoder
learning rate `1e-5` with `aggcos1e-1` straightening. The YAML sidecars in this
directory reproduce the fields consumed by the notebook.

The binary part files still need to be copied from the original owner's
`pusht_layerwise_probe_assets` Drive folder before this bundle is complete.
