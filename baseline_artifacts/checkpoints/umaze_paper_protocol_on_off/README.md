# UMaze paper-protocol checkpoints

This directory contains the frozen straightening-OFF and straightening-ON
checkpoints used by the standalone UMaze layerwise probe notebook.  The binary
files are split into sub-100 MB parts so they can be stored in ordinary Git.

Run `restore_checkpoints.sh` from the repository root to concatenate and verify
both checkpoints. The notebook performs the same restoration automatically.

## Provenance

- `off/model_20.pth`: paper-matched OSF seed-0 checkpoint supplied as
  `umaze-paper-off-osf-seed0-lr1e6-model_20.zip`; SHA-256
  `6f864080cd6bef3b818fdeec52459208780ba8f217b6153d359eafb93000ba98`.
- `on/model_20.pth`: paper straightening-ON checkpoint formerly distributed as
  Google Drive file `1KpxRh3g7wBgwwZ5lsI7Tr2LhA-UUaefw`; SHA-256
  `b41f3eabffe6e24585deaec2d2b6afb2f09cf3b840d905958493187e64ad7156`.

The original checkpoint archives did not contain Hydra output. The YAML files
here are transparent provenance sidecars reconstructed from Appendix A.2/B.2
of the Temporal Straightening paper and the released `conf/train.yaml`
defaults. They are not represented as byte-for-byte original run configs.

