# ALOHA mobile cabinet

LeRobot-format hardware demonstrations used by the ALOHA mobile-dynamics and
control experiments. The snapshot contains 85 episodes, 127,500 rows at 50 Hz,
and the columns `observation.state`, `observation.effort`, `action`, episode and
frame indices, timestamps, terminal flags, and task indices.

The training code identifies the source as `lerobot/aloha_mobile_cabinet`, but
the exact Hub revision and original download command were not retained. See
`provenance.yaml`; fill in the revision before treating this snapshot as fully
reproducible.

The Parquet payload is stored outside Git. After restoring it, verify it with:

```bash
sha256sum -c <(printf '%s  %s\n' \
  a3409d3282b2cf9a94d32b301756fbd15dca5862cc1c32973c7316dd8853c57a \
  data/aloha_mobile_cabinet/file-000.parquet)
```
