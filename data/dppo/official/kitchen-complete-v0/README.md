# Official DPPO Kitchen Complete snapshot

Unmodified preprocessed Kitchen dataset downloaded from the official DPPO
Google Drive folder. It is the source for the episode-disjoint split in
`data/dppo/kitchen-complete-v0`.

Acquire it with:

```bash
python -m experiments.cond_stoch_interpolant_dppo.download_kitchen \
  --environment kitchen-complete-v0
```

The downloader opens/downloads the official folder; its precise file placement
may require manual handling. Record the upstream revision or download date in
`provenance.yaml` when the snapshot is refreshed.
