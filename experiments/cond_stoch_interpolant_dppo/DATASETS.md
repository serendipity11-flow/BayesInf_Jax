# Prepared datasets

Split seed is 0 and validation fraction is 0.2. Splits are disjoint by complete
trajectory; no action chunk crosses an episode boundary.

| Dataset | Split | Episodes | Transitions | 4-step chunks |
|---|---:|---:|---:|---:|
| D4RL Door human-v2 | train | 20 | 5,444 | 5,384 |
| D4RL Door human-v2 | validation | 5 | 1,285 | 1,270 |
| DPPO Kitchen Complete | train | 15 | 2,915 | 2,870 |
| DPPO Kitchen Complete | validation | 4 | 765 | 753 |

Door source is Minari `D4RL/door/human-v2`. Its normalization statistics are
computed from training episodes only. Kitchen source is DPPO's official Google
Drive folder. That folder contains 19 training trajectories and an empty
published validation NPZ. The prepared Kitchen split preserves DPPO's official
already-normalized arrays and full-dataset min/max normalization to avoid
changing the baseline preprocessing.

Prepared paths:

* `data/dppo/door-human-v2/{train,val,normalization}.npz`
* `data/dppo/kitchen-complete-v0/{train,val,normalization}.npz`

Each directory contains `manifest.json` with source, split policy, and original
episode IDs.

## SHA-256

```text
9ae28104c104e70236bdf69c0f44bb4ae4b4a7932617d1dea54c4d9625249ced  door-human-v2/normalization.npz
1cecac9f69c7fdc6ce4a8799818b81be81cae68520da91203d16d46d60fdd968  door-human-v2/train.npz
0e7b6423c1722c9e5f0ac27fec83f3a45fcfba07f4aca0fe1d24a575310882d8  door-human-v2/val.npz
4bc1107ba2d51c747d511ca0dcbe30972322e466509a2ecf1eefbff3f17a738a  kitchen-complete-v0/normalization.npz
ea1723d4b8bb252e98dab6d7dd67a80861e0831803b727c3d4617985d8b5c360  kitchen-complete-v0/train.npz
8f9b92f3e0a565b4dcf7793df99e041bfc4ff0eabc19005fc390549ef69af54f  kitchen-complete-v0/val.npz
```

## Tests performed

* required keys, dimensions, dtypes, finite values, and trajectory sums;
* train/validation episode-ID disjointness;
* minimum trajectory length and exact four-step chunk counts;
* official DPPO `StitchedSequenceDataset` loading for all four NPZs;
* CondStochInterpolant forward/backward loss on every split;
* live `AdroitHandDoor-v1` reset and step.

The Kitchen simulator itself is not installed in the Door environment because
DPPO requires its conflicting legacy D4RL/Gym dependencies to remain isolated.
