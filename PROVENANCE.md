# Provenance record

## Public source scope

The repository owner authorized the public-repository cleanup and selected the
[MIT License](LICENSE) for original project source and documentation on
2026-10-27. Third-party rights remain separate and are recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[dependency-licenses.json](dependency-licenses.json).

The public source snapshot excludes the top-level `data/`, `evaluation/`,
`scripts/`, and `visualization/` directories, assistant plugins, bytecode, and
private experiments. It does not distribute trained weights or installed
dependency binaries. Removing files from the current tree does not remove earlier Git
history or copies already obtained by others.

## Historical implementation record

The implementation record from 2026-08-17 and 2026-08-18 identifies these sources
for the neural stochastic flow (NSF) work:

| Source | Recorded use |
|---|---|
| Naoki Kiyohara, Edward Johns, Yingzhen Li, “Neural Stochastic Flows: Solver-Free Modelling and Inference for SDE Solutions,” NeurIPS 2025, arXiv:2510.25769v1 | Mathematical and architectural specification, including the appendices |
| JAX, Equinox, Optax, Python, and NumPy documentation | Language, array, neural-module, and optimizer APIs |

That record reports an implementation from the paper equations rather than a
copy of an external NSF implementation. It records that the paper's source-code
link was not inspected during those milestones. This historical statement is
specific to that work; it is not a retrospective originality certification of
all pre-existing project files.

The record describes the NSF implementation, OU reference system, bridge
objectives, nested optimizer schedule, observation models, particle filtering,
controlled Duffing benchmark, and related tests as independently written under
human direction. It also records that the pendulum simulator and reconstruction
workflow used standard equations and locally generated trajectories. This
cleanup preserves that provenance without publishing the associated local data
or generated evaluation results.

## Third-party source and references

| Repository | Recorded revision | Role and licensing scope |
|---|---|---|
| [FrEIA](https://github.com/vislearn/FrEIA) | `421249652c478b3f240f124f69eb5777b6a76bf2` | FrEIA 0.2 source vendored on 2026-08-18 for RealNVP coupling comparison; full MIT license retained in `third_party/FrEIA/LICENSE` |
| [D4RL](https://github.com/Farama-Foundation/D4RL) | `89141a689b0353b0dac3da5cba60da4b1b16254d` | Optional upstream Git reference; Apache-2.0 code, separate asset/dataset terms |
| [DPPO](https://github.com/irom-princeton/dppo) | `cc7234ad7ff39a8f32de3af903606723a16f0648` | Optional upstream Git reference; MIT |
| [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | `71f066ad0be9cd271f7ed58c030243ef157af9f4` | Optional upstream Git reference; each model retains its own license, including BSD-3-Clause for ALOHA |

The earlier NSF record states that D4RL, DPPO, and Menagerie were inspected for
license/provenance purposes rather than used to implement the NSF method.
Separate experiments can use their optional upstream code or assets. The
project license does not supersede any of those upstream terms.

## AI assistance and review limits

The historical record identifies OpenAI Codex assistance for NSF Milestones 1–6,
tests, experiment entry points, and documentation under human direction on
2026-08-17 and 2026-08-18. Further AI assistance prepared the 2026-10-27 repository
cleanup, README, and license inventory under the owner's instructions.
AI-generated material is not evidence of originality, correctness, or rights to
third-party code. Contributors and maintainers remain responsible for reviewing
and maintaining their contributions.

The 2026-10-27 review checks the described public source tree, retained FrEIA
license, optional upstream references, and direct requirements against installed
metadata/license files and upstream license texts. It does not certify complete
human review of every historical contribution, complete provenance of every
optional dataset/model, or redistribution compliance for every transitive wheel,
GPU environment, container, or earlier Git commit. Scientific correctness and
reproducibility must be assessed for each claimed result separately.
