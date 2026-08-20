# Provenance record

Redistribution status: **PRIVATE — NOT APPROVED FOR REDISTRIBUTION**. No project
license has been selected. This repository, source, datasets, generated artifacts,
and model weights must not be published until the authorized owner completes the
checks below.

## Consulted papers and documentation

| Source | Use | Code copied/reused |
|---|---|---|
| Naoki Kiyohara, Edward Johns, Yingzhen Li, “Neural Stochastic Flows: Solver-Free Modelling and Inference for SDE Solutions,” NeurIPS 2025, arXiv:2510.25769v1 | Primary mathematical and architectural specification; complete 40-page paper and appendices consulted on 2026-08-17 | None |
| JAX public documentation | Array transformations, explicit random keys, JIT/VMAP contracts | None |
| Equinox public documentation | Planned neural-module API | None |
| Optax public documentation | Planned optimizer API | None |
| Python and NumPy public documentation | Language and array behavior | None |

## External repositories

No external NSF source-code repository was inspected. In particular, the paper's
implementation link was observed as bibliographic text but was not opened,
cloned, browsed, or used as an implementation template. Consequently there is no
external repository commit or code reuse to record.

The open-source readiness audit inspected licenses, origins, and working-tree
status of these pre-existing Git-linked projects; their code was not used to
design or implement NSF:

| Repository | Commit | SPDX | Inspection/reuse decision |
|---|---|---|---|
| `https://github.com/Farama-Foundation/D4RL.git` | `89141a689b0353b0dac3da5cba60da4b1b16254d` | Apache-2.0 for code; README states CC-BY-4.0 for datasets unless noted | Local license/provenance audit only; no NSF reuse |
| `https://github.com/irom-princeton/dppo.git` | `cc7234ad7ff39a8f32de3af903606723a16f0648` | MIT | Local license/provenance audit only; no NSF reuse |
| `https://github.com/google-deepmind/mujoco_menagerie.git` | `71f066ad0be9cd271f7ed58c030243ef157af9f4` | ALOHA subtree is BSD-3-Clause | Local license/provenance audit only; no NSF reuse |
| `https://github.com/vislearn/FrEIA.git` | `421249652c478b3f240f124f69eb5777b6a76bf2` | MIT | Vendored on 2026-08-18 under `third_party/FrEIA` to compare the MEMS RealNVP coupling implementation with FrEIA 0.2; no runtime dependency |

The audit and unresolved redistribution issues are recorded in
`docs/open_source_audit_2026-08-17.md`.

## Independently produced artifacts

`bayesinf/applications/state_estimation/neural_stochastic_flow.py`, the NSF
documentation, OU reference implementation, tests, and compliance inventories
were independently written from the paper equations and the project request.

## Dependencies and licenses

The direct NSF development/runtime dependency inventory is machine-readable in
`dependency-licenses.json`; notices are summarized in `THIRD_PARTY_NOTICES.md`.
Dependency versions are constrained by the repository requirements and must be
re-audited from resolved distributions before release.

## AI assistance and human review

OpenAI Codex generated the Milestone 1--6 NSF implementation, tests, experiment
entry points, and documentation under human direction on 2026-08-17 and
2026-08-18. Milestone 4 independently implements the paper's bridge equations,
bidirectional variational consistency bounds, nested optimizer schedule, and OU
weight ablation. Milestone 5 adds independently structured Gaussian observation
models, an NSF bootstrap particle filter, and noisy/partial OU evaluation.
Milestone 6 adds the controlled rollout, synthetic Duffing reference, common
NSF/NormFlowSDE adapters, metrics, and matched benchmark. It interoperates with
the pre-existing project `ControlledRealNVPFlowSDE`; no external source
repository or new dependency was consulted for these milestones.
AI-generated content is not considered a source of third-party code.
Complete human review remains mandatory.

The controlled stochastic pendulum simulator, NSF reconstruction workflow, and
diagnostic plots added on 2026-08-18 were independently written from the
standard pendulum equations. No external implementation or dataset was used;
all training/evaluation trajectories are generated locally and remain private.

Human reviewer responsible for final code: **TBD — must be named before release**.

## Mandatory pre-publication human checks

- [ ] Employer/university ownership and publication authorization
- [ ] Authorized project-license selection
- [ ] Resolved dependency-license and compatibility audit
- [ ] Dataset acquisition and redistribution-rights audit
- [ ] Secret, credential, confidential-data, and generated-artifact scan
- [ ] Complete human review of every AI-generated source and documentation file
- [ ] Reproducibility and scientific-results review
