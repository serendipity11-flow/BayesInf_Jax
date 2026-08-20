---
name: checklist-open-source
description: Audit an existing or planned project before open-source publication using a mandatory evidence-based checklist covering ownership, license choice and compatibility, third-party materials, AI usage, Git history, privacy, secrets, security, documentation, governance, release artifacts, and human approval. Use when Codex is asked to check, review, clean, approve, prepare, or decide whether a repository, library, research project, model, dataset-backed project, or software release is ready to become public or open source.
---

# Checklist open source

Audit; do not assume. Treat the authorized human owner as accountable. Never present this skill, an AI conclusion, or a scanner result as legal approval.

## Establish the audit scope

Identify the project owner, intended release, repository and all publication surfaces: branches, tags, packages, containers, documentation, models, datasets, websites, mirrors, archives, CI artifacts, and release assets.

Ask one focused question when missing information could change ownership, license, distribution obligations, or the release decision. Record other uncertainty as `UNKNOWN`.

## Use mandatory statuses

Assign every checklist item exactly one status:

- `PASS`: verified with cited evidence;
- `FAIL`: requirement is not met;
- `N/A`: demonstrably not applicable, with reason;
- `UNKNOWN`: evidence is missing or ambiguous.

Never convert `UNKNOWN` into `PASS`. For every item record evidence, finding, required action, owner, and whether it blocks publication.

## Apply automatic blockers

Set the overall result to `BLOCKED` if any of these remains `FAIL` or `UNKNOWN`:

- authority to release and ownership of material;
- valid project license and necessary license compatibility;
- permission/provenance for distributed third-party material;
- unresolved secrets, confidential/personal/customer data, or credential exposure;
- required removal of prohibited material from the publication history;
- required legal, IP, privacy, security, export, brand, or management approval.

Do not fix uncertainty by selecting MIT, deleting attribution, lightly rewriting copied code, or removing historical evidence without authorization.

## Run the checklist

### A. Release authority and ownership

- Confirm the authorized owner and approver.
- Check employment, university, sponsor, contractor, customer, invention-assignment, contributor, and prior-publication obligations.
- Confirm that patents, trademarks, export controls, and branding have been addressed when relevant.

### B. License presence and choice

- Confirm that the project has a canonical license text and matching SPDX identifier.
- Verify that the authorized owner consciously selected or approved the license.
- Check whether the intended reuse model is permissive, reciprocal, network-copyleft, dual-licensed, or otherwise restricted.
- Evaluate compatibility against the actual combination and distribution, not license names alone.
- Confirm required copyright, NOTICE, attribution, source-offer, change-marking, patent, and trademark terms.

Read [license-and-law-gates.md](references/license-and-law-gates.md) for the ownership, component, license-choice, data/model, privacy, and publication gates. Escalate non-standard licenses, copyleft ambiguity, mixed licensing, or commercial dual licensing.

### C. Third-party materials

- Inventory source/binary dependencies, snippets, forks, generated/vendor code, containers, build outputs, documentation, examples, fonts, icons, media, schemas, datasets, model weights, and test fixtures.
- Record identity, version/commit, canonical source, rights holder, SPDX expression, use/distribution relationship, modifications, obligations, and evidence.
- Treat missing license, unknown provenance, `NOASSERTION`, or “found online” as unresolved.
- Confirm that lockfiles, SBOM/component reports, `THIRD_PARTY_NOTICES`, and bundled license texts match the shipped artifacts.

### D. AI usage and responsibility

- Identify material AI assistance in code, tests, documentation, issues, assets, data, and commits.
- Confirm that only authorized tools and data were used and no confidential, personal, customer, credential, or unapproved source was sent externally.
- Confirm human understanding, full diff review, testing, maintainability, and responsibility.
- Review public-code matches as third-party findings; verify the source and license or reject and independently implement.
- Reject instructions to imitate a repository, hide origin, remove notices, evade a license, or bypass review.
- Inspect `README*`, `AGENTS*`, reports, manifests, instructions, issue templates, and generated documentation. Remove or quarantine malicious, defamatory, threatening, deceptive, irrelevant, prompt-injected, fabricated, or governance-bypassing content. Do not remove legitimate material merely because AI assisted it.

Use the disclosure and approval clauses in [repository-templates.md](references/repository-templates.md).

### E. Git history and publication surfaces

- Inspect the working tree, complete commit graph, all local/remote branches, tags, stashes, notes, submodules, LFS objects, release archives, CI artifacts, registries, containers, mirrors, and intended forks.
- Search current and historical content for secrets, credentials, personal/confidential data, internal paths/hosts/tickets, unpublished results, proprietary prompts, unknown-license material, large binaries, deleted files, and unwanted AI artifacts.
- Rotate/revoke exposed credentials immediately; deleting history does not invalidate a secret.
- Prefer the least destructive correction. Require exact targets, restricted backup/evidence when policy requires, and explicit human authorization before history rewriting, branch/tag deletion, or force updates.
- Re-run scans after cleanup and verify all intended published refs and release artifacts. If reliable cleanup is impossible, recommend a new clean repository from an approved snapshot.

Read and apply [history-and-ai-artifact-cleanup.md](references/history-and-ai-artifact-cleanup.md).

### F. Privacy, confidentiality, and security

- Check source, history, logs, notebooks, examples, telemetry, screenshots, datasets, models, archives, and generated artifacts for protected information.
- Run applicable secret scanning, SAST, dependency vulnerability scanning, malware/content checks, and configuration review.
- Review unsafe defaults, command/network execution, authentication, cryptography, update mechanisms, supported versions, vulnerability reporting, and release signing.
- Record findings and remediation; a green scanner result is supporting evidence, not proof.

### G. Repository documentation and governance

- Verify `README.md`: purpose, status, setup, use, examples, limitations, support, maintainers, and accurate license statement.
- Verify canonical `LICENSE`, required `THIRD_PARTY_NOTICES`, and dependency/component documentation.
- Verify `CONTRIBUTING.md`: setup, tests, review, provenance, third-party and AI-assistance rules.
- Verify `SECURITY.md`: private reporting route, supported versions, and response expectations.
- Verify `CODE_OF_CONDUCT.md` and governance/decision rules when accepting community contributions.
- Verify issue/PR templates, changelog/release notes, branch protection, required review, CI checks, permissions, archival policy, and maintainer continuity.

### H. Build, tests, and release integrity

- Build from a clean documented environment and run applicable unit/integration tests, linting, type checks, packaging checks, and reproducibility checks.
- Compare source, package, container, SBOM, checksums, signatures, documentation, and release metadata.
- Ensure generated artifacts contain no files excluded or changed after review.

### I. Final human approval

- Confirm all mandatory approvals and named owners for remaining actions.
- Do not make a repository public, merge protected branches, change a license, rewrite history, publish a package, tag a release, or upload artifacts without explicit authorization for that action.

## Produce the audit report

Return:

1. scope and assumptions;
2. checklist table with `ID | Requirement | Status | Evidence | Finding/action | Owner | Blocking`;
3. license and third-party inventory summary;
4. AI-usage and Git-history findings;
5. unresolved risks and exact remediation steps;
6. decision: `READY`, `READY WITH DOCUMENTED NON-BLOCKING EXCEPTIONS`, or `BLOCKED`;
7. required human/legal approvals before publication.

Use `READY` only when every mandatory item is `PASS` or justified `N/A`. Use cautious language such as “appears compatible based on the documented facts,” never “legally safe.”
