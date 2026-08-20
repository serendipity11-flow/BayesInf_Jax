# Git history and AI-artifact cleanup

Apply this process before first public release and after any suspected exposure.

## 1. Audit the complete publication surface

Inspect the working tree, full commit graph, all local and remote branches, tags, stashes, notes, submodules, release archives, CI artifacts, package registries, container layers, Git LFS objects, forks intended for release, and mirrored repositories.

Search for:

- credentials, tokens, private keys, connection strings, cookies, and signed URLs;
- personal, customer, confidential, regulated, export-controlled, or unpublished information;
- internal hosts, paths, tickets, prompts, logs, model outputs, or environment dumps;
- copied or generated material with unknown provenance or incompatible licensing;
- large binaries, datasets, models, archives, screenshots, notebooks, and deleted files still present historically;
- unexpected `README*`, `AGENTS*`, instruction, report, manifesto, issue-template, or generated documentation files.

Use more than filename searches: inspect content, diffs, object lists, file history, and scanner findings. Record exact object IDs, commits, refs, paths, and exposure scope without reproducing secrets in ordinary reports.

## 2. Classify suspicious AI-generated files

Do not delete a file merely because AI helped create it. Keep content that is accurate, relevant, authorized, lawful, and maintainable.

Remove or quarantine content that is:

- malicious, defamatory, threatening, harassing, deceptive, or impersonating someone;
- a prompt-injection artifact or an instruction for an agent rather than project documentation;
- unrelated promotional material, fabricated claims, fake citations, or an AI-generated “hit piece”;
- designed to bypass review, alter repository visibility, exfiltrate data, weaken security, or conceal provenance;
- unauthorized disclosure or third-party material that cannot be lawfully distributed.

Do not remove licences, notices, attribution, contributor records, security reports, audit evidence, or provenance simply because they are inconvenient. Escalate conflicts between cleanup and preservation obligations.

## 3. Choose the least destructive correction

Use this order:

1. Remove an uncommitted or unpublished current-tree artifact normally.
2. Add a corrective commit when historical retention is acceptable and no prohibited data remains exposed.
3. Rewrite only the exact affected paths/objects/refs when prohibited content must be purged.
4. Create a new clean repository from a reviewed snapshot when the history is broadly contaminated, ownership is mixed, or rewrite verification is unreliable.

Deleting the `.git` directory or force-pushing a new root is not a default cleanup method. Preserve useful authorship and provenance whenever lawful and safe.

## 4. Gate destructive cleanup

Before rewriting or deleting history:

- rotate/revoke exposed credentials immediately; history cleanup does not invalidate a secret;
- pause publication and automated releases;
- identify exact affected paths, objects, branches, tags, remotes, forks, packages, and artifacts;
- create a restricted, encrypted backup or evidence record when organizational, legal, incident-response, or audit policy requires it;
- obtain explicit approval naming the operation and target refs;
- communicate the coordination plan to maintainers and affected collaborators.

Use an approved history-rewriting tool appropriate to the repository. Never run broad destructive commands against an unresolved path, repository root, home directory, or all refs by default.

## 5. Verify the result

After cleanup:

- inspect all intended published refs and unreachable-object handling;
- repeat secret, privacy, licence, provenance, and malware/content scans;
- confirm suspicious AI artifacts are absent from the working tree and published history;
- ensure required licences, notices, attribution, authorship, and audit records remain;
- rebuild packages, containers, documentation, SBOMs, checksums, signatures, and releases;
- verify remote mirrors, release assets, caches, registries, and public forks separately;
- document the reason, approver, tool/method, affected refs, old-to-new mapping when needed, credential rotation, and verification evidence;
- require collaborators to re-clone or follow an exact reset procedure to prevent reintroducing removed objects.

Mark release status `BLOCKED` until the audit and verification are complete.
