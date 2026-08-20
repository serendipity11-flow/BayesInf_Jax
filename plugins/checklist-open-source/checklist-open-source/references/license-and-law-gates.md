# License and law gates

Use these gates as a decision aid, not legal advice.

## Ownership gate

Confirm who can authorize release. Check employment duties, university or sponsor rules, invention assignment, contractor agreements, customer contracts, contribution agreements, joint authorship, and prior publications. A public repository or accessible file is not permission to reuse it.

## Component gate

Inventory source and binary dependencies, snippets, generated/vendor code, containers, build tools whose output is distributed, documentation, sample files, fonts, icons, images, audio/video, model weights, datasets, schemas, and test fixtures.

For each item record:

| Field | Required evidence |
| --- | --- |
| Identity | Name, version, commit/hash |
| Origin | Canonical URL or internal source record |
| Rights holder | Author or organization, when known |
| License | Canonical SPDX identifier and license text |
| Use | Copy, modification, link, plugin, service, build-only, bundled asset |
| Obligations | Attribution, NOTICE, source, change marking, patent/trademark terms |
| Decision | Include, replace, isolate, exclude, or escalate |

`NOASSERTION`, “found online,” or missing license means do not distribute until resolved.

## License-choice gate

- Permissive licenses generally maximize reuse but retain attribution and notice duties.
- Weak, strong, and network copyleft can impose different source-sharing duties depending on integration and distribution.
- Patent grants/termination, trademark rules, attribution, source offers, and modification notices can be material.
- “Compatible” depends on direction, combination, version, exceptions, and distribution—not only two license labels.
- A top-level license does not overwrite third-party licenses.
- Use exact canonical license text and SPDX expressions; do not invent a custom license unless legal counsel requires it.

Present choices to the authorized owner. Escalate non-standard licenses, dual licensing, copyleft ambiguity, or commercial licensing plans.

## Data and model gate

Evaluate software code, training/evaluation data, model architecture, weights, outputs, documentation, and hosted service terms separately. Check privacy, consent, database rights, data licenses, model licenses, acceptable-use restrictions, provenance, and whether restrictions prevent calling the result “open source.” Never assume a software license covers data or weights.

## Privacy and confidentiality gate

Search source and history for secrets, personal data, customer identifiers, internal hostnames/URLs, unpublished results, credentials, tokens, private keys, logs, telemetry, geolocation, faces/voices, and proprietary prompts. Redact or remove using an approved history-cleaning process, rotate exposed secrets, and document the response.

## Security and publication gate

Review vulnerability disclosure, supported versions, dependency risks, unsafe defaults, cryptography, command execution, network exposure, update mechanisms, and release signing. Obtain the organization’s required legal, IP, security, privacy, export, brand, and management approvals.

## AI-output gate

Require traceable human review. Investigate public-code matches and suspiciously specific or stylistically inconsistent output. The contributor must be able to explain, test, maintain, and license the contribution. AI tool claims about originality, licensing, security, or law are not evidence.

Inspect AI-created documentation as well as code. Remove malicious, defamatory, deceptive, irrelevant, prompt-injected, or governance-bypassing material before release, including unwanted `README` files. Do not remove legitimate records, attribution, licences, notices, security evidence, or required provenance under the label of “AI cleanup.”
