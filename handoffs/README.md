# Handoffs

One file per task that is being passed to someone else. Each is written to be
picked up cold: what the task is for, what already exists, what to build, how
to know it worked, and what would make it a negative result worth reporting.

| # | file | owner | blocked on |
|---|---|---|---|
| 02 | [Open-weight generators on Open Images V7](02-open-weight-generators-on-open-images.md) | unassigned | the Open Images harvest finishing |
| 03 | [Commercial APIs on Open Images V7](03-commercial-apis-on-open-images.md) | unassigned | a costing decision (§3.2) |
| 08 | [The ablation rungs, one at a time](08-ablation-rungs.md) | unassigned | nothing — a bank and an eval bank are on disk |

## House rules these all inherit

**One variable.** A rung, a preset, or a generation run differs from its
control in exactly one thing. If two things change and the number moves, the
run has told you nothing. `tests/test_rung_ladder.py` enforces this for rungs
by failing the build when a neighbouring pair differs in more than one flag.

**The manifest is frozen.** Feature banks index it positionally and verify a
`manifest_sha256` over the ordered `rel_path` column. Rebuilding the manifest
orphans every bank on disk. New data goes into a NEW preset and a NEW manifest,
never into the frozen one.

**Report the negative result.** Every one of these tasks can come back "this
did not work". That is a publishable outcome here and it is written into the
acceptance criteria below. Do not quietly reshape a task until it succeeds.

**Confounds before conclusions.** This corpus leaks the label through
sharpness, noise floor and JPEG history. Any new image source gets
`scripts/gate_confounds.py` run over it before a model trains on it. A source
that scores well because it is sharper than the fakes is measuring our
dataset, not the world.
