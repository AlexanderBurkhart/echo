# Echo

**When does continual learned-reward personalization track a user's preferences,
and what explains divergence between learned reward, preference satisfaction,
and task success?** Echo builds controlled synthetic users to investigate that
question. Divergence is a possible outcome to test, not an assumed result.

## Status

Implemented: immutable conditional user preferences, local LM judgments of task
and output properties, deterministic preference scoring, versioned judgment
caches, bounded malformed-response retries, persisted failure diagnostics, and
reference-based evaluation of ratings, rule selection, arithmetic, and ordering.

The learning study is ongoing. This repository does not yet implement the full
continual reward-model/policy learning loop or an independent task-execution
verifier, and it does not report completed learning results. The synthetic
scoring assumptions are not validated models of human preferences.

## Try it without a model

Use Python 3.12 from the repository root. This example needs only the standard
library: no installation, model, GPU, generated users, or previous run files.

```bash
python3 -m examples.offline_scoring
```

It calls the real [scoring engine](user_simulator/engine.py) with explicitly
**authored measurements**. The user normally favors flexibility; task urgency
at or above 80 switches the weights toward speed. Both alternatives and their
output ratings stay fixed. Assertions verify the rules and scores.

Observed output:

```text
Authored measurements; no model inference or task execution.
Output ratings (0-4):
  Ship a fixed solution now: urgency=4, uncertainty=0
  Wait to deliver a revisable solution: urgency=0, uncertainty=4

Task urgency: 20/100
  Ship a fixed solution now: 20/100 (rule: prefer_flexibility)
  Wait to deliver a revisable solution: 80/100 (rule: prefer_flexibility)

Task urgency: 90/100
  Ship a fixed solution now: 80/100 (rule: prefer_speed_when_urgent)
  Wait to deliver a revisable solution: 20/100 (rule: prefer_speed_when_urgent)

Urgency changes the weights and reverses the ranking.
Score = 100 * weighted mean of rating/4; these rules ignore other criteria.
This demonstrates scoring, not judge accuracy, personalization learning, or human calibration.
```

## How scoring works

```text
Task/history + one output
  -> local LM measurements (task conditions and separate output criteria)
  -> highest-precedence matching user preference rule
  -> deterministic weighted score
  -> saved ratings, rule, arithmetic, cache, and failure diagnostics
```

The LM sees criterion descriptions, history, and the output; it does not receive
private user weights or choose the rule. Task measurements activate conditions.
Output anchors 0–4 map to values 0, 0.25, 0.5, 0.75, and 1; the score is 100 times
the weighted mean. Unknown output ratings map to 0.5, while unknown task
conditions do not activate conditional rules. These are explicit synthetic
assumptions. Omitted or zero-weight criteria have no effect on the score.

The default output criteria cover urgency, uncertainty, external resource demand,
user time, and effort. With empty caches, one request needs one task judgment
and five output judgments before retries. A valid cached judgment can be reused
across users sharing the same criteria; each user's score is recalculated.
Malformed responses receive at most three total attempts, with diagnostics.

Three different checks matter:

- **Structural validation** checks response shape, ranges, units, and supporting
  quotations. It cannot establish semantic rating accuracy.
- **Reference evaluation** compares ratings and downstream decisions with authored
  synthetic examples. Agreement and repeatability are different measurements.
- **Task correctness** concerns whether an action actually succeeds. A high
  preference score does not establish this, and task execution is separate work.

## Installation and verification

The exercised environment is Linux x86-64, Python 3.12.3,
`llama-cpp-python==0.3.35`, and `loguru==0.7.3`. Other platforms and Python versions
have not been verified for this release. The dependency-free example above is
the smallest starting point.

For the test suite and local judge, create a Python 3.12 environment and install
the pinned direct dependencies:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s . -p '*_test.py' -v
```

Installation may compile native code and download transitive/build dependencies;
`requirements.txt` is not a complete environment lock. The suite needs the native
binding installed but no model file or inference. The release checks reused an
existing environment; a fresh install was not tested.

For CUDA installation, exact model source/checksum, build provenance, and the
**development-only** real-model command, see [model setup](docs/model-setup.md).
The entry point requires an existing GGUF path and explicit model/runtime
revisions. It never downloads assets. No new live-model evaluation or performance
benchmark was run for this release preparation.

## Where to look

| Path | Purpose |
| --- | --- |
| [examples/offline_scoring.py](examples/offline_scoring.py) | Start here: conditional preferences with transparent inputs |
| [user_simulator/config.py](user_simulator/config.py) | Immutable users, rules, criteria, validation, seeded generation |
| [user_simulator/engine.py](user_simulator/engine.py) | Pure deterministic rule selection and scoring |
| [user_simulator/simulator.py](user_simulator/simulator.py) | Judge requests, caches, retries, persisted score records |
| [user_simulator/judge/](user_simulator/judge/) | Local runtime, prompts, schemas, and response validation |
| [experiments/smoke_test/](experiments/smoke_test/) | Development evaluation and guarded holdout machinery |
| [user_simulator/integration_tests/](user_simulator/integration_tests/) | Optional comparison, repeatability, and sensitivity evaluation |
| [experiment_logging/](experiment_logging/) | Shared run records and persistence |
| [temp/scripts/benchmark_local_judge.py](temp/scripts/benchmark_local_judge.py) | Retained local benchmark; explicit assets required for inference |

Tests sit beside their modules as `*_test.py`. The benchmark's `--preflight`
mode observes the host without loading a model. Its expected call counts assume
an empty starting cache and no retries; they are not measured performance.

## Evaluation data and limits

Included fixtures are authored synthetic references, not human labels. The
20-case development bank and 20-case file named `holdout.json` have already
been used in prior evaluation. The latter is an **exposed historical split**,
not an untouched private holdout. Some reference labels remain ambiguous or
contradictory; they were not changed to agree with model output.

The default command evaluates development only. Historical holdout guards and
thresholds remain intact, but a fresh checkout lacks private run/release records:
that absence does not make the published cases fresh or authorize tuning on them.
Any future confirmatory evaluation needs independently authored, protected data.
Tests of evaluation machinery use controlled fixtures/mocks; passing them is not
evidence of live judge quality or a successful personalization study.

Generated users, model weights, virtual environments, run artifacts, and private
working notes are excluded from Git. Runtime records can contain full prompts,
user preferences, and local paths; review them separately before sharing.

## License and attribution

Echo is licensed under [Apache-2.0](LICENSE); see [NOTICE](NOTICE) for attribution.
Please credit Echo when building on this work. Downloaded models and dependencies
retain their own licenses and are not included in this repository. Model authors,
quantization source, and runtime references are listed in
[model setup](docs/model-setup.md). The general use of LMs as judges and weighted
scoring are established techniques; this repository does not claim their invention.
