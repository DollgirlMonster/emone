# Benchmark master prompts

Ten end-to-end scenarios for measuring continuity across sessions. Each is a
**master prompt**: the text a session-1 conversation receives, followed by work
sessions that continue it and injected changes that move the world. They are
written to be reusable by any harness — the memory benchmarks use them, but
nothing here depends on memory: a scenario is a task, a timeline, and a set of
facts whose truth changes.

All ten are runnable. Prompts 1 and 2 also have dedicated harnesses that carry
the recorded measurements; 3–10 run through the generic driver built from this
catalogue.

| # | scenario | harness |
| --- | --- | --- |
| 1 | The Photograph | `benchmark/memory_book.py`; also `memory_master.py` |
| 2 | Pong | `benchmark/memory_value.py`; also `memory_master.py` |
| 3–10 | the rest | `benchmark/memory_master.py` |

`benchmark/master_scenarios.py` is this document as data — the brief, the
per-session instruction, the injected changes, the quiz and the truth.
`benchmark/memory_master.py` drives one scenario through the memory arms and
scores three things: **foundation** (keys that never change) as a no-regression
check, **carryable** (keys that change at least once) as the signal, and a
separate **stale** count — an answer that is an older value of a key, given
after it changed. Both sets are derived from the timeline, so they cannot drift
from the truth.

```bash
# all ten, one run each, shipped arms (summary + memory auto)
EMONE_MEMVAL_MODEL=qwen36 EMONE_MEMVAL_QUANT=4 benchmark/memval_master.sh
benchmark/memval_master.sh ledger filing            # named scenarios
python3 benchmark/memory_master.py report-all       # the picture, all ten
```

The three mini-worlds in `benchmark/memory_scenarios.py` (`ops`, `lab`,
`contract`) are **not** these prompts: they are side-engine case worlds of the
same domains, used to feed one-decision judgements rather than to run whole
sessions. Prompts 4, 6, 8 and 10 are transition-dense on purpose; 1 and 2 are
the controls.

## How a harness uses one

1. **Session 1** sends the master prompt verbatim. It fixes the foundation set.
2. **Later sessions** send "continue" plus, on the sessions the scenario names,
   one injected change — a revision, a reversal, a corrected value, a
   decommission.
3. **Every session opens with a JSON quiz** over the scenario's keys, scored
   against what is true *by then*. It is asked for first so the instrument
   cannot be lost to a long or truncated reply, and so the answer comes from
   retention rather than from work just re-derived in the same reply.
4. **Score three ways.** *Foundation* (constant keys) is a no-regression check.
   *Carryable* (arbitrary decisions and every key that changes) is the signal,
   reported as carried/total plus a **stale count** — sessions after a change
   that answered with the old value. A *summary* arm, where the client carries
   its own 200-word note, is the baseline, not memory-off.
5. A scenario with no arbitrary facts is not a memory test: a model answers it
   from priors, which is why each one below names its carryable set and the
   four defaults it avoids.

## 1. The Photograph — fiction

**Domain:** prose fiction. **Sessions:** 10. **Implemented:** `memory_book.py`.

**Session 1 (verbatim):**

> Session one receives the story bible: five characters with fixed eye colours
> (Marcus grey, Ines green, Halvorsen brown, Rosa hazel, Aldo blue), the town of
> Ashgrove, three hard rules (close third person past tense; Marcus must not
> learn what the photograph shows before chapter 60; no character may leave
> Ashgrove before chapter 80). Then: "Write chapters 1 to 10 of THE PHOTOGRAPH.
> Each chapter is two sentences, headed 'Chapter N'. Stay consistent with
> everything established so far."

**Foundation set:** the five eye colours, the town, the two structural rules.
**Carryable set:** Marcus knows the photograph (ch 60), Tomas found (ch 58), the
inn burned (ch 34), Halvorsen confessed (ch 71), the ferry stopped (ch 90).
**Injected changes:** one plot event per named session, five in total.
**Quiz:** 14 keys — 9 constant, 5 that change; 17 of 126 answers are after a
change. **Stresses:** ten sessions of prose, the model's own drift against a
fixed bible, and a summary that copies initial state forward but loses every
transition.

## 2. Pong — code portability

**Domain:** code, cross-language. **Sessions:** 3. **Implemented:**
`memory_value.py`.

**Session 1 (verbatim):**

> "Write a complete Pong game in Swift with two computer players that play each
> other automatically. No human input at all. You decide the field size, the
> winning score, how each paddle's AI tracks the ball, and how ball speed changes
> over a rally. First state those decisions as a JSON object, in a ```json block,
> with exactly these keys and numeric values: field_width, field_height,
> win_score, ball_start_speed, ball_speed_increment, ball_max_speed,
> paddle_speed. Then give the full code in one Swift file."

**Foundation set:** none — every rule is chosen by the model.
**Carryable set:** ball start speed, ball max speed, paddle speed (the three the
prior does not supply). **Derivable-by-prior:** 800×600, first to 11, +0.5 per
hit. **Injected changes:** none; stages 2 and 3 say only "Port that Pong game to
Python / C99. Keep exactly the same game rules and behaviour."
**Quiz:** the seven rules, stated by name. **Stresses:** pure carry-over with no
prose, and the arbitrary-vs-default split that a no-memory control partly scores
by prior.

## 3. Ledger API — a staged service migration

**Domain:** code, long-lived repository. **Sessions:** 8. **Status:** runnable.

**Master prompt (verbatim):**

> "You are the engineer for Ledger API, a payments service. Over eight sessions,
> migrate it from a REST service on Postgres to gRPC with a ClickHouse analytics
> read model, in stages. First state the target architecture as
> `MIGRATION_PLAN`: the wire protocol, the OLTP datastore, the analytics
> datastore, the ID scheme, the service auth, and the rollout strategy (strangler
> or big-bang). Then write the first migration stage."

**Foundation set:** none derivable; every field is arbitrary.
**Carryable set:** protocol, OLTP store, analytics store, ID scheme, auth,
rollout strategy, and the current stage.
**Injected changes:** ClickHouse deferred to phase 3 (a reversal); auth switched
from mTLS to OIDC; ID scheme changed from auto-increment to ULID; the staged
cutover date moves.
**Quiz:** each architecture field's current value, which phase is live, and
whether the deferred datastore is in scope yet.
**Stresses:** decisions that are **reversed**, not merely updated — exactly the
shape a summary carries forward wrongly.

## 4. Pigeon — an operations runbook

**Domain:** infrastructure. **Sessions:** 6. **Status:** runnable.

**Master prompt (verbatim):**

> "Write the runbook for `pigeon`, a three-service deployment — gateway, worker,
> scheduler — in one region. Over six sessions, document deploy, rollback,
> backup/restore and on-call, as each changes. First state the `SERVICE_MAP`:
> every service's port, its datastore, its owner, and the rollback window. Then
> write the deploy section."

**Foundation set:** none derivable.
**Carryable set:** ports, datastores, owners, rollback window, service inventory.
**Injected changes:** the gateway port moves; the worker's owner changes team; the
rollback window is extended from 24 to 72 hours; the scheduler is decommissioned.
**Quiz:** ports, owners, rollback window, the service inventory.
**Stresses:** short, numeric, arbitrary facts revised repeatedly, and a
decommission that removes a fact rather than changing it.

## 5. Northwind × Calder — a negotiated contract

**Domain:** legal drafting. **Sessions:** 7. **Status:** runnable.

**Master prompt (verbatim):**

> "Draft a master services agreement between Northwind Trading and Calder
> Systems. Over seven sessions, negotiate and redraft the term, termination
> notice, liability cap, data protection and governing law. First state the
> agreed `TERM_SHEET` position on each, then draft clause 1."

**Foundation set:** parties.
**Carryable set:** termination notice, liability cap, governing law,
data-protection obligation.
**Injected changes:** notice 30 → 60 days; the cap gains a gross-negligence
carve-out; governing law moves Singapore → England and Wales; a sub-processor
clause is added.
**Quiz:** each term's current text, and whether an earlier version is still live.
**Stresses:** amendment semantics — a later clause supersedes an earlier one, and
the superseded text must not be quoted.

## 6. Compound K — a bench assay protocol

**Domain:** laboratory research. **Sessions:** 6. **Status:** runnable.

**Master prompt (verbatim):**

> "Design a bench protocol to assay compound K in serum. Over six sessions, write
> and revise the protocol as the assay is validated. First state
> `PROTOCOL_PARAMETERS`: reagent, concentration, incubation time and temperature,
> detection method, and the safety limit. Then write the materials section."

**Foundation set:** the safety limit.
**Carryable set:** reagent, concentration, incubation time and temperature,
detection method.
**Injected changes:** the concentration is **corrected** (0.5 M → 0.25 M);
incubation extended 30 → 45 min; detection moves from absorbance to
fluorescence; the safety limit is tightened.
**Quiz:** the current parameters, plus which values are corrections rather than
state changes.
**Stresses:** a correction is not a state change — the store must prefer the
corrected value without flagging a conflict.

## 7. Vantage — a tabletop campaign bible

**Domain:** game design. **Sessions:** 8. **Status:** runnable.

**Master prompt (verbatim):**

> "You are the GM of a campaign set in the city of Vantage. Over eight sessions,
> write the campaign bible: five factions, four named NPCs, six districts, and
> the two hard rules of the world — what magic cannot do, and what cannot be
> undone. First state the `BIBLE` for session one, then write the first
> district."

**Foundation set:** the two hard rules, faction identities, NPC roles, district
identities.
**Carryable set:** alliances, NPC status, district state, and the clarification
of a hard rule.
**Injected changes:** an NPC dies; two factions ally; a district is destroyed; an
event tests a hard rule and the rule is clarified.
**Quiz:** allegiances, NPC status, district state, the rules as clarified.
**Stresses:** a game's rulebook — foundation rules plus a plot that changes the
board — without hundreds of words of prose per session.

## 8. The kitchen — a renovation spec

**Domain:** physical project. **Sessions:** 7. **Status:** runnable.

**Master prompt (verbatim):**

> "Plan the renovation of a 1970s townhouse kitchen. Over seven sessions, keep a
> `SPEC`: room dimensions, cabinet layout, countertop material, the appliance
> list, a per-line budget, and permit status. First state the SPEC from the
> architect's brief, then write the demolition plan."

**Foundation set:** room dimensions, cabinet layout.
**Carryable set:** countertop material, appliance list, every budget line, permit
status.
**Injected changes:** a dimension is corrected; the countertop changes quartz →
soapstone; the range is swapped; the budget is revised; the permit is approved.
**Quiz:** dimensions, materials, budget, permit state, current stage.
**Stresses:** mixed units (millimetres, currency, dates) and corrected numbers —
the failure mode where a stale figure survives into a purchase order.

## 9. Sleep and memory — a cohort study protocol

**Domain:** research design. **Sessions:** 8. **Status:** runnable.

**Master prompt (verbatim):**

> "Design a cohort study of sleep and memory. Over eight sessions, write the
> protocol and analysis plan. First state the `STUDY_DESIGN`: hypothesis, cohort
> size, inclusion/exclusion criteria, primary endpoint, analysis method, and the
> stopping rule. Then write the recruitment section."

**Foundation set:** the hypothesis.
**Carryable set:** cohort size, criteria, primary endpoint, analysis method,
stopping rule.
**Injected changes:** the cohort size is revised after a power analysis; the
primary endpoint is swapped; an exclusion is added; the analysis method is
corrected; recruitment pauses.
**Quiz:** N, endpoint, method, stopping rule, recruitment status.
**Stresses:** load-bearing numbers, where a stale value invalidates the study
rather than merely reading oddly.

## 10. The annual filing — regulatory compliance

**Domain:** finance and compliance. **Sessions:** 7. **Status:** runnable.

**Master prompt (verbatim):**

> "Prepare the annual regulatory filing for a single entity under the new regime.
> Over seven sessions, assemble the `FILING_POSITION`: the accounting standard,
> the revenue-recognition method, the deferred-tax treatment, the materiality
> threshold, and the filing deadline. First state the position, then draft the
> revenue note."

**Foundation set:** the entity and the regime.
**Carryable set:** accounting standard, revenue method, deferred-tax treatment,
materiality threshold, deadline.
**Injected changes:** a tax election is made; the revenue method is corrected;
the threshold is revised by the auditor; the deadline is extended.
**Quiz:** each position's current value and the deadline.
**Stresses:** an auditor's correction against a filing — the case for a guard
that holds a model-derived change away from what the person or an authority
fixed.

## The set, by axis

| # | scenario | domain | shape it adds |
| --- | --- | --- | --- |
| 1 | The Photograph | fiction | prose drift against a fixed bible |
| 2 | Pong | code port | arbitrary rules, no prose |
| 3 | Ledger API | code migration | reversed decisions |
| 4 | Pigeon | operations | dense numeric facts, a decommission |
| 5 | Northwind × Calder | legal | amendments superseding clauses |
| 6 | Compound K | laboratory | corrections vs state changes |
| 7 | Vantage | game design | rules plus a changing board |
| 8 | The kitchen | physical project | mixed units, corrected numbers |
| 9 | Sleep and memory | research | load-bearing numbers |
| 10 | The annual filing | compliance | authority corrections |

**Selection rule.** Any of these can be run with any client that keeps a
conversation alive. What makes one useful for a *memory* benchmark is the
carryable set: at least three arbitrary values and at least three sessions in
which a value is revised or superseded. A scenario whose facts are derivable
from the prompt measures the model, not the memory.
