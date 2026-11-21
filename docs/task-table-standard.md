# One task table: the standard

A tracker is a decision queue, not a diary: it holds only what is not resolved
yet, and the state of each item is a *field*, never the section it sits in. This
is the shape emone uses — the table itself is the wiki's
[Project Tracker](https://github.com/Pummelchen/TinyTitan/wiki/Project-Tracker) —
and the block below is the reusable form. Paste it into a new project or a new
agent session to get the same table.

```text
Maintain exactly ONE task table for this project: it is the only place open work
lives. Do not create Open/Blocked/Parked sections, a second backlog, or a status
heading — status is a column.

HARD RULES
1. One table under `## Tasks`. One row = one independently closable outcome; an
   epic is a project, not a row.
2. IDs are stable and never reused. Closing deletes the row; the gap is correct
   and keeps every old reference (commit, release note, issue) valid forever.
3. History does not live here. What was tried, measured or rejected goes to the
   changelog/notes and the closing commit; the open row links to it.
4. Every row has a next step. If you cannot name one, the task is not understood
   or not actionable yet — split it, block it, or park it.
5. The tracker page is the table and one line pointing at this standard. Do not
   repeat the type/status/size legends on it; they are defined here once, so
   there is nothing on the page to drift.

COLUMNS, in this order
| ID | Task | Type | Area | Size | Status | Owner | Next step |
- ID: a stable prefix plus a zero-padded number (e.g. TT-001), assigned once.
- Task: one line, outcome-shaped. If it needs a paragraph, it is an epic — split.
- Type: Bug | Improvement | Investigation | Chore.
- Area: the component, subsystem or external surface; one or two words.
- Size: S | M | L, for effort and uncertainty, never value. S = a self-contained
  change plus its test. M = needs a model run, a second instance, or real
  investigation. L = engine/feature work or a cross-cutting change. `—` if unknown.
- Status: Open | Blocked | Parked.
- Owner: who must act next — `here` by default, otherwise the named party
  (`upstream`, `maintainer`, `operator`, a team, a person).
- Next step: the single next action; for a blocked row, the missing thing and who
  owns it. Link the evidence (PR, issue, discussion, commit, measurement). State
  facts, not hopes.

STATUS MEANINGS
- Open: ready to start here — scope clear, no external dependency, and the next
  step is an action this checkout can take.
- Blocked: cannot proceed until something outside this checkout moves. Owner names
  who, Next step names what. A blocked row with no owner is a defect in the table.
- Parked: deliberately not scheduled. State the condition that would revive it;
  "no time" is not a condition.

ORDER
Open first, then by Size (S, M, L), then by cost to close: a fast unit test before
an intermittent run under load before a loaded-model measurement before engine
work. Blocked and Parked sort last. Priority IS row order — there is no priority
column, so the order is the one thing to keep honest.

MAINTENANCE
- Update a row the moment its state changes, not on a schedule.
- Blocked -> Open when the dependency clears; Open -> Blocked the moment it turns
  external; either -> deleted when done or abandoned, with the reason in the notes
  and the closing commit.
- Before starting work, read the table top to bottom; the top Open row is the
  default next task.
- An empty table is a healthy state. A row with no next step is not.
```

## Why these rules

- **One table, state as a column.** Sections multiply the same task across
  headings and drift apart; a single table has one source of truth and one place
  to sort.
- **Stable IDs, deleted rows.** This is a log-structured backlog: the ID is a
  permanent handle, the row is only its current state. Renumbering or recycling
  breaks every historical reference; a gap is free.
- **Owner plus Blocked.** "Waiting on someone" is the most common real state and
  the one most often lost. Naming the party in a column makes it a fact rather
  than a paragraph, and removes the need for a separate "Waiting" status.
- **Size is not priority.** Effort and value are different axes; conflating them
  produces a table sorted by neither. Row order carries priority.
- **Atomic, outcome-shaped rows.** A row that cannot be closed in one step cannot
  be estimated, owned or verified, so it cannot be managed.
- **History elsewhere.** A tracker that accumulates narrative stops being
  scannable. The notes and the closing commit are the audit trail; the tracker is
  the queue.
