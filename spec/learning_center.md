# Learning Center

AceAI should treat self-learning as a reviewable product workflow, not as hidden
automatic mutation. Learning output should enter a durable queue, remain
inspectable, and wait for user approval before it changes configuration,
prompts, skills, code, or project memory.

## Source Groups

Learning sources should be organized around what kind of signal taught AceAI
something.

### Learn From Environment

Environment learning comes from changes outside the current interaction. In many
workflows this means external systems or project surroundings changed, and AceAI
may need to update assumptions.

Examples:

- repository structure, dependency, configuration, or CI changes;
- API, SDK, documentation, model, or external service behavior changes;
- deployment environment, permission, or operational policy changes;
- project conventions, release rules, or team workflow changes.

The core question is: what changed in the world, and should AceAI update an
assumption because of it?

### Learn From Interaction

Interaction learning comes from the user's behavior and explicit feedback during
conversation.

Examples:

- repeated corrections about scope, style, or architecture boundaries;
- approval and rejection patterns for tool calls;
- recurring user constraints that are provided after AceAI starts work;
- user rollbacks or comments that reveal unwanted behavior;
- steering messages that consistently redirect the same kind of mistake.

The core question is: what is the user teaching AceAI through interaction?

### Learn From Execution

Execution learning comes from AceAI's own attempts to perform work.

Examples:

- tool-call success rate dropping below a threshold;
- failed trajectories that reveal recurring tool, prompt, or planning issues;
- subagent failures or repeated delegation problems;
- test, lint, deploy, or verification failures;
- cost, latency, context-pressure, or retry anomalies;
- repeated inefficient tool or skill usage patterns.

The core question is: what did real execution expose about the system's behavior?

## Learning Queue

Learning should be represented as durable review items rather than transient
notifications. A learning item should be small enough to scan, but connected to
evidence that can be inspected later.

Possible fields:

- `item_id`
- `source_group`: `environment`, `interaction`, or `execution`
- `source_kind`: a narrower classifier such as `tool_health`,
  `user_correction`, `repo_change`, or `failed_trajectory`
- `scope`: project, session, tool, skill, provider, subagent, or repo
- `status`: `pending`, `approved`, `rejected`, `archived`, or `applied`
- `severity`
- `title`
- `summary`
- `evidence_refs`
- `recommendation`
- `created_at`
- `reviewed_at`

Approval should mean the user accepts the learning item as a useful insight. A
separate follow-up action can decide whether to update config, create a spec,
generate a patch, save project memory, or add an eval.

## UI Direction

The TUI should eventually expose a Learning Center, for example through
`/learning`, where pending items can be reviewed in a queue.

The first version should stay conservative:

- show pending, approved, rejected, archived, and applied items;
- support detail view with evidence references;
- support approve, reject, and archive actions;
- avoid automatic code or config changes.

Stats should continue to answer what is happening now. Learning Center should
answer what AceAI has learned from environment, interaction, and execution.

## Open Questions

- Which learning source group should be implemented first?
- Should learning items live under `aceai/agent/memory/learning.py`?
- Should the first storage backend be SQLite, matching ideas?
- How should approved learning items become project memory, config changes,
  specs, evals, or patches?
- What cooldown and deduplication rules prevent noisy repeated suggestions?
