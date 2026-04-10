# CLAUDE.md — mlx-kv-client

## What this repo is

A thin, typed Python HTTP client for `mlx-kv-server`. Published to PyPI.
Consumed by `kai-daemon` to drive all local inference.

## Role in the system

`mlx-kv-server` is the inference server running on bare-metal Apple Silicon.
It exposes five primitives over HTTP. This library wraps those primitives in a
typed Python interface so `kai-daemon` never speaks raw HTTP to the server.

## What depends on this repo

- `kai-daemon` — the only consumer

## What this repo depends on

Nothing in this project. Standalone library.

## Build, test, lint

```bash
uv sync
uv run pytest
uv run tox
uv run pyright
uv run ruff check .
```

## Cross-repo development

When iterating on this client alongside `kai-daemon`, install it as an editable
dependency from inside the `kai-daemon` devcontainer:

```bash
uv add --editable /workspaces/mlx-kv-client
```

This works because the DLS devcontainer mounts the parent directory of the repo,
making siblings visible at `/workspaces/`. Once this client has a stable release,
switch `kai-daemon` to the published version.

## Architecture references

- `../kai-project/docs/kai-architecture.md` §1 — the primitives, /status, and their role
- `../kai-project/docs/kai-architecture.md` §6e — preemption model (checkpoint/rollback)
- `../kai-project/docs/adr-001-mlx-kv-server-status-endpoint.md` — /status decision rationale

## The five primitives + status

`mlx-kv-server` exposes five inference primitives and one read-only status endpoint
(added in Stage 3.5 — see `adr-001-mlx-kv-server-status-endpoint.md`). Wrap all six.

| Method       | Purpose                                              |
|--------------|------------------------------------------------------|
| `prefill`    | Feed a prompt into the KV cache                      |
| `generate`   | Generate tokens from current KV cache state          |
| `checkpoint` | Save current KV cache state (enables suspend)        |
| `rollback`   | Restore KV cache to last checkpoint (enables resume) |
| `evict`      | Clear the KV cache                                   |
| `status`     | Read KV cache state — read-only, never modifies state|

`checkpoint` and `rollback` together implement the `suspend` preemption mode
in `kai-daemon`'s workflow engine. This is their primary purpose.

`status()` returns a typed dataclass with fields: `cache_used_tokens`,
`cache_capacity_tokens`, `cache_used_fraction`, `checkpoint_present`,
`checkpoint_tokens`, `last_operation`, `last_operation_at`, `model`,
`uptime_seconds`.

## Critical constraints

**Typed interface throughout.** All inputs and outputs are typed Python
dataclasses. No raw dicts passed to or returned from public API. Pyright strict
mode must pass.

**Both sync and async.** `kai-daemon` workflows are async; some calling contexts
may be sync. Provide both variants.

**Connection errors must be explicit.** Never silently swallow a connection
failure. Raise a typed exception with enough information to diagnose the
problem. `kai-daemon` decides how to handle failures — this library just reports
them clearly.

**No business logic.** This library is a transport layer. It does not know what
a workflow is, what a fascination is, or what the daemon does. It speaks HTTP
to `mlx-kv-server` and hands back the result.

## Acceptance criteria

- [ ] All five primitives wrapped with typed inputs and outputs
- [x] `status()` wrapped with typed output matching `/status` response schema
- [ ] Sync and async variants for all six methods
- [x] Sync and async variants for `status()`
- [x] Connection errors raise typed exceptions, never silently fail
- [ ] Pyright strict mode passes
- [ ] Sphinx docs cover all public API

## GitHub issue hygiene

```bash
gh issue close <number> --repo distilledecho/mlx-kv-client
bash ../kai-project/setup/project-move.sh <issue-url> "Done"
```

## Review

Run in a **fresh Claude Code session** when implementation is complete:

```
/review stage=0
```
