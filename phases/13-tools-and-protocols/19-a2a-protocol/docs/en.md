# A2A — Agent-to-Agent Protocol

> MCP is agent-to-tool. A2A (Agent2Agent) is agent-to-agent — an open protocol for letting opaque agents built on different frameworks collaborate. Released by Google in April 2025, donated to the Linux Foundation in June 2025, reaching v1.0 in April 2026 with 150+ supporters including AWS, Cisco, Microsoft, Salesforce, SAP, and ServiceNow. It absorbed IBM's ACP and added the AP2 payments extension. This lesson walks the Agent Card, Task lifecycle, and the v1 protocol bindings.

**Type:** Build
**Languages:** Python, TypeScript (stdlib, Agent Card + Task harness)
**Prerequisites:** Phase 13 · 06 (MCP fundamentals), Phase 13 · 08 (MCP client)
**Time:** ~75 minutes

## Learning Objectives

- Distinguish agent-to-tool (MCP) from agent-to-agent (A2A) use cases.
- Publish an Agent Card at `/.well-known/agent-card.json` with skills and versioned interfaces.
- Walk the v1 `TASK_STATE_*` lifecycle through working, interrupted, and terminal states.
- Use Messages with `text` / `raw` / `url` / `data` Parts and Artifacts as outputs.

## The Problem

A customer-service agent needs to delegate report-writing to a specialized writer agent. Options pre-A2A:

- Custom REST API. Works but every pairing is a one-off.
- Shared codebase. Requires the two agents to run the same framework.
- MCP. Doesn't fit: MCP is for calling tools, not for two agents collaborating while preserving each agent's opaque internal reasoning.

A2A fills the gap. It models the interaction as one agent sending a Task to another, with a lifecycle, messages, and artifacts. The called agent's internal state stays opaque — the caller sees only task state transitions and eventual outputs.

A2A is the "let agents across frameworks talk to each other" protocol. It does not replace MCP; the two are complementary.

## The Concept

### Agent Card

Every A2A-compliant agent publishes a card at `/.well-known/agent-card.json`:

```json
{
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "supportedInterfaces": [
    {
      "url": "https://research.example.com/a2a",
      "protocolBinding": "JSONRPC",
      "protocolVersion": "1.0"
    }
  ],
  "version": "1.2.0",
  "defaultInputModes": ["text/plain", "application/pdf", "application/json"],
  "defaultOutputModes": ["text/markdown"],
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "tags": ["research", "summarization"],
      "inputModes": ["text/plain", "application/pdf"],
      "outputModes": ["text/markdown"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

Discovery is URL-based: fetch the card, choose the first supported entry in ordered `supportedInterfaces`, and enumerate skills. The agent's own `version` and each interface's A2A `protocolVersion` are separate fields.

### Signed Agent Cards

A2A v1 Agent Cards can carry a `signatures` array of JSON Web Signatures. Consumers can verify a card's origin and integrity before trusting its endpoints or skills. AP2 is a separate payments protocol, not the feature that defines Agent Card signatures.

### Task lifecycle

```text
TASK_STATE_SUBMITTED -> TASK_STATE_WORKING -> TASK_STATE_COMPLETED | TASK_STATE_FAILED
                         |                    TASK_STATE_CANCELED | TASK_STATE_REJECTED
                         -> TASK_STATE_INPUT_REQUIRED -> TASK_STATE_WORKING
```

The abstract operation is `SendMessage`. The v1 JSON-RPC binding uses method `SendMessage`; the HTTP+JSON binding uses `POST /message:send`. Legacy `tasks/send` is not a v1 method. The called agent transitions through states; clients subscribe to updates via streaming or poll with the selected binding.

### Messages and Parts

A Message has a required `messageId`, a `ROLE_USER` or `ROLE_AGENT` role, and one or more Parts. A v1 Part has no `kind` discriminator: it contains exactly one of `text`, `raw`, `url`, or `data`, plus optional `filename`, `mediaType`, and metadata.

- `text` — plain content.
- `raw` — base64-encoded bytes in JSON.
- `url` — a URL pointing to file content.
- `data` — any structured JSON value.

Example:

```json
{
  "messageId": "msg-123",
  "role": "ROLE_USER",
  "parts": [
    {"text": "Summarize this paper."},
    {"raw": "...base64...", "filename": "paper.pdf", "mediaType": "application/pdf"},
    {"data": {"targetLength": "3 paragraphs"}, "mediaType": "application/json"}
  ]
}
```

### Artifacts

Outputs are Artifacts, not raw strings. An Artifact is a named, typed output:

```json
{
  "artifactId": "artifact-123",
  "name": "summary",
  "parts": [{"text": "...", "mediaType": "text/markdown"}]
}
```

`Task.artifacts` is always plural; every Artifact requires an `artifactId` unique within that Task. The Task's current state lives under `status.state`, while exchanged Messages live under `history`. Artifacts can be streamed as chunks keyed by the same `artifactId`.

### Protocol bindings

1. **JSON-RPC over HTTP.** PascalCase methods such as `SendMessage`, with SSE for streaming.
2. **gRPC.** RPC operations carrying the canonical protobuf data model.
3. **HTTP+JSON.** Resource-oriented routes such as `POST /message:send` and `GET /tasks/{id}`.

Both bindings carry the same logical message shape.

### Opacity preservation

A key design principle: the called agent's internal state is opaque. The caller sees task state and artifacts. The called agent's chain-of-thought, its tool calls, its sub-agent delegation — all invisible. This is different from MCP, where tool calls are transparent.

Rationale: A2A enables competitors to collaborate without revealing internals. A2A can be "call this customer-service agent" without the caller learning how that agent implements the service.

### Timeline

- **2025-04-09.** Google announces A2A.
- **2025-06-23.** Donated to Linux Foundation.
- **2025-08.** Absorbs IBM's ACP.
- **2025-09.** AP2 extension (Agent Payments) ships.
- **2026-04.** v1.0 released with 150+ supporting organizations.

### Relationship to MCP

| Dimension | MCP | A2A |
|-----------|-----|-----|
| Use case | Agent-to-tool | Agent-to-agent |
| Opacity | Transparent tool calls | Opaque inner reasoning |
| Typical caller | Agent runtime | Another agent |
| State | Tool-call result | Task with lifecycle |
| Authorization | OAuth 2.1 (Phase 13 · 16) | Agent Card security schemes and per-request credentials |
| Transport | Stdio / Streamable HTTP | JSON-RPC / gRPC / HTTP+JSON bindings |

Use MCP when you want to invoke a specific tool. Use A2A when you want to delegate a whole task to another agent. Many production systems use both: an agent uses MCP for its tool layer and A2A for its collaboration layer.

```figure
a2a-task-lifecycle
```

## Use It

`code/main.py` and `code/main.ts` implement a minimal A2A v1 harness: a research agent publishes its card, a writer agent handles `SendMessage` with text and PDF Parts, transitions through `TASK_STATE_WORKING` → `TASK_STATE_INPUT_REQUIRED` → `TASK_STATE_WORKING` → `TASK_STATE_COMPLETED`, and returns an Artifact in `Task.artifacts`. Both use an in-memory transport to focus on message shapes.

What to look at:

- Agent Card JSON shape.
- Task id assignment and state transitions.
- Messages with mixed-type parts.
- Input-required branch mid-task.
- Artifact return on completion.

## Ship It

This lesson produces `outputs/skill-a2a-agent-spec.md`. Given a new agent that should be callable by other agents, the skill produces the Agent Card JSON, skills schema, and endpoint blueprint.

## Exercises

1. Run `code/main.py`. Trace the full Task lifecycle, including the input-required pause where the called agent asks for a clarification.

2. Add a signed Agent Card. Sign with HMAC over the card's canonical JSON. Write a verifier and confirm it fails on a mutated card.

3. Implement task streaming: the writer agent emits three incremental artifact chunks over SSE and the caller accumulates them.

4. Design an A2A agent that wraps an MCP server. Map each MCP tool to an A2A skill. Note the trade-offs — what opacity is lost?

5. Read the A2A v1.0 announcement and identify the one feature that is not yet implemented by any framework as of April 2026. (Hint: it relates to multi-hop task delegation.)

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| A2A | "Agent-to-Agent protocol" | Open protocol for opaque agent collaboration |
| Agent Card | "`.well-known/agent-card.json`" | Published metadata describing skills and ordered `supportedInterfaces` |
| Skill | "A callable unit" | A named operation the agent supports (analog to MCP tool) |
| Task | "Unit of delegation" | `{id, contextId?, status, history?, artifacts?}` work item |
| Message | "Task input" | Identified communication carrying `ROLE_USER`/`ROLE_AGENT` and Parts |
| Part | "Typed chunk" | Exactly one of `text` / `raw` / `url` / `data`; no `kind` discriminator |
| Artifact | "Task output" | Output with required `artifactId` stored in plural `Task.artifacts` |
| Agent Card signature | "Signed discovery" | JWS entry in `AgentCard.signatures` for origin and integrity checks |
| Opacity | "Black-box collaboration" | Called agent's internals are hidden from caller |
| Input-required | "Task pause" | Lifecycle state when the agent needs more info |

## Further Reading

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — canonical A2A specification
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — reference implementations and SDKs
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — June 2025 governance transfer
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — roadmap and partner momentum
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — v1.0 release notes and backward-compat guidance
