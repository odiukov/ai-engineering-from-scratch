// Phase 13 Lesson 19 — A2A v1.0 wire-shape harness in TypeScript.
//
// Builds current AgentCard, Message, Part, TaskStatus, Task, and Artifact
// shapes by hand, then simulates JSON-RPC SendMessage in memory.
//
// Spec: https://a2a-protocol.org/latest/specification/
// Run: npx tsx code/main.ts

import { randomUUID } from "node:crypto";

type AgentCard = {
  name: string;
  description: string;
  supportedInterfaces: Array<{
    url: string;
    protocolBinding: "JSONRPC" | "GRPC" | "HTTP+JSON";
    protocolVersion: string;
  }>;
  version: string;
  capabilities: { streaming: boolean; pushNotifications: boolean };
  defaultInputModes: string[];
  defaultOutputModes: string[];
  skills: Array<{
    id: string;
    name: string;
    description: string;
    tags: string[];
    inputModes: string[];
    outputModes: string[];
  }>;
};

type TextPart = { text: string; mediaType?: string };
type RawPart = { raw: string; filename?: string; mediaType?: string };
type DataPart = { data: unknown; mediaType?: string };
type UrlPart = { url: string; filename?: string; mediaType?: string };
type Part = TextPart | RawPart | DataPart | UrlPart;

type Message = {
  messageId: string;
  role: "ROLE_USER" | "ROLE_AGENT";
  parts: Part[];
  taskId?: string;
  contextId?: string;
};

type TaskState =
  | "TASK_STATE_WORKING"
  | "TASK_STATE_INPUT_REQUIRED"
  | "TASK_STATE_COMPLETED";

type TaskStatus = { state: TaskState; message?: Message };
type Artifact = { artifactId: string; name?: string; parts: Part[] };
type Task = {
  id: string;
  contextId: string;
  status: TaskStatus;
  history: Message[];
  artifacts: Artifact[];
};

const AGENT_CARD_PATH = "/.well-known/agent-card.json";
const WRITER_AGENT_CARD: AgentCard = {
  name: "writer-agent",
  description: "Drafts technical summaries and reports from source material.",
  supportedInterfaces: [
    {
      url: "https://writer.example.com/a2a",
      protocolBinding: "JSONRPC",
      protocolVersion: "1.0",
    },
  ],
  version: "1.0.0",
  capabilities: { streaming: true, pushNotifications: false },
  defaultInputModes: ["text/plain", "application/pdf", "application/json"],
  defaultOutputModes: ["text/markdown"],
  skills: [
    {
      id: "draft_report",
      name: "Draft report",
      description: "Produce a report from source material.",
      tags: ["writing", "summarization"],
      inputModes: ["text/plain", "application/pdf", "application/json"],
      outputModes: ["text/markdown"],
    },
  ],
};

const TASK_STORE = new Map<string, Task>();

function newId(prefix: string): string {
  return `${prefix}_${randomUUID().replaceAll("-", "").slice(0, 10)}`;
}

function textPart(text: string, mediaType = "text/plain"): TextPart {
  return { text, mediaType };
}

function message(
  role: Message["role"],
  parts: Part[],
  links: Pick<Message, "taskId" | "contextId"> = {},
): Message {
  return { messageId: newId("msg"), role, parts, ...links };
}

function finish(task: Task, length: string): void {
  task.artifacts = [
    {
      artifactId: newId("artifact"),
      name: "summary",
      parts: [
        textPart(
          `[writer agent] ${length} summary: topic identified, key points extracted, conclusion drafted.`,
          "text/markdown",
        ),
      ],
    },
  ];
  task.status = { state: "TASK_STATE_COMPLETED" };
  console.log(`    WRITER  : completed task ${task.id}`);
}

function sendMessage(request: { message: Message }): Task {
  const incoming = request.message;
  let task: Task;
  if (incoming.taskId === undefined) {
    const id = newId("task");
    task = {
      id,
      contextId: incoming.contextId ?? newId("context"),
      status: { state: "TASK_STATE_WORKING" },
      history: [incoming],
      artifacts: [],
    };
    TASK_STORE.set(id, task);
  } else {
    const stored = TASK_STORE.get(incoming.taskId);
    if (stored === undefined) throw new Error(`unknown task ${incoming.taskId}`);
    task = stored;
    task.history.push(incoming);
    task.status = { state: "TASK_STATE_WORKING" };
  }

  const data = incoming.parts.find((part): part is DataPart => "data" in part)?.data;
  if (typeof data !== "object" || data === null || !("targetLength" in data)) {
    task.status = {
      state: "TASK_STATE_INPUT_REQUIRED",
      message: message(
        "ROLE_AGENT",
        [textPart("Please provide targetLength as a data Part.")],
        { taskId: task.id, contextId: task.contextId },
      ),
    };
  } else {
    finish(task, String((data as { targetLength: unknown }).targetLength));
  }
  return task;
}

console.log("=".repeat(72));
console.log("PHASE 13 LESSON 19 - A2A V1 SENDMESSAGE (TYPESCRIPT)");
console.log("=".repeat(72));
console.log(`\nGET ${AGENT_CARD_PATH}`);
console.log(
  JSON.stringify(
    {
      name: WRITER_AGENT_CARD.name,
      supportedInterfaces: WRITER_AGENT_CARD.supportedInterfaces,
      skills: WRITER_AGENT_CARD.skills,
    },
    null,
    2,
  ),
);

const initial = message("ROLE_USER", [
  textPart("Summarize the attached paper."),
  {
    raw: Buffer.from("fake-pdf").toString("base64"),
    filename: "paper.pdf",
    mediaType: "application/pdf",
  },
]);
let task = sendMessage({ message: initial });
console.log(`\nSendMessage -> ${task.status.state}`);

if (task.status.state === "TASK_STATE_INPUT_REQUIRED") {
  const followup = message(
    "ROLE_USER",
    [{ data: { targetLength: "3 paragraphs" }, mediaType: "application/json" }],
    { taskId: task.id, contextId: task.contextId },
  );
  task = sendMessage({ message: followup });
}

console.log("\nTask response:");
console.log(JSON.stringify(task, null, 2));
console.log(`\nartifactId: ${task.artifacts[0].artifactId}`);
