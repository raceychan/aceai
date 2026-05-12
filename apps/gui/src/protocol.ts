export type ReplyPayload<T = unknown> =
  | { status: "ok"; response: T }
  | { status: "error"; error: { code: string; message: string; detail: unknown } };

export type SocketEnvelope<T = unknown> = {
  topic: string;
  event: string;
  payload: T;
  ref: string | null;
  join_ref: string | null;
  event_id: string | null;
  seq: number | null;
};

export type SessionMetadata = {
  session_id: string;
  project_id: string;
  project_name: string;
  title: string;
  created_at: string;
  updated_at: string;
  path?: string;
};

export type ThreadMetadata = {
  session_id: string;
  thread_id: string;
  role: string;
  title: string;
  status: string;
  agent_id: string;
  parent_thread_id: string | null;
  parent_run_id: string | null;
  parent_tool_call_id: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

export type SessionListItem = SessionMetadata & {
  event_count: number;
  total_cost_usd: number;
  thread_count: number;
  active_thread: ThreadMetadata;
};

export type SessionsPayload = {
  sessions: SessionListItem[];
};

export type EmptySessionCleanupPayload = {
  deleted: number;
  session_ids: string[];
};

export type ReferenceItem = {
  kind: "file" | "idea";
  value: string;
  label: string;
  description: string;
  idea_id?: string;
};

export type ReferencesPayload = {
  items: ReferenceItem[];
};

export type FilePayload = {
  path: string;
  content: string;
  size: number;
  updated_at: string;
};

export type FileSavePayload = {
  file: FilePayload;
};

export type IdeaItem = {
  index: number;
  idea_id: string;
  created_at: string;
  project_id: string;
  project_name: string;
  workspace: string;
  content: string;
  source_session_id: string | null;
};

export type IdeasPayload = {
  ideas: IdeaItem[];
};

export type IdeaCapturePayload = {
  idea: IdeaItem;
};

export type IdeaMutationPayload = {
  idea: IdeaItem;
};

export type SessionEvent = {
  event_id: string;
  kind: string;
  payload: Record<string, unknown>;
  created_at: string;
  run_id: string;
  thread_id: string;
};

export type ToolApprovalRequest = {
  call: {
    call_id: string;
    name: string;
    arguments: string;
  };
  tool_name: string;
  reason: string;
  policy: string;
};

export type RuntimePayload = {
  queued_questions: string[];
  pending_approval: ToolApprovalRequest | null;
  is_running_suspended: boolean;
  active_run_id: string | null;
  active_run_status: string | null;
  provider_name: string;
  selected_model: string;
  reasoning_level: string;
};

export type UsageSummary = {
  context_tokens: number | null;
  cache_hit_rate: number | null;
  input_tokens: number;
  cached_input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  total_cost_usd: number;
};

export type EventCount = {
  kind: string;
  count: number;
};

export type ToolCallStat = {
  name: string;
  calls: number;
  succeeded: number;
  failed: number;
  approval_requests: number;
};

export type TrajectoryItem = {
  event_id: string;
  kind: string;
  thread_id: string;
  run_id: string;
  created_at: string;
  summary: string;
};

export type DebugEvent = {
  event_id: string;
  kind: string;
  thread_id: string;
  run_id: string;
  created_at: string;
  payload: Record<string, unknown>;
};

export type ObservabilityPayload = {
  usage: UsageSummary;
  event_counts: EventCount[];
  tool_calls: ToolCallStat[];
  trajectory: TrajectoryItem[];
  debug_events: DebugEvent[];
};

export type SnapshotPayload = {
  session: SessionMetadata;
  runtime: RuntimePayload;
  observability: ObservabilityPayload;
  active_thread_id: string;
  threads: ThreadMetadata[];
  events: SessionEvent[];
};

export type AppEventPayload =
  | {
      kind: "session";
      thread_id: string;
      agent_id: string;
      event: SessionEvent;
    }
  | {
      kind: "agent";
      thread_id: string;
      agent_id: string;
      event: {
        event_type: string;
        run_id: string;
        step_id: string;
        step_index: number;
        payload: Record<string, unknown>;
      };
    };

const encoder = new TextEncoder();

export function encodeEnvelope(topic: string, event: string, payload: unknown, ref: string) {
  return encoder.encode(JSON.stringify({ topic, event, payload, ref }));
}

export function isOkReply<T>(payload: ReplyPayload<T>): payload is { status: "ok"; response: T } {
  return payload.status === "ok";
}
