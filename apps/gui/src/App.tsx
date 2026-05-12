import {
  Activity,
  AlertTriangle,
  Bot,
  Braces,
  Check,
  ChevronDown,
  ChevronRight,
  CircleStop,
  Clock3,
  Copy,
  Database,
  Edit3,
  FileText,
  GitBranch,
  Layers,
  MessageSquare,
  Mic,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRight,
  Plus,
  RefreshCw,
  Save,
  Search,
  Send,
  Sparkles,
  SquareTerminal,
  TerminalSquare,
  Trash2,
  WifiOff
} from "lucide-react";
import Editor, { Monaco } from "@monaco-editor/react";
import { CSSProperties, FormEvent, KeyboardEvent, MouseEvent, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  AppEventPayload,
  EmptySessionCleanupPayload,
  FilePayload,
  FileSavePayload,
  IdeaCapturePayload,
  IdeaItem,
  IdeaMutationPayload,
  IdeasPayload,
  ReplyPayload,
  ReferenceItem,
  ReferencesPayload,
  SessionListItem,
  SessionEvent,
  SessionsPayload,
  SnapshotPayload,
  SocketEnvelope,
  ToolApprovalRequest,
  DebugEvent,
  ObservabilityPayload,
  RuntimePayload,
  encodeEnvelope,
  isOkReply
} from "./protocol";

type ConnectionState = "idle" | "connecting" | "connected" | "closed" | "error";

type TranscriptItem =
  | { id: string; role: "user"; text: string; time: string }
  | { id: string; role: "assistant"; text: string; time: string }
  | { id: string; role: "system"; text: string; time: string };

type PendingRequest = {
  event: string;
  resolve: (payload: ReplyPayload<unknown>) => void;
  reject: (error: Error) => void;
};

type TimelineItem = {
  id: string;
  title: string;
  detail: string;
  tone: "neutral" | "good" | "bad" | "live";
};

type ComposerCommand = {
  name: string;
  label: string;
  hint: string;
};

const MARKDOWN_PLUGINS = [remarkGfm];
const INSPECTOR_WIDTH_KEY = "aceai.gui.inspectorWidth";
const DEFAULT_INSPECTOR_WIDTH = 420;
const MIN_INSPECTOR_WIDTH = 320;
const MIN_CONVERSATION_WIDTH = 520;

type QueuedTurn = {
  id: string;
  content: string;
};

type ArtifactItem = {
  id: string;
  kind: "file" | "patch" | "tool" | "subagent";
  title: string;
  subtitle: string;
  status: string;
  createdAt: string;
  content: string;
};

type OpenFileItem = FilePayload;

type WorkspaceMode = "chat" | "sessions" | "ideas" | "threads" | "events" | "artifacts";
type MonacoThemeChoice = "aceai" | "light" | "dark";

const COMPOSER_COMMANDS: ComposerCommand[] = [
  { name: "/clear", label: "Clear transcript view", hint: "Hide visible messages for this view" },
  { name: "/sessions", label: "Find a session", hint: "Open saved sessions" },
  { name: "/stats", label: "Review stats", hint: "Jump to session metrics" },
  { name: "/debug", label: "Inspect events", hint: "Open events" },
  { name: "/trajectory", label: "Review timeline", hint: "Jump to the run timeline" },
  { name: "/subagents", label: "Review threads", hint: "Open threads and subagents" },
  { name: "/config", label: "Open settings", hint: "Show connection settings" },
  { name: "/idea", label: "Save or reference idea", hint: "Use /idea <text> or open Ideas" },
  { name: "/steer", label: "Steer active run", hint: "Cancel the current run and send a new turn" }
];
const MONACO_THEME_OPTIONS: { value: MonacoThemeChoice; label: string; theme: string }[] = [
  { value: "aceai", label: "AceAI", theme: "aceai-light" },
  { value: "light", label: "Light", theme: "vs" },
  { value: "dark", label: "Dark", theme: "vs-dark" }
];

export function App() {
  const [serverUrl] = useState(() => localStorage.getItem("aceai.gui.ws") ?? defaultWebSocketUrl());
  const [joinRef, setJoinRef] = useState<string | null>(null);
  const [connectionState, setConnectionState] = useState<ConnectionState>("idle");
  const [input, setInput] = useState("");
  const [sessions, setSessions] = useState<SessionListItem[]>([]);
  const [ideas, setIdeas] = useState<IdeaItem[]>([]);
  const [selectedIdeaIndex, setSelectedIdeaIndex] = useState(1);
  const [ideaDraft, setIdeaDraft] = useState("");
  const [newIdeaDraft, setNewIdeaDraft] = useState("");
  const [sessionQuery, setSessionQuery] = useState("");
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [snapshot, setSnapshot] = useState<SnapshotPayload | null>(null);
  const [events, setEvents] = useState<SessionEvent[]>([]);
  const [activity, setActivity] = useState<SocketEnvelope[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [queuedTurns, setQueuedTurns] = useState<QueuedTurn[]>([]);
  const [optimisticTurns, setOptimisticTurns] = useState<TranscriptItem[]>([]);
  const [liveAssistantText, setLiveAssistantText] = useState("");
  const [liveTimeline, setLiveTimeline] = useState<TimelineItem[]>([]);
  const [pendingApproval, setPendingApproval] = useState<ToolApprovalRequest | null>(null);
  const [referenceItems, setReferenceItems] = useState<ReferenceItem[]>([]);
  const [selectedReferenceIndex, setSelectedReferenceIndex] = useState(0);
  const [selectedDebugEventId, setSelectedDebugEventId] = useState<string | null>(null);
  const [selectedArtifactId, setSelectedArtifactId] = useState<string | null>(null);
  const [openFile, setOpenFile] = useState<OpenFileItem | null>(null);
  const [fileDraft, setFileDraft] = useState("");
  const [fileLoading, setFileLoading] = useState(false);
  const [fileEditMode, setFileEditMode] = useState(false);
  const [monacoTheme, setMonacoTheme] = useState<MonacoThemeChoice>("aceai");
  const [inspectorWidth, setInspectorWidth] = useState(() => storedInspectorWidth());
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [showTranscriptWorkHistory, setShowTranscriptWorkHistory] = useState(false);
  const [selectedCommandIndex, setSelectedCommandIndex] = useState(0);
  const [workspaceMode, setWorkspaceMode] = useState<WorkspaceMode>("chat");
  const apiBaseUrl = useMemo(() => apiBaseFromWebSocketUrl(serverUrl), [serverUrl]);
  const socketRef = useRef<WebSocket | null>(null);
  const activeTopicRef = useRef("session:new");
  const refCounter = useRef(0);
  const pending = useRef(new Map<string, PendingRequest>());
  const transcriptEndRef = useRef<HTMLDivElement | null>(null);
  const composerRef = useRef<HTMLTextAreaElement | null>(null);
  const sessionSearchRef = useRef<HTMLInputElement | null>(null);
  const timelineRef = useRef<HTMLElement | null>(null);
  const statsRef = useRef<HTMLElement | null>(null);
  const eventsRef = useRef<HTMLElement | null>(null);
  const threadsRef = useRef<HTMLElement | null>(null);

  const transcript = useMemo(() => buildTranscript(events), [events]);
  const visibleTranscript = useMemo(
    () => mergeOptimisticTranscript(transcript, optimisticTurns, liveAssistantText),
    [transcript, optimisticTurns, liveAssistantText]
  );
  const artifacts = useMemo(() => buildArtifacts(events), [events]);
  const latestRun = useMemo(() => findLatestRun(events), [events]);
  const eventKinds = useMemo(() => summarizeEventKinds(events), [events]);
  const runtime = runtimeOf(snapshot);
  const observability = observabilityOf(snapshot);
  const observableEventKinds = observability?.event_counts ?? eventKinds;
  const observableTrajectory = observability?.trajectory ?? [];
  const observableToolCalls = observability?.tool_calls ?? [];
  const observableUsage = observability?.usage;
  const observableDebugEvents = observability?.debug_events ?? [];
  const selectedDebugEvent = observableDebugEvents.find((event) => event.event_id === selectedDebugEventId) ?? observableDebugEvents[0];
  const selectedArtifact = artifacts.find((artifact) => artifact.id === selectedArtifactId) ?? artifacts[0];
  const inspectedArtifact = selectedArtifactId === null ? null : selectedArtifact;
  const selectedIdea = ideas.find((idea) => idea.index === selectedIdeaIndex) ?? ideas[0];
  const timeline = useMemo(
    () => buildTimeline(events, activity, liveTimeline, observableTrajectory),
    [events, activity, liveTimeline, observableTrajectory]
  );
  const workHistoryLabel = useMemo(() => workHistoryDuration(events, latestRun, isRunning), [events, latestRun, isRunning]);
  const visibleSessions = useMemo(() => filterSessions(sessions, sessionQuery), [sessions, sessionQuery]);
  const emptySessionCount = useMemo(() => sessions.filter((session) => session.event_count === 0).length, [sessions]);
  const commandMatches = useMemo(() => matchingCommands(input), [input]);
  const activeReference = useMemo(() => activeReferencePrefix(input), [input]);
  const connected = connectionState === "connected";
  const connectionLabel = connectionState === "idle" ? "ready" : connectionState;
  const activeThread = snapshot?.threads.find((thread) => thread.thread_id === snapshot.active_thread_id);
  const showCommandMenu = connected && commandMatches.length > 0 && input.startsWith("/");
  const showReferenceMenu = connected && activeReference !== null && referenceItems.length > 0;
  const isBlockedForApproval = pendingApproval !== null || runtime.is_running_suspended;
  const hasWorkspaceObject = fileLoading || openFile !== null || inspectedArtifact !== null;
  const hasTranscriptWorkHistory = timeline.length > 0 || observableToolCalls.length > 0 || observableEventKinds.length > 0;
  const composerStatus = isBlockedForApproval ? "approval" : isRunning ? "running" : connected ? "ready" : "offline";

  useEffect(() => {
    if (!connected || isRunning || isBlockedForApproval || queuedTurns.length === 0) return;
    const [nextTurn, ...remainingTurns] = queuedTurns;
    setQueuedTurns(remainingTurns);
    void startMessage(nextTurn.content);
  }, [connected, isRunning, isBlockedForApproval, queuedTurns]);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ block: "end" });
  }, [visibleTranscript.length, activity.length]);

  useEffect(() => {
    if (!connected || !isRunning) return;
    const timer = window.setInterval(() => void refreshSnapshot(), 2500);
    return () => window.clearInterval(timer);
  }, [connected, isRunning]);

  useEffect(() => {
    void loadSessions();
    void loadIdeas();
  }, []);

  useEffect(() => {
    if (!connected || activeReference === null) {
      setReferenceItems([]);
      setSelectedReferenceIndex(0);
      return;
    }
    const controller = new AbortController();
    void loadReferences(activeReference, controller.signal);
    return () => controller.abort();
  }, [connected, activeReference]);

  useEffect(() => {
    return () => {
      socketRef.current?.close();
    };
  }, []);

  function nextRef(prefix: string) {
    refCounter.current += 1;
    return `${prefix}-${refCounter.current}`;
  }

  function sendCommand<T = unknown>(event: string, payload: unknown): Promise<T> {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error("WebSocket is not connected"));
    }
    const ref = nextRef(event);
    socket.send(encodeEnvelope(activeTopicRef.current, event, payload, ref));
    return new Promise<T>((resolve, reject) => {
      pending.current.set(ref, {
        event,
        resolve: (reply) => {
          if (isOkReply(reply)) {
            resolve(reply.response as T);
            return;
          }
          reject(new Error(reply.error.message));
        },
        reject
      });
    });
  }

  async function loadSessions() {
    setSessionsLoading(true);
    try {
      const response = await fetch(apiEndpoint(apiBaseUrl, "/api/sessions"));
      const payload = await readApiJson<SessionsPayload>(response, "Load sessions");
      setSessions(payload.sessions);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load sessions");
    } finally {
      setSessionsLoading(false);
    }
  }

  async function loadIdeas(preferredIndex = selectedIdeaIndex) {
    try {
      const response = await fetch(apiEndpoint(apiBaseUrl, "/api/ideas"));
      const payload = await readApiJson<IdeasPayload>(response, "Load ideas");
      setIdeas(payload.ideas);
      const nextIdea = payload.ideas.find((idea) => idea.index === preferredIndex) ?? payload.ideas[0];
      setSelectedIdeaIndex(nextIdea?.index ?? 1);
      setIdeaDraft(nextIdea?.content ?? "");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load ideas");
    }
  }

  async function loadReferences(query: string, signal: AbortSignal) {
    try {
      const response = await fetch(apiEndpoint(apiBaseUrl, `/api/references?q=${encodeURIComponent(query)}`), { signal });
      const payload = await readApiJson<ReferencesPayload>(response, "Load references");
      setReferenceItems(payload.items);
      setSelectedReferenceIndex(0);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setReferenceItems([]);
    }
  }

  function connectSession(sessionId: string) {
    const nextTopic = `session:${sessionId}`;
    activeTopicRef.current = nextTopic;
    socketRef.current?.close();
    pending.current.clear();
    localStorage.setItem("aceai.gui.ws", serverUrl);
    setError(null);
    setNotice(null);
    setSnapshot(null);
    setEvents([]);
    setActivity([]);
    setQueuedTurns([]);
    setOptimisticTurns([]);
    setLiveAssistantText("");
    setLiveTimeline([]);
    setPendingApproval(null);
    setSelectedDebugEventId(null);
    setSelectedArtifactId(null);
    setOpenFile(null);
    setFileDraft("");
    setFileEditMode(false);
    setJoinRef(null);
    setConnectionState("connecting");

    const socket = new WebSocket(serverUrl);
    socketRef.current = socket;

    socket.onopen = () => {
      const ref = nextRef("join");
      socket.send(encodeEnvelope(nextTopic, "join", {}, ref));
      pending.current.set(ref, {
        event: "join",
        resolve: (payload) => {
          if (!isOkReply(payload)) {
            setError(payload.error.message);
            setConnectionState("error");
            return;
          }
          setJoinRef(ref);
          setConnectionState("connected");
          void refreshSnapshot();
        },
        reject: (err) => {
          setError(err.message);
          setConnectionState("error");
        }
      });
    };

    socket.onmessage = (message) => {
      if (socketRef.current !== socket) return;
      const envelope = JSON.parse(message.data) as SocketEnvelope;
      setActivity((items) => [envelope, ...items].slice(0, 120));

      if (envelope.event === "reply" && envelope.ref) {
        const request = pending.current.get(envelope.ref);
        if (request) {
          pending.current.delete(envelope.ref);
          request.resolve(envelope.payload as ReplyPayload);
        }
        return;
      }

      if (envelope.event === "agent.event") {
        handleAppEvent(envelope.payload as AppEventPayload);
      }

      if (envelope.event === "session.event") {
        appendSessionEvent(envelope.payload as SessionEvent);
      }

      if (envelope.event === "run.cancelled") {
        setIsRunning(false);
        setPendingApproval(null);
      }
    };

    socket.onerror = () => {
      if (socketRef.current !== socket) return;
      setError("WebSocket connection failed");
      setConnectionState("error");
    };

    socket.onclose = () => {
      if (socketRef.current !== socket) return;
      setConnectionState((current) => (current === "error" ? current : "closed"));
      setIsRunning(false);
      for (const request of pending.current.values()) {
        request.reject(new Error("WebSocket closed"));
      }
      pending.current.clear();
    };
  }

  function createSession() {
    connectSession("new");
  }

  async function deleteSession(session: SessionListItem) {
    if (!window.confirm(`Delete "${session.title}"?`)) {
      return;
    }
    try {
      const response = await fetch(apiEndpoint(apiBaseUrl, `/api/sessions/${session.session_id}`), { method: "DELETE" });
      if (!response.ok) {
        setError("Session delete failed");
        return;
      }
      if (snapshot?.session.session_id === session.session_id) {
        socketRef.current?.close();
        setSnapshot(null);
        setEvents([]);
        setActivity([]);
        setQueuedTurns([]);
        setOptimisticTurns([]);
        setLiveAssistantText("");
        setLiveTimeline([]);
        setPendingApproval(null);
        setSelectedDebugEventId(null);
        setOpenFile(null);
        setFileDraft("");
        setFileEditMode(false);
        setConnectionState("idle");
        activeTopicRef.current = "session:new";
      }
      await loadSessions();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Session delete failed");
    }
  }

  async function clearEmptySessions() {
    if (emptySessionCount === 0) return;
    if (!window.confirm(`Delete ${emptySessionCount} empty sessions?`)) {
      return;
    }
    try {
      const response = await fetch(apiEndpoint(apiBaseUrl, "/api/session-cleanup/empty"), { method: "DELETE" });
      if (!response.ok) {
        setError("Empty session cleanup failed");
        return;
      }
      const payload = await readApiJson<EmptySessionCleanupPayload>(response, "Clean empty sessions");
      if (snapshot && payload.session_ids.includes(snapshot.session.session_id)) {
        socketRef.current?.close();
        setSnapshot(null);
        setEvents([]);
        setActivity([]);
        setQueuedTurns([]);
        setOptimisticTurns([]);
        setLiveAssistantText("");
        setLiveTimeline([]);
        setPendingApproval(null);
        setSelectedDebugEventId(null);
        setOpenFile(null);
        setFileDraft("");
        setFileEditMode(false);
        setConnectionState("idle");
        activeTopicRef.current = "session:new";
      }
      setNotice(`Deleted ${payload.deleted} empty sessions.`);
      await loadSessions();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Empty session cleanup failed");
    }
  }

  async function switchThread(threadId: string) {
    if (!connected || snapshot?.active_thread_id === threadId) return;
    try {
      const response = await sendCommand<SnapshotPayload>("switch_thread", { thread_id: threadId });
      setSnapshot(response);
      setEvents(response.events);
      setQueuedTurns(snapshotQueuedTurns(response));
      const responseRuntime = runtimeOf(response);
      setPendingApproval(responseRuntime.pending_approval);
      setIsRunning(snapshotIsRunning(responseRuntime));
      if (!snapshotIsRunning(responseRuntime)) {
        setLiveAssistantText("");
      }
      setSelectedDebugEventId(observabilityOf(response)?.debug_events[0]?.event_id ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Thread switch failed");
    }
  }

  function handleAppEvent(payload: AppEventPayload) {
    if (payload.kind === "session") {
      appendSessionEvent(payload.event);
      if (payload.event.kind === "run_completed" || payload.event.kind === "run_failed") {
        setIsRunning(false);
        setPendingApproval(null);
      }
      return;
    }
    const eventType = payload.event.event_type;
    if (eventType === "agent.llm.output_text.delta") {
      appendLiveAssistantDelta(payload.event.payload);
      return;
    }
    appendLiveTimelineEvent(payload);
    if (eventType === "agent.tool.approval_requested" || eventType === "agent.run.suspended") {
      setPendingApproval(eventApprovalRequest(payload.event.payload));
      setIsRunning(false);
      return;
    }
    if (eventType === "agent.tool.approval_resolved") {
      setPendingApproval(null);
      setIsRunning(true);
      return;
    }
    if (isTerminalAgentRunEvent(payload)) {
      setIsRunning(false);
      setPendingApproval(null);
      window.setTimeout(() => void refreshSnapshot(), 250);
    }
  }

  function appendSessionEvent(event: SessionEvent) {
    setEvents((current) => {
      if (current.some((item) => item.event_id === event.event_id)) {
        return current;
      }
      return [...current, event];
    });
  }

  async function refreshSnapshot() {
    try {
      const response = await sendCommand<SnapshotPayload>("snapshot", {});
      setSnapshot(response);
      setEvents(response.events);
      setQueuedTurns(snapshotQueuedTurns(response));
      const responseRuntime = runtimeOf(response);
      setPendingApproval(responseRuntime.pending_approval);
      setIsRunning(snapshotIsRunning(responseRuntime));
      if (!snapshotIsRunning(responseRuntime)) {
        setLiveAssistantText("");
        setLiveTimeline([]);
      }
      setSelectedDebugEventId(observabilityOf(response)?.debug_events[0]?.event_id ?? null);
      void loadSessions();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Snapshot failed");
    }
  }

  async function submitMessage(event: FormEvent) {
    event.preventDefault();
    if (!input) return;
    if (input.startsWith("/")) {
      executeComposerCommand(input);
      return;
    }
    const content = input;
    setInput("");
    if (isRunning || isBlockedForApproval) {
      enqueueComposerTurn(content);
      return;
    }
    await startMessage(content);
  }

  async function startMessage(content: string) {
    appendOptimisticUserMessage(content);
    setLiveAssistantText("");
    setLiveTimeline([]);
    setIsRunning(true);
    try {
      await sendCommand("send_message", { content });
    } catch (err) {
      setIsRunning(false);
      setError(err instanceof Error ? err.message : "Send failed");
    }
  }

  async function cancelRun() {
    try {
      await sendCommand("cancel", {});
      setIsRunning(false);
      setPendingApproval(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Cancel failed");
    }
  }

  function enqueueComposerTurn(content: string) {
    setQueuedTurns((turns) => [...turns, { id: nextRef("queued"), content }]);
  }

  function appendOptimisticUserMessage(content: string) {
    setOptimisticTurns((turns) => [
      ...turns,
      {
        id: nextRef("optimistic-user"),
        role: "user",
        text: content,
        time: new Date().toISOString()
      }
    ]);
  }

  function appendLiveAssistantDelta(payload: Record<string, unknown>) {
    const delta = payload.text_delta;
    if (typeof delta !== "string" || delta === "") return;
    setLiveAssistantText((current) => current + delta);
  }

  function appendLiveTimelineEvent(payload: AppEventPayload) {
    if (payload.kind !== "agent") return;
    const item = liveTimelineItem(payload);
    if (item === null) return;
    setLiveTimeline((items) => [item, ...items.filter((existing) => existing.id !== item.id)].slice(0, 8));
  }

  function cancelQueuedTurn(index: number) {
    setQueuedTurns((turns) => turns.filter((_, turnIndex) => turnIndex !== index));
  }

  async function steerQueuedTurn(index: number) {
    const queuedTurn = queuedTurns[index];
    if (!queuedTurn) return;
    setQueuedTurns((turns) => turns.filter((_, turnIndex) => turnIndex !== index));
    if (isRunning) {
      await cancelRun();
    }
    await startMessage(queuedTurn.content);
  }

  async function approvePendingTool() {
    if (!pendingApproval) return;
    setIsRunning(true);
    try {
      await sendCommand("approve_tool", { tool_call_id: pendingApproval.call.call_id });
      setPendingApproval(null);
    } catch (err) {
      setIsRunning(false);
      setError(err instanceof Error ? err.message : "Approval failed");
    }
  }

  async function rejectPendingTool() {
    if (!pendingApproval) return;
    setIsRunning(true);
    try {
      await sendCommand("reject_tool", { tool_call_id: pendingApproval.call.call_id, reason: "rejected by user" });
      setPendingApproval(null);
    } catch (err) {
      setIsRunning(false);
      setError(err instanceof Error ? err.message : "Rejection failed");
    }
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (showReferenceMenu && event.key === "ArrowDown") {
      event.preventDefault();
      setSelectedReferenceIndex((index) => (index + 1) % referenceItems.length);
      return;
    }
    if (showReferenceMenu && event.key === "ArrowUp") {
      event.preventDefault();
      setSelectedReferenceIndex((index) => (index - 1 + referenceItems.length) % referenceItems.length);
      return;
    }
    if (showReferenceMenu && (event.key === "Tab" || event.key === "Enter")) {
      event.preventDefault();
      applyReference(referenceItems[selectedReferenceIndex]);
      return;
    }
    if (showCommandMenu && event.key === "ArrowDown") {
      event.preventDefault();
      setSelectedCommandIndex((index) => (index + 1) % commandMatches.length);
      return;
    }
    if (showCommandMenu && event.key === "ArrowUp") {
      event.preventDefault();
      setSelectedCommandIndex((index) => (index - 1 + commandMatches.length) % commandMatches.length);
      return;
    }
    if (showCommandMenu && event.key === "Tab") {
      event.preventDefault();
      applyCommand(commandMatches[selectedCommandIndex]);
      return;
    }
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  function applyCommand(command: ComposerCommand) {
    setComposerValue(command.name);
    setSelectedCommandIndex(0);
  }

  function setComposerValue(value: string) {
    setInput(value);
    window.requestAnimationFrame(() => composerRef.current?.focus());
  }

  function applyReference(item: ReferenceItem) {
    setInput((value) => replaceActiveReference(value, item.value));
    setReferenceItems([]);
    setSelectedReferenceIndex(0);
    if (item.kind === "file") {
      void openProjectFile(filePathFromReference(item.value));
    }
    window.requestAnimationFrame(() => composerRef.current?.focus());
  }

  async function openProjectFile(path: string, options: { edit?: boolean } = {}) {
    setFileLoading(true);
    setError(null);
    try {
      const response = await fetch(apiEndpoint(apiBaseUrl, `/api/files?path=${encodeURIComponent(path)}`));
      const payload = await readApiJson<FileSavePayload>(response, "Open file");
      setOpenFile(payload.file);
      setFileDraft(payload.file.content);
      setFileEditMode(options.edit === true);
      setSelectedArtifactId(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Open file failed");
    } finally {
      setFileLoading(false);
    }
  }

  async function saveOpenFile() {
    if (openFile === null) return;
    try {
      const response = await fetch(apiEndpoint(apiBaseUrl, `/api/files?path=${encodeURIComponent(openFile.path)}`), {
        method: "PUT",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ content: fileDraft })
      });
      const payload = await readApiJson<FileSavePayload>(response, "Save file");
      setOpenFile(payload.file);
      setFileDraft(payload.file.content);
      setFileEditMode(false);
      setNotice(`Saved ${payload.file.path}.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save file failed");
    }
  }

  function inspectArtifact(artifact: ArtifactItem) {
    setSelectedArtifactId(artifact.id);
    setOpenFile(null);
    setFileDraft("");
    setFileEditMode(false);
    if (artifact.kind === "file") {
      void openProjectFile(artifact.title);
    }
  }

  function startInspectorResize(event: MouseEvent<HTMLDivElement>) {
    event.preventDefault();
    const startX = event.clientX;
    const startWidth = inspectorWidth;
    const shellWidth = event.currentTarget.parentElement?.getBoundingClientRect().width ?? window.innerWidth;

    function resize(moveEvent: globalThis.MouseEvent) {
      const nextWidth = startWidth - (moveEvent.clientX - startX);
      const maxWidth = Math.max(MIN_INSPECTOR_WIDTH, shellWidth - MIN_CONVERSATION_WIDTH);
      const clampedWidth = Math.min(Math.max(nextWidth, MIN_INSPECTOR_WIDTH), maxWidth);
      setInspectorWidth(clampedWidth);
      localStorage.setItem(INSPECTOR_WIDTH_KEY, String(clampedWidth));
    }

    function stopResize() {
      document.body.classList.remove("resizing-inspector");
      window.removeEventListener("mousemove", resize);
      window.removeEventListener("mouseup", stopResize);
    }

    document.body.classList.add("resizing-inspector");
    window.addEventListener("mousemove", resize);
    window.addEventListener("mouseup", stopResize);
  }

  async function saveIdea(content: string) {
    try {
      const response = await fetch(apiEndpoint(apiBaseUrl, "/api/ideas"), {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ content })
      });
      if (!response.ok) {
        setError("Idea save failed");
        return false;
      }
      const payload = await readApiJson<IdeaCapturePayload>(response, "Save idea");
      setNotice(`Saved idea ${payload.idea.index}.`);
      setSelectedIdeaIndex(payload.idea.index);
      setIdeaDraft(payload.idea.content);
      await loadIdeas(payload.idea.index);
      return true;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Idea save failed");
      return false;
    }
  }

  async function saveWorkspaceIdea() {
    if (!newIdeaDraft) return;
    const saved = await saveIdea(newIdeaDraft);
    if (saved) {
      setNewIdeaDraft("");
    }
  }

  async function updateSelectedIdea() {
    if (!selectedIdea) return;
    try {
      const response = await fetch(apiEndpoint(apiBaseUrl, `/api/ideas/${selectedIdea.index}`), {
        method: "PUT",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ content: ideaDraft })
      });
      const payload = await readApiJson<IdeaMutationPayload>(response, "Update idea");
      setNotice(`Updated idea ${payload.idea.index}.`);
      await loadIdeas();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Idea update failed");
    }
  }

  async function deleteSelectedIdea() {
    if (!selectedIdea) return;
    if (!window.confirm(`Delete idea ${selectedIdea.index}?`)) return;
    try {
      const response = await fetch(apiEndpoint(apiBaseUrl, `/api/ideas/${selectedIdea.index}`), { method: "DELETE" });
      const payload = await readApiJson<IdeaMutationPayload>(response, "Delete idea");
      setNotice(`Deleted idea ${payload.idea.index}.`);
      await loadIdeas();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Idea delete failed");
    }
  }

  function selectIdea(idea: IdeaItem) {
    setSelectedIdeaIndex(idea.index);
    setIdeaDraft(idea.content);
  }

  function referenceSelectedIdea() {
    if (!selectedIdea) return;
    setComposerValue(`@idea:${selectedIdea.index} `);
  }

  async function copyText(text: string, label: string) {
    await navigator.clipboard.writeText(text);
    setNotice(`${label} copied.`);
  }

  function executeComposerCommand(value: string) {
    const commandName = value.split(/\s+/, 1)[0];
    const commandArg = value.startsWith(`${commandName} `)
      ? value.slice(commandName.length + 1)
      : "";
    setInput("");
    setSelectedCommandIndex(0);
    if (commandName === "/clear") {
      setEvents([]);
      return;
    }
    if (commandName === "/sessions") {
      setWorkspaceMode("sessions");
      return;
    }
    if (commandName === "/stats") {
      statsRef.current?.scrollIntoView({ block: "start" });
      return;
    }
    if (commandName === "/debug") {
      setWorkspaceMode("events");
      return;
    }
    if (commandName === "/trajectory") {
      timelineRef.current?.scrollIntoView({ block: "start" });
      return;
    }
    if (commandName === "/subagents") {
      setWorkspaceMode("threads");
      return;
    }
    if (commandName === "/config") {
      setNotice("Settings are not exposed in this GUI yet.");
      return;
    }
    if (commandName === "/idea") {
      if (commandArg === "") {
        setWorkspaceMode("ideas");
        return;
      }
      void saveIdea(commandArg);
      return;
    }
    if (commandName === "/steer") {
      if (!commandArg) {
        setError("Usage: /steer <message>");
        return;
      }
      if (isRunning) {
        void cancelRun().then(() => startMessage(commandArg));
        return;
      }
      void startMessage(commandArg);
      return;
    }
    setError(`Unknown command: ${commandName}`);
  }

  return (
    <main className={`shell ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <TerminalSquare size={18} />
          </div>
          <div className="brand-copy">
            <strong>AceAI</strong>
            <span>Web GUI</span>
          </div>
          <button
            className="sidebar-toggle"
            onClick={() => setSidebarCollapsed((collapsed) => !collapsed)}
            title={sidebarCollapsed ? "Expand sessions" : "Collapse sessions"}
            aria-label={sidebarCollapsed ? "Expand sessions sidebar" : "Collapse sessions sidebar"}
          >
            {sidebarCollapsed ? <PanelLeftOpen size={15} /> : <PanelLeftClose size={15} />}
          </button>
        </div>

        <section className="side-section new-session-panel" aria-label="New session">
          <button className="primary connect-button" onClick={createSession} title="Start a new AceAI session">
            <Plus size={16} />
            New Chat
          </button>
        </section>

        <section className="side-section sessions-panel">
          <div className="section-title">
            <span>
              <Layers size={15} />
              Sessions
            </span>
            <button
              className="session-cleanup-button"
              disabled={emptySessionCount === 0}
              onClick={() => void clearEmptySessions()}
              title="Delete empty sessions"
            >
              <Trash2 size={12} />
              {emptySessionCount}
            </button>
          </div>
          <label className="session-search">
            <Search size={14} />
            <input
              ref={sessionSearchRef}
              value={sessionQuery}
              onChange={(event) => setSessionQuery(event.target.value)}
              placeholder="Search sessions"
            />
          </label>
          <div className="session-stack">
            {sessionsLoading ? <div className="subtle-empty">Loading sessions...</div> : null}
            {visibleSessions.length === 0 && !sessionsLoading ? <div className="subtle-empty">No saved sessions.</div> : null}
            {visibleSessions.map((session) => {
              const active = snapshot?.session.session_id === session.session_id;
              return (
                <div className={`session-row ${active ? "active online" : ""}`} key={session.session_id}>
                  <button className="session-open" onClick={() => connectSession(session.session_id)} title={`Open ${session.title}`}>
                    <span className="session-status" />
                    <div>
                      <strong>{session.title}</strong>
                      <span>{session.project_name} / {formatShortDate(session.updated_at)}</span>
                      <small>{session.thread_count} threads / {formatUsd(session.total_cost_usd)}</small>
                    </div>
                    <code>{session.event_count}</code>
                  </button>
                  <button
                    aria-label={`Delete ${session.title} ${session.session_id}`}
                    className="session-delete"
                    onClick={() => void deleteSession(session)}
                    title="Delete session"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              );
            })}
          </div>
        </section>

      </aside>

      <section className="workspace">
        <header className="topbar">
          <div className="workspace-title">
            <StatusDot state={connectionState} />
            <div>
              <strong>{snapshot?.session.title || "New AceAI session"}</strong>
              <span>
                {connectionLabel}
                {snapshot ? ` / ${snapshot.session.project_name}` : ""}
                {observableUsage?.total_tokens ? ` / ${formatCompactNumber(observableUsage.total_tokens)} tokens` : ""}
              </span>
            </div>
          </div>
          <div className="topbar-center" aria-label="Workspace mode">
            <button
              className={`mode-pill ${workspaceMode === "chat" ? "active" : ""}`}
              onClick={() => setWorkspaceMode("chat")}
              title="Show chat view"
            >
              <MessageSquare size={13} />
              Chat
            </button>
            <button
              className={`mode-pill ${workspaceMode === "sessions" ? "active" : ""}`}
              onClick={() => setWorkspaceMode("sessions")}
              title="Show saved sessions"
            >
              <Layers size={13} />
              Sessions
            </button>
            <button
              className={`mode-pill ${workspaceMode === "ideas" ? "active" : ""}`}
              onClick={() => setWorkspaceMode("ideas")}
              title="Show saved ideas"
            >
              <FileText size={13} />
              Ideas
            </button>
            <button
              className={`mode-pill ${workspaceMode === "threads" ? "active" : ""}`}
              onClick={() => setWorkspaceMode("threads")}
              title="Show threads and subagents"
            >
              <GitBranch size={13} />
              Threads
            </button>
            <button
              className={`mode-pill ${workspaceMode === "events" ? "active" : ""}`}
              onClick={() => setWorkspaceMode("events")}
              title="Show events view"
            >
              <Activity size={13} />
              Events
            </button>
            <button
              className={`mode-pill ${workspaceMode === "artifacts" ? "active" : ""}`}
              onClick={() => setWorkspaceMode("artifacts")}
              title="Show artifacts view"
            >
              <Braces size={13} />
              Artifacts
            </button>
          </div>
          <div className="topbar-actions">
            {!connected ? (
              <button className="mobile-connect" onClick={createSession} title="Start a new AceAI session">
                <Plus size={16} />
                New Chat
              </button>
            ) : null}
            <button onClick={refreshSnapshot} disabled={!connected} title="Refresh session snapshot">
              <RefreshCw size={16} />
            </button>
            <button onClick={cancelRun} disabled={!connected || !isRunning} title="Cancel active run">
              <CircleStop size={16} />
            </button>
          </div>
        </header>

        {error ? (
          <div className="notice">
            <AlertTriangle size={16} />
            {error}
          </div>
        ) : null}

        {notice ? (
          <div className="notice good-notice">
            <Check size={16} />
            {notice}
          </div>
        ) : null}

        <div
          className="content-grid"
          style={{ "--inspector-width": `${inspectorWidth}px` } as CSSProperties}
        >
          <section className="conversation-pane" aria-label="Transcript">
            <div className="pane-header">
              <div>
                <span>Conversation</span>
                <strong>{isRunning ? "Streaming" : connected ? "Ready" : "Offline"}</strong>
              </div>
              <div className="context-strip" aria-label="Context">
                <span>
                  <SquareTerminal size={13} />
                  {activeThread?.role ?? "agent"}
                </span>
                <span>
                  <Braces size={13} />
                  {latestRun ? "run bound" : "no run"}
                </span>
                <span>
                  <FileText size={13} />
                  {events.length}
                </span>
                {observableUsage?.total_cost_usd ? (
                  <span>
                    <Database size={13} />
                    {formatUsd(observableUsage.total_cost_usd)}
                  </span>
                ) : null}
              </div>
              <div className="run-chip">
                {isRunning ? <Sparkles size={14} /> : <Clock3 size={14} />}
                {isRunning ? "running" : latestRun ? "completed" : "idle"}
              </div>
            </div>

            {workspaceMode === "chat" ? <div className="transcript">
              {visibleTranscript.length === 0 ? (
                <div className="empty-state">
                  <div className="empty-icon">
                    <Bot size={32} />
                  </div>
                  <strong>{connected ? "Session is ready" : "Choose a session"}</strong>
                  <span>{connected ? "Send the first message from the composer." : "Open a saved session or start a new chat."}</span>
                </div>
              ) : (
                visibleTranscript.map((item) => (
                  <article className={`message ${item.role}`} key={item.id}>
                    <div className="message-meta">
                      <span>{item.role}</span>
                      <div>
                        <time>{formatTime(item.time)}</time>
                        <button
                          type="button"
                          className="tiny-icon-button"
                          onClick={() => void copyText(item.text, "Message")}
                          title="Copy message"
                          aria-label="Copy message"
                        >
                          <Copy size={13} />
                        </button>
                      </div>
                    </div>
                    <MarkdownText text={item.text} onOpenFile={(path) => void openProjectFile(path)} />
                  </article>
                ))
              )}
              {hasTranscriptWorkHistory ? (
                <TranscriptWorkHistory
                  durationLabel={workHistoryLabel}
                  eventKinds={observableEventKinds}
                  expanded={showTranscriptWorkHistory}
                  items={timeline}
                  onToggle={() => setShowTranscriptWorkHistory((visible) => !visible)}
                  toolCalls={observableToolCalls}
                />
              ) : null}
              <div ref={transcriptEndRef} />
            </div> : null}

            {workspaceMode === "sessions" ? (
              <div className="workspace-panel">
                <div className="workspace-panel-header">
                  <div>
                    <strong>Sessions</strong>
                    <span>{visibleSessions.length} saved chats / {emptySessionCount} empty</span>
                  </div>
                  <button
                    type="button"
                    className="workspace-header-action"
                    onClick={() => void clearEmptySessions()}
                    disabled={emptySessionCount === 0}
                    title="Delete sessions with no events"
                  >
                    <Trash2 size={14} />
                    Clear empty
                  </button>
                </div>
                <label className="session-workspace-search">
                  <Search size={14} />
                  <input
                    value={sessionQuery}
                    onChange={(event) => setSessionQuery(event.target.value)}
                    placeholder="Search sessions"
                  />
                </label>
                <div className="session-workspace-list">
                  {sessionsLoading ? <div className="subtle-empty">Loading sessions...</div> : null}
                  {visibleSessions.length === 0 && !sessionsLoading ? (
                    <div className="empty-state inline-empty">
                      <strong>No saved sessions</strong>
                      <span>Start a new chat or adjust the search.</span>
                    </div>
                  ) : null}
                  {visibleSessions.map((session) => {
                    const active = snapshot?.session.session_id === session.session_id;
                    return (
                      <div className={`session-workspace-row ${active ? "active" : ""}`} key={session.session_id}>
                        <button
                          className="session-workspace-open"
                          onClick={() => {
                            connectSession(session.session_id);
                            setWorkspaceMode("chat");
                          }}
                          title={`Open ${session.title}`}
                        >
                          <div>
                            <strong>{session.title}</strong>
                            <span>{session.project_name} / {formatShortDate(session.updated_at)}</span>
                          </div>
                          <code>{session.thread_count} threads</code>
                        </button>
                        <button
                          aria-label={`Delete ${session.title} ${session.session_id}`}
                          className="session-workspace-delete"
                          onClick={() => void deleteSession(session)}
                          title="Delete session"
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : null}

            {workspaceMode === "ideas" ? (
              <div className="workspace-panel">
                <div className="workspace-panel-header">
                  <strong>Ideas</strong>
                  <span>{ideas.length} saved notes</span>
                </div>
                <section className="idea-capture-panel" aria-label="Capture idea">
                  <textarea
                    value={newIdeaDraft}
                    onChange={(event) => setNewIdeaDraft(event.target.value)}
                    placeholder="Capture an idea for later context"
                  />
                  <button type="button" className="primary" onClick={() => void saveWorkspaceIdea()} disabled={!newIdeaDraft}>
                    Save Idea
                  </button>
                </section>
                {ideas.length === 0 ? (
                  <div className="empty-state inline-empty">
                    <strong>No saved ideas</strong>
                    <span>Capture an idea here or use /idea with text.</span>
                  </div>
                ) : (
                  <div className="idea-workspace">
                    <div className="idea-workspace-list">
                      {ideas.map((idea) => (
                        <button
                          className={idea.index === selectedIdea?.index ? "active" : ""}
                          key={idea.idea_id}
                          onClick={() => selectIdea(idea)}
                        >
                          <span>{ideaTitle(idea.content)}</span>
                          <code>@idea:{idea.index}</code>
                        </button>
                      ))}
                    </div>
                    {selectedIdea ? (
                      <article className="idea-workspace-detail">
                        <div className="idea-detail-head">
                          <div>
                            <strong>@idea:{selectedIdea.index}</strong>
                            <span>{selectedIdea.project_name} / {formatShortDate(selectedIdea.created_at)}</span>
                          </div>
                          <button
                            type="button"
                            className="tiny-icon-button"
                            onClick={() => void copyText(selectedIdea.content, "Idea")}
                            title="Copy idea"
                            aria-label="Copy idea"
                          >
                            <Copy size={13} />
                          </button>
                        </div>
                        <textarea
                          value={ideaDraft}
                          onChange={(event) => setIdeaDraft(event.target.value)}
                          aria-label="Idea content"
                        />
                        <div className="idea-workspace-actions">
                          <button type="button" onClick={referenceSelectedIdea}>
                            <Search size={13} />
                            Reference
                          </button>
                          <button type="button" onClick={() => void updateSelectedIdea()}>
                            <Save size={13} />
                            Save
                          </button>
                          <button type="button" onClick={() => void deleteSelectedIdea()}>
                            <Trash2 size={13} />
                            Delete
                          </button>
                        </div>
                      </article>
                    ) : null}
                  </div>
                )}
              </div>
            ) : null}

            {workspaceMode === "threads" ? (
              <div className="workspace-panel">
                <div className="workspace-panel-header">
                  <strong>Threads</strong>
                  <span>{snapshot?.threads.length ?? 0} active lanes</span>
                </div>
                {!snapshot ? (
                  <div className="empty-state inline-empty">
                    <strong>No active session</strong>
                    <span>Open a session to inspect main and subagent threads.</span>
                  </div>
                ) : (
                  <div className="thread-workspace-list">
                    {snapshot.threads.map((thread) => (
                      <button
                        className={thread.thread_id === snapshot.active_thread_id ? "active" : ""}
                        key={thread.thread_id}
                        onClick={() => {
                          void switchThread(thread.thread_id);
                          setWorkspaceMode("chat");
                        }}
                      >
                        <div>
                          <strong>{thread.title || thread.thread_id}</strong>
                          <span>{thread.role} / {thread.status}</span>
                        </div>
                        <code>{shortThreadId(thread.thread_id)}</code>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ) : null}

            {workspaceMode === "events" ? (
              <div className="workspace-panel">
                <div className="workspace-panel-header">
                  <strong>Events</strong>
                  <span>{events.length} session events</span>
                </div>
                {observableDebugEvents.length === 0 ? (
                  <div className="empty-state inline-empty">
                    <strong>No events yet</strong>
                    <span>Open a session or run a turn to inspect event payloads.</span>
                  </div>
                ) : (
                  <div className="event-workspace">
                    <div className="workspace-event-list">
                      {observableDebugEvents.map((event) => (
                        <button
                          className={event.event_id === selectedDebugEvent?.event_id ? "active" : ""}
                          key={event.event_id}
                          onClick={() => setSelectedDebugEventId(event.event_id)}
                        >
                          <span>{event.kind}</span>
                          <small>{shortEventId(event.event_id)}</small>
                        </button>
                      ))}
                    </div>
                    {selectedDebugEvent ? (
                      <article className="event-detail">
                        <div className="event-detail-head">
                          <div>
                            <strong>{selectedDebugEvent.kind}</strong>
                            <span>{selectedDebugEvent.thread_id} / {formatTime(selectedDebugEvent.created_at)}</span>
                          </div>
                          <div className="detail-actions">
                            <code>{shortEventId(selectedDebugEvent.event_id)}</code>
                            <button
                              type="button"
                              className="tiny-icon-button"
                              onClick={() => void copyText(debugPayloadPreview(selectedDebugEvent), "Event")}
                              title="Copy event payload"
                              aria-label="Copy event payload"
                            >
                              <Copy size={13} />
                            </button>
                          </div>
                        </div>
                        <pre>{debugPayloadPreview(selectedDebugEvent)}</pre>
                      </article>
                    ) : null}
                  </div>
                )}
              </div>
            ) : null}

            {workspaceMode === "artifacts" ? (
              <div className="workspace-panel">
                <div className="workspace-panel-header">
                  <strong>Artifacts</strong>
                  <span>{artifacts.length} session outputs</span>
                </div>
                {artifacts.length === 0 ? (
                  <div className="empty-state inline-empty">
                    <strong>No artifacts yet</strong>
                    <span>Tool outputs, changed files, and subagent handoffs will appear here.</span>
                  </div>
                ) : (
                  <div className="artifact-workspace">
                    <div className="artifact-list">
                      {artifacts.map((artifact) => (
                        <button
                          className={artifact.id === selectedArtifact?.id ? "active" : ""}
                          key={artifact.id}
                          onClick={() => inspectArtifact(artifact)}
                        >
                          <div>
                            <strong>{artifact.title}</strong>
                            <span>{artifact.subtitle}</span>
                          </div>
                          <code>{artifact.kind}</code>
                        </button>
                      ))}
                    </div>
                    {selectedArtifact ? (
                      <article className="artifact-detail">
                        <div className="artifact-detail-head">
                          <div>
                            <strong>{selectedArtifact.title}</strong>
                            <span>{selectedArtifact.subtitle}</span>
                          </div>
                          <div className="detail-actions">
                            <code>{selectedArtifact.status}</code>
                            <button
                              type="button"
                              className="tiny-icon-button"
                              onClick={() => void copyText(selectedArtifact.content, "Artifact")}
                              title="Copy artifact"
                              aria-label="Copy artifact"
                            >
                              <Copy size={13} />
                            </button>
                          </div>
                        </div>
                        <pre>{selectedArtifact.content}</pre>
                      </article>
                    ) : null}
                  </div>
                )}
              </div>
            ) : null}

            {pendingApproval ? (
              <section className="approval-card" aria-label="Tool approval">
                <div className="approval-main">
                  <div className="approval-title">
                    <div>
                      <strong>{pendingApproval.tool_name}</strong>
                      <span>{pendingApproval.policy || "approval required"} / {pendingApproval.reason || "Review before running this tool."}</span>
                    </div>
                    <code>{pendingApproval.call.call_id}</code>
                  </div>
                  <ApprovalArguments argumentsJson={pendingApproval.call.arguments} />
                </div>
                <div className="approval-actions">
                  <button type="button" onClick={() => void rejectPendingTool()}>
                    Reject
                  </button>
                  <button type="button" className="primary" onClick={() => void approvePendingTool()}>
                    Approve
                  </button>
                </div>
              </section>
            ) : null}

            {queuedTurns.length > 0 ? (
              <section className="queued-turns" aria-label="Queued messages">
                <div className="queued-title">
                  <Clock3 size={14} />
                  <span>{queuedTurns.length} queued</span>
                </div>
                <div className="queued-list">
                  {queuedTurns.map((turn, index) => (
                    <div className="queued-row" key={turn.id}>
                      <MarkdownText text={turn.content} compact />
                      <button type="button" onClick={() => void steerQueuedTurn(index)}>
                        Send now
                      </button>
                      <button type="button" onClick={() => cancelQueuedTurn(index)}>
                        Cancel
                      </button>
                    </div>
                  ))}
                </div>
              </section>
            ) : null}

            <form className="composer" onSubmit={submitMessage}>
              <div className="composer-card">
                <div className="composer-input-area">
                <textarea
                  ref={composerRef}
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={handleComposerKeyDown}
                  placeholder={connected ? composerPlaceholder(isRunning, isBlockedForApproval) : "Choose a session first"}
                  disabled={!connected}
                />
                {showCommandMenu ? (
                  <div className="command-menu" role="listbox" aria-label="Commands">
                    {commandMatches.map((command, index) => (
                      <button
                        type="button"
                        className={index === selectedCommandIndex ? "selected" : ""}
                        key={command.name}
                        onMouseDown={(event) => event.preventDefault()}
                        onClick={() => applyCommand(command)}
                      >
                        <code>{command.name}</code>
                        <span>{command.label}</span>
                        <small>{command.hint}</small>
                      </button>
                    ))}
                  </div>
                ) : null}
                {showReferenceMenu ? (
                  <div className="reference-menu" role="listbox" aria-label="References">
                    {referenceItems.map((item, index) => (
                      <button
                        type="button"
                        className={index === selectedReferenceIndex ? "selected" : ""}
                        key={`${item.kind}-${item.value}`}
                        onMouseDown={(event) => event.preventDefault()}
                        onClick={() => applyReference(item)}
                      >
                        <code>{item.value}</code>
                        <span>{item.description}</span>
                      </button>
                    ))}
                  </div>
                ) : null}
              </div>
                <div className="composer-toolbar" aria-label="Composer tools">
                  <div className="composer-tool-group">
                    <button type="button" className="composer-icon-button" onClick={() => setComposerValue("@")} title="Add a reference" aria-label="Add a reference">
                      <Plus size={18} />
                    </button>
                    <button type="button" className="composer-access-button" title="Tool access">
                      <AlertTriangle size={15} />
                      Full access
                      <ChevronDown size={15} />
                    </button>
                  </div>
                  <div className="composer-tool-group composer-tool-group-right">
                    <span className={`composer-run-state ${composerStatus}`}>
                      <Clock3 size={15} />
                      {composerStatus}
                    </span>
                    <button type="button" className="composer-model-button" title="Selected model">
                      <Sparkles size={15} />
                      {formatModelName(runtime.selected_model)}
                      <span>{formatReasoningLevel(runtime.reasoning_level)}</span>
                      <ChevronDown size={15} />
                    </button>
                    <button type="button" className="composer-icon-button" onClick={() => setWorkspaceMode("ideas")} title="Open ideas" aria-label="Open ideas">
                      <FileText size={16} />
                    </button>
                    <button type="button" className="composer-icon-button" onClick={() => setWorkspaceMode("events")} title="Open events" aria-label="Open events">
                      <Activity size={16} />
                    </button>
                    <button type="button" className="composer-icon-button" title="Voice input" aria-label="Voice input" disabled>
                      <Mic size={16} />
                    </button>
                    <button type="submit" className="composer-send-button" disabled={!connected || !input} title="Send">
                      <Send size={17} />
                    </button>
                  </div>
                </div>
              </div>
            </form>
          </section>

          <div
            aria-label="Resize workspace"
            className="split-resizer"
            onMouseDown={startInspectorResize}
            role="separator"
            title="Drag to resize workspace"
          />

          <aside
            className={`inspector ${hasWorkspaceObject ? "object-open" : ""}`}
            aria-label="Run inspector"
          >
            <section className="inspector-section object-inspector" aria-label="Workspace object">
              <div className="section-title">
                <FileText size={15} />
                Workspace
              </div>
              {fileLoading ? (
                <div className="object-empty">Opening file...</div>
              ) : openFile ? (
                <article className="file-inspector">
                  <div className="object-head" onDoubleClick={() => setFileEditMode(true)}>
                    <div>
                      <strong>{openFile.path}</strong>
                      <span>{formatBytes(openFile.size)} / {formatShortDate(openFile.updated_at)} / {fileEditMode ? "Editing" : "Read only"}</span>
                    </div>
                    <code>{fileEditMode ? "editing" : "read only"}</code>
                    <label className="theme-select" title="Editor theme">
                      <span>Theme</span>
                      <select value={monacoTheme} onChange={(event) => setMonacoTheme(event.target.value as MonacoThemeChoice)}>
                        {MONACO_THEME_OPTIONS.map((option) => (
                          <option key={option.value} value={option.value}>{option.label}</option>
                        ))}
                      </select>
                    </label>
                    <button
                      type="button"
                      className="tiny-icon-button"
                      onClick={() => {
                        setOpenFile(null);
                        setFileDraft("");
                        setFileEditMode(false);
                      }}
                      title="Close file"
                      aria-label="Close file"
                    >
                      <CircleStop size={13} />
                    </button>
                  </div>
                  <div className={`file-editor-shell ${fileEditMode ? "editing" : "read-only"}`} onDoubleClick={() => setFileEditMode(true)}>
                    <Editor
                      className="file-editor"
                      beforeMount={defineAceAIMonacoTheme}
                      height="100%"
                      key={`${openFile.path}:${inspectorWidth}`}
                      language={languageForPath(openFile.path)}
                      onChange={(value) => setFileDraft(value ?? "")}
                      options={{
                        automaticLayout: true,
                        domReadOnly: !fileEditMode,
                        fontFamily: "SFMono-Regular, Consolas, Liberation Mono, monospace",
                        fontSize: 12,
                        minimap: { enabled: true },
                        readOnly: !fileEditMode,
                        readOnlyMessage: { value: "Double-click to edit this file." },
                        renderLineHighlight: "all",
                        scrollBeyondLastLine: false,
                        smoothScrolling: true,
                        tabSize: 2,
                        wordWrap: "on"
                      }}
                      theme={monacoThemeName(monacoTheme)}
                      value={fileDraft}
                    />
                    {!fileEditMode ? <div className="file-readonly-hint">Double-click to edit</div> : null}
                  </div>
                  <div className="object-actions">
                    <button type="button" onClick={() => void copyText(fileDraft, "File")}>
                      <Copy size={13} />
                      Copy
                    </button>
                    {fileEditMode ? (
                      <>
                        <button type="button" onClick={() => {
                          setFileDraft(openFile.content);
                          setFileEditMode(false);
                        }}>
                          Cancel
                        </button>
                        <button type="button" className="primary" onClick={() => void saveOpenFile()} disabled={fileDraft === openFile.content}>
                          <Save size={13} />
                          Save
                        </button>
                      </>
                    ) : (
                      <button type="button" onClick={() => setFileEditMode(true)}>
                        <Edit3 size={13} />
                        Edit
                      </button>
                    )}
                  </div>
                </article>
              ) : inspectedArtifact ? (
                <article className="artifact-inspector">
                  <div className="object-head">
                    <div>
                      <strong>{inspectedArtifact.title}</strong>
                      <span>{inspectedArtifact.subtitle}</span>
                    </div>
                    <code>{inspectedArtifact.kind}</code>
                  </div>
                  <pre>{inspectedArtifact.content}</pre>
                  <div className="object-actions">
                    <button type="button" onClick={() => void copyText(inspectedArtifact.content, "Artifact")}>
                      <Copy size={13} />
                      Copy
                    </button>
                  </div>
                </article>
              ) : (
                <div className="object-empty">Select a file reference or artifact to inspect it here.</div>
              )}
            </section>

            {!hasWorkspaceObject ? (
              <div className="work-history" aria-label="Work history">
                <section className="inspector-section" ref={statsRef}>
                  <div className="section-title">
                    <PanelRight size={15} />
                    Run
                  </div>
                  <div className="run-card">
                    <div>
                      <span>State</span>
                      <strong>{pendingApproval ? "approval" : isRunning ? "active" : latestRun ? "settled" : "waiting"}</strong>
                    </div>
                    <div>
                      <span>Thread</span>
                      <strong>{activeThread?.status ?? "main"}</strong>
                    </div>
                  </div>
                </section>

                <section className="inspector-section">
                  <div className="section-title">
                    <Database size={15} />
                    Usage
                  </div>
                  <div className="usage-grid">
                    <Metric label="Tokens" valueText={formatCompactNumber(observableUsage?.total_tokens ?? 0)} />
                    <Metric label="Input" valueText={formatCompactNumber(observableUsage?.input_tokens ?? 0)} />
                    <Metric label="Output" valueText={formatCompactNumber(observableUsage?.output_tokens ?? 0)} />
                    <Metric label="Cost" valueText={formatUsd(observableUsage?.total_cost_usd ?? 0)} />
                  </div>
                  <div className="usage-detail">
                    <span>Context {formatCompactNumber(observableUsage?.context_tokens ?? 0)}</span>
                    <span>Cached {formatCompactNumber(observableUsage?.cached_input_tokens ?? 0)}</span>
                    <span>Cache {formatPercent(observableUsage?.cache_hit_rate)}</span>
                  </div>
                </section>

            <section className="inspector-section" ref={threadsRef}>
              <div className="section-title">
                <GitBranch size={15} />
                Threads
              </div>
              {!snapshot ? (
                <div className="subtle-empty">No active session.</div>
              ) : (
                <div className="thread-list">
                  {snapshot.threads.map((thread) => (
                    <button
                      className={thread.thread_id === snapshot.active_thread_id ? "active" : ""}
                      key={thread.thread_id}
                      onClick={() => void switchThread(thread.thread_id)}
                    >
                      <div>
                        <strong>{thread.title || thread.thread_id}</strong>
                        <span>{thread.role} / {thread.status}</span>
                      </div>
                      <code>{shortThreadId(thread.thread_id)}</code>
                    </button>
                  ))}
                </div>
              )}
            </section>

            <section className="inspector-section">
              <div className="section-title">
                <FileText size={15} />
                Ideas
              </div>
              {ideas.length === 0 ? (
                <div className="subtle-empty">No saved ideas.</div>
              ) : (
                <div className="idea-panel">
                  <div className="idea-list">
                    {ideas.map((idea) => (
                      <button
                        className={idea.index === selectedIdea?.index ? "active" : ""}
                        key={idea.idea_id}
                        onClick={() => selectIdea(idea)}
                      >
                        <span>{ideaTitle(idea.content)}</span>
                        <code>@idea:{idea.index}</code>
                      </button>
                    ))}
                  </div>
                  {selectedIdea ? (
                    <div className="idea-editor">
                      <textarea
                        value={ideaDraft}
                        onChange={(event) => setIdeaDraft(event.target.value)}
                        aria-label="Idea content"
                      />
                      <div className="idea-actions">
                        <button type="button" onClick={referenceSelectedIdea}>
                          <Search size={13} />
                          Reference
                        </button>
                        <button type="button" onClick={() => void updateSelectedIdea()}>
                          <Save size={13} />
                          Save
                        </button>
                        <button type="button" onClick={() => void deleteSelectedIdea()}>
                          <Trash2 size={13} />
                          Delete
                        </button>
                      </div>
                    </div>
                  ) : null}
                </div>
              )}
            </section>

            <section className="inspector-section" ref={timelineRef}>
              <div className="section-title">
                <Sparkles size={15} />
                Trajectory
              </div>
              {observableTrajectory.length === 0 && timeline.length === 0 ? (
                <div className="subtle-empty">No trajectory entries.</div>
              ) : (
                <div className="timeline-list">
                  {timeline.map((item) => (
                    <div className={`timeline-row ${item.tone}`} key={item.id}>
                      <span />
                      <div>
                        <strong>{item.title}</strong>
                        <small>{item.detail}</small>
                      </div>
                    </div>
                  ))}
                  {(observableTrajectory.length > 0 ? observableTrajectory.slice(-10).reverse() : []).map((item) => (
                    <div className={`timeline-row ${trajectoryTone(item.kind)}`} key={item.event_id}>
                      <span />
                      <div>
                        <strong>{item.kind}</strong>
                        <small>{item.summary || formatTime(item.created_at)}</small>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </section>

            <section className="inspector-section">
              <div className="section-title">
                <SquareTerminal size={15} />
                Tool Calls
              </div>
              {observableToolCalls.length === 0 ? (
                <div className="subtle-empty">No tool calls.</div>
              ) : (
                <div className="tool-stat-list">
                  {observableToolCalls.map((tool) => (
                    <div className="tool-stat-row" key={tool.name}>
                      <div>
                        <strong>{tool.name}</strong>
                        <span>{tool.calls} calls / {tool.succeeded} ok / {tool.failed} failed</span>
                      </div>
                      <code>{tool.approval_requests}</code>
                    </div>
                  ))}
                </div>
              )}
            </section>

            <section className="inspector-section" ref={eventsRef}>
              <div className="section-title">
                <GitBranch size={15} />
                Event Mix
              </div>
              {observableEventKinds.length === 0 ? (
                <div className="subtle-empty">No session events.</div>
              ) : (
                <div className="event-kinds">
                  {observableEventKinds.map((item) => (
                    <div className="kind-row" key={item.kind}>
                      <span>{item.kind}</span>
                      <strong>{item.count}</strong>
                    </div>
                  ))}
                </div>
              )}
            </section>

            <section className="inspector-section">
              <div className="section-title">
                <Braces size={15} />
                Debug
              </div>
              {observableDebugEvents.length === 0 ? (
                <div className="subtle-empty">No debug events.</div>
              ) : (
                <div className="debug-panel">
                  <div className="debug-list">
                    {observableDebugEvents.slice(0, 10).map((event) => (
                      <button
                        className={event.event_id === selectedDebugEvent?.event_id ? "active" : ""}
                        key={event.event_id}
                        onClick={() => setSelectedDebugEventId(event.event_id)}
                      >
                        <span>{event.kind}</span>
                        <code>{shortEventId(event.event_id)}</code>
                      </button>
                    ))}
                  </div>
                  {selectedDebugEvent ? <pre>{debugPayloadPreview(selectedDebugEvent)}</pre> : null}
                </div>
              )}
            </section>

            <section className="inspector-section protocol-section">
              <div className="section-title">
                <Activity size={15} />
                Activity
              </div>
              {activity.length === 0 ? (
                <div className="subtle-empty">No live activity.</div>
              ) : (
                <div className="event-list">
                  {activity.map((item, index) => (
                    <div className="event-row" key={`${item.seq ?? index}-${item.event}-${item.ref ?? ""}`}>
                      <div>
                        <strong>{friendlySocketEvent(item.event)}</strong>
                        <span>{friendlyTopic(item.topic)}</span>
                      </div>
                      <code>{item.seq ?? "-"}</code>
                    </div>
                  ))}
                </div>
              )}
            </section>
              </div>
            ) : null}
          </aside>
        </div>
      </section>
    </main>
  );
}

function Metric({ label, value, valueText }: { label: string; value?: number; valueText?: string }) {
  return (
    <div className="metric">
      <strong>{valueText ?? value}</strong>
      <span>{label}</span>
    </div>
  );
}

function StatusDot({ state }: { state: ConnectionState }) {
  if (state === "connected") return <Check className="status-icon good" size={17} />;
  if (state === "error") return <AlertTriangle className="status-icon bad" size={17} />;
  if (state === "closed") return <WifiOff className="status-icon muted-icon" size={17} />;
  return <span className={`status-dot ${state}`} />;
}

function TranscriptWorkHistory({
  durationLabel,
  eventKinds,
  expanded,
  items,
  onToggle,
  toolCalls
}: {
  durationLabel: string;
  eventKinds: { kind: string; count: number }[];
  expanded: boolean;
  items: TimelineItem[];
  onToggle: () => void;
  toolCalls: { name: string; calls: number; succeeded: number; failed: number; approval_requests: number }[];
}) {
  return (
    <section className={`transcript-history ${expanded ? "expanded" : ""}`} aria-label="Work history">
      <button className="transcript-history-toggle" type="button" onClick={onToggle}>
        <span>{durationLabel}</span>
        {expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
      </button>
      {expanded ? (
        <div className="transcript-history-body">
          {items.length > 0 ? (
            <div className="transcript-history-list">
              {items.map((item) => (
                <div className={`transcript-history-row ${item.tone}`} key={item.id}>
                  <span />
                  <div>
                    <strong>{item.title}</strong>
                    <small>{item.detail}</small>
                  </div>
                </div>
              ))}
            </div>
          ) : null}
          {toolCalls.length > 0 ? (
            <div className="transcript-history-summary">
              {toolCalls.slice(0, 6).map((tool) => (
                <div key={tool.name}>
                  <strong>{tool.name}</strong>
                  <span>{tool.calls} calls / {tool.succeeded} ok / {tool.failed} failed</span>
                </div>
              ))}
            </div>
          ) : null}
          {eventKinds.length > 0 ? (
            <div className="transcript-history-kinds">
              {eventKinds.slice(0, 8).map((item) => (
                <span key={item.kind}>{item.kind} {item.count}</span>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}

function MarkdownText({ text, compact = false, onOpenFile }: { text: string; compact?: boolean; onOpenFile?: (path: string) => void }) {
  return (
    <div className={compact ? "markdown markdown-compact" : "markdown"}>
      <ReactMarkdown
        components={{
          code({ children, className, ...props }) {
            const value = String(children).trim();
            const filePath = filePathFromText(value);
            if (!className && filePath !== null && onOpenFile !== undefined) {
              return (
                <button
                  type="button"
                  className="file-reference-button"
                  onClick={() => onOpenFile(filePath)}
                  title={`Preview ${filePath}`}
                >
                  {value}
                </button>
              );
            }
            return <code className={className} {...props}>{children}</code>;
          },
          a({ children, href, ...props }) {
            const filePath = typeof href === "string" ? filePathFromText(href) : null;
            if (filePath !== null && onOpenFile !== undefined) {
              return (
                <button
                  type="button"
                  className="file-reference-button"
                  onClick={() => onOpenFile(filePath)}
                  title={`Preview ${filePath}`}
                >
                  {children}
                </button>
              );
            }
            return <a href={href} {...props}>{children}</a>;
          }
        }}
        remarkPlugins={MARKDOWN_PLUGINS}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}

function ApprovalArguments({ argumentsJson }: { argumentsJson: string }) {
  const parsed = jsonObjectFromString(argumentsJson);
  if (parsed === null) {
    return <pre className="approval-raw">{argumentsJson}</pre>;
  }
  const entries = Object.entries(parsed);
  if (entries.length === 0) {
    return <div className="approval-empty">No arguments.</div>;
  }
  return (
    <div className="approval-argument-grid">
      {entries.map(([key, value]) => (
        <div className="approval-argument" key={key}>
          <span>{key}</span>
          <code>{approvalArgumentPreview(value)}</code>
        </div>
      ))}
    </div>
  );
}

function approvalArgumentPreview(value: unknown) {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return JSON.stringify(value);
  if (value === null) return "null";
  return JSON.stringify(value, null, 2);
}

function buildArtifacts(events: SessionEvent[]): ArtifactItem[] {
  return events.flatMap((event) => {
    if (event.kind !== "tool_result") return [];
    const toolName = stringPayload(event.payload, "tool_name");
    const callId = stringPayload(event.payload, "tool_call_id");
    if (toolName === null || callId === null) return [];
    const argumentsPayload = jsonObjectFromString(stringPayload(event.payload, "tool_arguments"));
    const output = stringPayload(event.payload, "output") ?? stringPayload(event.payload, "content") ?? "";
    const truncatedOutput = stringPayload(event.payload, "truncated_output");
    const outputPayload = jsonObjectFromString(output);
    const path = pathFromToolPayload(argumentsPayload) ?? pathFromToolPayload(outputPayload);
    const status = stringPayload(event.payload, "status") ?? "completed";
    const kind = artifactKind(toolName, outputPayload);
    const title = path ?? readableToolName(toolName);
    const subtitle = path === null ? readableToolName(toolName) : `${readableToolName(toolName)} / ${formatTime(event.created_at)}`;
    const content = artifactContent({
      output,
      truncatedOutput,
      outputPayload,
      argumentsPayload,
      toolName
    });
    return [
      {
        id: `${event.event_id}-${callId}`,
        kind,
        title,
        subtitle,
        status,
        createdAt: event.created_at,
        content
      }
    ];
  }).reverse();
}

function artifactKind(toolName: string, outputPayload: Record<string, unknown> | null): ArtifactItem["kind"] {
  if (toolName === "apply_patch" || toolName === "preview_patch" || toolName.includes("patch")) return "patch";
  if (toolName === "delegate_to_subagent" || toolName === "spawn_subagent" || outputPayload?.type === "subagent_audit") {
    return "subagent";
  }
  if (
    toolName.includes("file") ||
    toolName === "write_text_file" ||
    toolName === "replace_text_in_file" ||
    outputPayload?.path !== undefined
  ) {
    return "file";
  }
  return "tool";
}

function artifactContent({
  output,
  truncatedOutput,
  outputPayload,
  argumentsPayload,
  toolName
}: {
  output: string;
  truncatedOutput: string | null;
  outputPayload: Record<string, unknown> | null;
  argumentsPayload: Record<string, unknown> | null;
  toolName: string;
}) {
  const preview = truncatedOutput ?? output;
  const sections = [`tool: ${toolName}`];
  if (argumentsPayload !== null) {
    sections.push(`arguments:\n${JSON.stringify(argumentsPayload, null, 2)}`);
  }
  if (outputPayload !== null) {
    sections.push(`output:\n${JSON.stringify(outputPayload, null, 2)}`);
  } else if (preview !== "") {
    sections.push(`output:\n${preview}`);
  }
  return sections.join("\n\n");
}

function pathFromToolPayload(payload: Record<string, unknown> | null) {
  if (payload === null) return null;
  const path = payload.path;
  if (typeString(path)) return path;
  const filePath = payload.file_path;
  if (typeString(filePath)) return filePath;
  return null;
}

function jsonObjectFromString(value: string | null) {
  if (value === null) return null;
  const trimmed = value.trim();
  if (!trimmed.startsWith("{")) return null;
  try {
    const parsed = JSON.parse(trimmed) as unknown;
    if (parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
    return null;
  } catch {
    return null;
  }
}

function stringPayload(payload: Record<string, unknown>, key: string) {
  const value = payload[key];
  return typeString(value) ? value : null;
}

function typeString(value: unknown): value is string {
  return typeof value === "string";
}

function readableToolName(name: string) {
  return name.replaceAll("_", " ");
}

function ideaTitle(content: string) {
  const firstLine = content.split("\n").find((line) => line !== "");
  return _shortText(firstLine ?? content, 64);
}

function _shortText(value: string, maxLength: number) {
  if (value.length <= maxLength) return value;
  return `${value.slice(0, maxLength - 3)}...`;
}

function buildTranscript(events: SessionEvent[]): TranscriptItem[] {
  const items = events.flatMap<TranscriptItem>((event) => {
    const content = typeof event.payload.content === "string" ? event.payload.content : "";
    if (!content && event.kind !== "run_failed") return [];
    if (event.kind === "user_message") {
      return [{ id: event.event_id, role: "user", text: content, time: event.created_at }];
    }
    if (event.kind === "assistant_message" || event.kind === "run_completed") {
      return [{ id: event.event_id, role: "assistant", text: content, time: event.created_at }];
    }
    if (event.kind === "run_failed" || event.kind === "error") {
      return [{ id: event.event_id, role: "system", text: content || "Run failed", time: event.created_at }];
    }
    return [];
  });
  return dedupeTranscript(items);
}

function mergeOptimisticTranscript(
  persisted: TranscriptItem[],
  optimistic: TranscriptItem[],
  liveAssistantText: string,
) {
  const unmatched = optimistic.filter(
    (turn) =>
      !persisted.some(
        (item) => item.role === turn.role && item.text === turn.text,
      ),
  );
  const merged = [...persisted, ...unmatched];
  if (
    liveAssistantText !== "" &&
    !persisted.some((item) => item.role === "assistant" && item.text === liveAssistantText)
  ) {
    merged.push({
      id: "live-assistant",
      role: "assistant",
      text: liveAssistantText,
      time: new Date().toISOString()
    });
  }
  return merged;
}

function dedupeTranscript(items: TranscriptItem[]) {
  const deduped: TranscriptItem[] = [];
  for (const item of items) {
    const previous = deduped[deduped.length - 1];
    if (previous && previous.role === item.role && previous.text === item.text) {
      continue;
    }
    deduped.push(item);
  }
  return deduped;
}

function findLatestRun(events: SessionEvent[]) {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    if (events[index].run_id) return events[index].run_id;
  }
  return "";
}

function summarizeEventKinds(events: SessionEvent[]) {
  const counts = new Map<string, number>();
  for (const event of events) {
    counts.set(event.kind, (counts.get(event.kind) ?? 0) + 1);
  }
  return Array.from(counts.entries())
    .map(([kind, count]) => ({ kind, count }))
    .sort((left, right) => right.count - left.count);
}

function filterSessions(sessions: SessionListItem[], query: string) {
  if (!query) return sessions.slice(0, 30);
  const lowered = query.toLowerCase();
  return sessions.filter((session) => {
    return (
      session.title.toLowerCase().includes(lowered) ||
      session.project_name.toLowerCase().includes(lowered) ||
      session.session_id.toLowerCase().includes(lowered)
    );
  }).slice(0, 30);
}

function matchingCommands(input: string) {
  if (!input.startsWith("/")) return [];
  const commandName = input.split(/\s+/, 1)[0];
  return COMPOSER_COMMANDS.filter((command) => command.name.startsWith(commandName)).slice(0, 10);
}

async function readApiJson<T>(response: Response, label: string): Promise<T> {
  const text = await response.text();
  if (!response.ok) {
    throw new Error(apiErrorMessage(label, response, text));
  }
  if (text === "") {
    throw new Error(`${label} failed: AceAI GUI backend returned an empty response.`);
  }
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new Error(`${label} failed: AceAI GUI backend returned invalid JSON.`);
  }
}

function apiErrorMessage(label: string, response: Response, body: string) {
  if (response.status === 502 || response.status === 503 || body === "") {
    return `${label} failed: AceAI GUI backend is unavailable. Restart aceai-gui.`;
  }
  return `${label} failed (${response.status}).`;
}

function activeReferencePrefix(input: string) {
  const match = /(^|\s)@([^\s]*)$/.exec(input);
  if (!match) return null;
  return match[2];
}

function replaceActiveReference(input: string, reference: string) {
  return input.replace(/(^|\s)@([^\s]*)$/, (_match, prefix: string) => `${prefix}${reference} `);
}

function filePathFromReference(reference: string) {
  return reference.startsWith("@") ? reference.slice(1) : reference;
}

function filePathFromText(value: string) {
  const cleaned = value.replace(/^@/, "").replace(/^[`"']+|[`"',.;:!?）\])]+$/g, "");
  if (cleaned.startsWith("http://") || cleaned.startsWith("https://")) return null;
  if (!cleaned.includes("/") && !cleaned.startsWith(".")) return null;
  if (!/\.(md|mdx|txt|py|ts|tsx|js|jsx|json|css|html|yml|yaml|toml|sh|zsh|bash)$/i.test(cleaned)) return null;
  return cleaned;
}

function monacoThemeName(choice: MonacoThemeChoice) {
  return MONACO_THEME_OPTIONS.find((option) => option.value === choice)?.theme ?? "aceai-light";
}

function defineAceAIMonacoTheme(monaco: Monaco) {
  monaco.editor.defineTheme("aceai-light", {
    base: "vs",
    inherit: true,
    rules: [
      { token: "comment", foreground: "6d7873" },
      { token: "keyword", foreground: "1c5861", fontStyle: "bold" },
      { token: "string", foreground: "21684f" },
      { token: "number", foreground: "9b5d1f" },
      { token: "type", foreground: "315d63" }
    ],
    colors: {
      "editor.background": "#fbfcfb",
      "editor.foreground": "#20302c",
      "editorLineNumber.foreground": "#6f918e",
      "editorLineNumber.activeForeground": "#1c5861",
      "editorCursor.foreground": "#195866",
      "editor.selectionBackground": "#cfe3dd",
      "editor.inactiveSelectionBackground": "#e7f0ec",
      "editor.lineHighlightBackground": "#eef5f1",
      "editorGutter.background": "#fbfcfb",
      "editorWidget.background": "#ffffff",
      "editorWidget.border": "#dce3dd",
      "minimap.background": "#fbfcfb"
    }
  });
}

function storedInspectorWidth() {
  const stored = Number(localStorage.getItem(INSPECTOR_WIDTH_KEY));
  if (Number.isFinite(stored) && stored >= MIN_INSPECTOR_WIDTH) {
    return stored;
  }
  return DEFAULT_INSPECTOR_WIDTH;
}

function formatBytes(size: number) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${Math.round(size / 1024)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

function languageForPath(path: string) {
  const extension = path.split(".").pop()?.toLowerCase();
  if (extension === "ts" || extension === "tsx") return "typescript";
  if (extension === "js" || extension === "jsx" || extension === "mjs" || extension === "cjs") return "javascript";
  if (extension === "py") return "python";
  if (extension === "json") return "json";
  if (extension === "md" || extension === "mdx") return "markdown";
  if (extension === "css") return "css";
  if (extension === "html") return "html";
  if (extension === "yaml" || extension === "yml") return "yaml";
  if (extension === "toml") return "ini";
  if (extension === "sh" || extension === "bash" || extension === "zsh") return "shell";
  return "plaintext";
}

function runtimeOf(snapshot: SnapshotPayload | null): RuntimePayload {
  return snapshot?.runtime ?? {
    queued_questions: [],
    pending_approval: null,
    is_running_suspended: false,
    active_run_id: null,
    active_run_status: null,
    provider_name: "openai",
    selected_model: "gpt-5.5",
    reasoning_level: "medium"
  };
}

function observabilityOf(snapshot: SnapshotPayload | null): ObservabilityPayload | null {
  return snapshot?.observability ?? null;
}

function snapshotQueuedTurns(snapshot: SnapshotPayload): QueuedTurn[] {
  return runtimeOf(snapshot).queued_questions.map((content, index) => ({
    id: `${snapshot.session.session_id}-queued-${index}`,
    content
  }));
}

function eventApprovalRequest(payload: Record<string, unknown>): ToolApprovalRequest | null {
  const request = payload.request;
  if (!isRecord(request)) return null;
  const call = request.call;
  if (!isRecord(call)) return null;
  if (typeof call.call_id !== "string") return null;
  if (typeof call.name !== "string") return null;
  if (typeof call.arguments !== "string") return null;
  if (typeof request.tool_name !== "string") return null;
  if (typeof request.reason !== "string") return null;
  if (typeof request.policy !== "string") return null;
  return {
    call: {
      call_id: call.call_id,
      name: call.name,
      arguments: call.arguments
    },
    tool_name: request.tool_name,
    reason: request.reason,
    policy: request.policy
  };
}

function isTerminalAgentRunEvent(payload: AppEventPayload) {
  if (payload.kind !== "agent") return false;
  const eventType = payload.event.event_type;
  if (eventType === "agent.run.completed" || eventType === "agent.run.failed") return true;
  return "final_answer" in payload.event.payload;
}

function snapshotIsRunning(runtime: RuntimePayload) {
  return runtime.is_running_suspended || runtime.active_run_status === "running" || runtime.active_run_status === "suspended";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function composerPlaceholder(isRunning: boolean, isBlockedForApproval: boolean) {
  if (isBlockedForApproval) return "Approval is waiting. New messages will queue.";
  if (isRunning) return "Agent is running. New messages will queue.";
  return "Message AceAI";
}

function formatModelName(model: string) {
  if (model.startsWith("gpt-")) return model.replace("gpt-", "");
  return model;
}

function formatReasoningLevel(level: string) {
  if (level === "auto") return "Auto";
  return level.charAt(0).toUpperCase() + level.slice(1);
}

function trajectoryTone(kind: string): TimelineItem["tone"] {
  if (kind === "run_failed" || kind === "error" || kind === "tool_failed") return "bad";
  if (kind === "run_completed" || kind === "tool_result") return "good";
  if (kind === "run_suspended" || kind === "tool_approval_requested") return "live";
  return "neutral";
}

function formatCompactNumber(value: number) {
  return value.toLocaleString([], { notation: "compact", maximumFractionDigits: 1 });
}

function formatPercent(value: number | null | undefined) {
  if (value === null || value === undefined) return "-";
  return value.toLocaleString([], { style: "percent", maximumFractionDigits: 1 });
}

function shortEventId(eventId: string) {
  return eventId.slice(0, 8);
}

function debugPayloadPreview(event: DebugEvent) {
  return JSON.stringify(event.payload, null, 2);
}

function liveTimelineItem(payload: Extract<AppEventPayload, { kind: "agent" }>): TimelineItem | null {
  const eventType = payload.event.event_type;
  const fields = payload.event.payload;
  if (eventType === "agent.llm.started") {
    return liveTimelineRow(payload, "Thinking", "Model started", "live");
  }
  if (eventType === "agent.llm.retrying") {
    return liveTimelineRow(payload, "Retrying model", stringField(fields, "error") ?? "Waiting before retry", "live");
  }
  if (eventType === "agent.tool.started") {
    return liveTimelineRow(payload, stringField(fields, "tool_name") ?? "Tool started", "Running", "live");
  }
  if (eventType === "agent.tool.completed") {
    return liveTimelineRow(payload, stringField(fields, "tool_name") ?? "Tool completed", "Completed", "good");
  }
  if (eventType === "agent.tool.failed") {
    return liveTimelineRow(payload, stringField(fields, "tool_name") ?? "Tool failed", stringField(fields, "error") ?? "Failed", "bad");
  }
  if (eventType === "agent.tool.approval_requested" || eventType === "agent.run.suspended") {
    return liveTimelineRow(payload, "Approval required", stringField(fields, "tool_name") ?? "Review tool call", "live");
  }
  if (eventType === "agent.step.completed") {
    return liveTimelineRow(payload, "Step completed", "Ready for next step", "good");
  }
  if (eventType === "agent.run.completed") {
    return liveTimelineRow(payload, "Run completed", stringField(fields, "final_answer") ?? "Completed", "good");
  }
  if (eventType === "agent.run.failed") {
    return liveTimelineRow(payload, "Run failed", stringField(fields, "error") ?? "Failed", "bad");
  }
  return null;
}

function liveTimelineRow(
  payload: Extract<AppEventPayload, { kind: "agent" }>,
  title: string,
  detail: string,
  tone: TimelineItem["tone"],
): TimelineItem {
  return {
    id: `${payload.event.event_type}-${payload.event.run_id}-${payload.event.step_id}`,
    title,
    detail,
    tone,
  };
}

function stringField(payload: Record<string, unknown>, key: string) {
  const value = payload[key];
  return typeof value === "string" ? value : null;
}

function buildTimeline(
  events: SessionEvent[],
  activity: SocketEnvelope[],
  liveItems: TimelineItem[],
  observableTrajectory: { event_id: string; kind: string; created_at: string; summary: string }[],
): TimelineItem[] {
  if (observableTrajectory.length > 0) {
    const trajectoryItems: TimelineItem[] = observableTrajectory.slice(-10).reverse().map((event) => ({
      id: event.event_id,
      title: event.kind,
      detail: event.summary || formatTime(event.created_at),
      tone: trajectoryTone(event.kind)
    }));
    return [...liveItems, ...trajectoryItems].slice(0, 10);
  }
  const sessionItems: TimelineItem[] = events.slice(-6).map((event) => ({
    id: event.event_id,
    title: event.kind,
    detail: formatTime(event.created_at) || event.thread_id,
    tone: event.kind === "run_failed" || event.kind === "error" ? "bad" : event.kind === "run_completed" ? "good" : "neutral"
  }));
  const protocolItems: TimelineItem[] = activity.slice(0, 4).map((event, index) => ({
    id: `${event.event}-${event.ref ?? event.seq ?? index}`,
    title: friendlySocketEvent(event.event),
    detail: friendlyTopic(event.topic),
    tone: event.event === "reply" ? "good" : "live"
  }));
  return [...liveItems, ...protocolItems, ...sessionItems].slice(0, 8);
}

function friendlySocketEvent(event: string) {
  if (event === "reply") return "Session synced";
  if (event === "agent.event") return "Agent update";
  if (event === "session.event") return "Session update";
  if (event === "run.cancelled") return "Run cancelled";
  return event;
}

function friendlyTopic(topic: string) {
  if (topic === "session:new") return "New session";
  if (topic.startsWith("session:")) return "Current session";
  return topic;
}

function defaultWebSocketUrl() {
  if (!window.location.host) return "ws://127.0.0.1:8765/ws";
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws`;
}

function apiBaseFromWebSocketUrl(serverUrl: string) {
  const url = new URL(serverUrl);
  url.protocol = url.protocol === "wss:" ? "https:" : "http:";
  url.pathname = "/";
  url.search = "";
  url.hash = "";
  return url.toString();
}

function apiEndpoint(apiBaseUrl: string, path: string) {
  return new URL(path, apiBaseUrl).toString();
}

function formatShortDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString([], { month: "short", day: "numeric" });
}

function formatTime(value: string) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function workHistoryDuration(events: SessionEvent[], runId: string, isRunning: boolean) {
  const runEvents = runId ? events.filter((event) => event.run_id === runId) : events;
  if (runEvents.length === 0) return isRunning ? "Working" : "Worked";
  const first = new Date(runEvents[0].created_at).getTime();
  const lastEvent = runEvents[runEvents.length - 1];
  const last = isRunning ? Date.now() : new Date(lastEvent.created_at).getTime();
  if (!Number.isFinite(first) || !Number.isFinite(last)) return isRunning ? "Working" : "Worked";
  return `${isRunning ? "Working for" : "Worked for"} ${formatDuration(Math.max(0, last - first))}`;
}

function formatDuration(milliseconds: number) {
  const totalSeconds = Math.max(1, Math.round(milliseconds / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes === 0) return `${seconds}s`;
  return `${minutes}m ${seconds}s`;
}

function sessionCost(sessionId: string | undefined, sessions: SessionListItem[]) {
  if (!sessionId) return 0;
  const session = sessions.find((item) => item.session_id === sessionId);
  return session?.total_cost_usd ?? 0;
}

function formatUsd(value: number) {
  return value.toLocaleString([], {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: value === 0 ? 0 : 4,
    maximumFractionDigits: 4
  });
}

function shortThreadId(threadId: string) {
  if (threadId === "main") return "main";
  return threadId.slice(0, 8);
}
