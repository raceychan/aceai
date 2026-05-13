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
  ExternalLink,
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
  SlidersHorizontal,
  Sparkles,
  SquareTerminal,
  TerminalSquare,
  Trash2,
  WifiOff
} from "lucide-react";
import Editor, { Monaco } from "@monaco-editor/react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import hljs from "highlight.js/lib/common";
import "highlight.js/styles/github-dark.css";
import { CSSProperties, ClipboardEvent, FormEvent, KeyboardEvent, MouseEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ApiApi, Configuration, FetchError, ResponseError } from "./generated/api";
import type {
  FilePayload as GeneratedFilePayload,
  IdeaItemPayload,
  ReferenceItemPayload,
  SessionListItemPayload,
  ThreadMetadataPayload
} from "./generated/api";
import {
  AppEventPayload,
  FilePayload,
  IdeaItem,
  ReplyPayload,
  ReferenceItem,
  SessionListItem,
  SessionEvent,
  SnapshotPayload,
  SocketEnvelope,
  ToolApprovalRequest,
  DebugEvent,
  ObservabilityPayload,
  QueuedTurnPayload,
  RuntimePayload,
  encodeEnvelope,
  isOkReply
} from "./protocol";

type ConnectionState = "idle" | "connecting" | "connected" | "closed" | "error";

type TranscriptItem = {
  id: string;
  images?: ImageAttachmentPayload[];
  reasoning?: string;
  role: "user" | "assistant" | "system";
  runId?: string;
  retries?: TimelineItem[];
  text: string;
  time: string;
  workHistory?: RunWorkHistory;
};

type PendingRequest = {
  event: string;
  resolve: (payload: ReplyPayload<unknown>) => void;
  reject: (error: Error) => void;
};

type TimelineItem = {
  content?: string;
  id: string;
  kind?: string;
  runId?: string;
  title: string;
  detail: string;
  tone: "neutral" | "good" | "bad" | "live";
};

type RunWorkHistory = {
  durationLabel: string;
  headline: string;
  isRunning: boolean;
  items: TimelineItem[];
  runId: string;
};

type ComposerCommand = {
  name: string;
  label: string;
  hint: string;
};

const MARKDOWN_PLUGINS = [remarkGfm];
const INSPECTOR_WIDTH_KEY = "aceai.gui.inspectorWidth";
const SESSION_URL_PARAM = "session";
const DEFAULT_INSPECTOR_WIDTH = 420;
const MIN_INSPECTOR_WIDTH = 320;
const MIN_CONVERSATION_WIDTH = 520;
const WORKSPACE_REVEAL_SUPPRESSION_MS = 45_000;

type QueuedTurn = {
  id: string;
  attachments: ImageAttachmentPayload[];
  content: string;
};

type QueuedTurnsResponse = {
  queued_turns: QueuedTurnPayload[];
};

type ImageAttachmentPayload = {
  mime_type: string;
  data: string;
};

type ComposerImageAttachment = ImageAttachmentPayload & {
  id: string;
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

type WorkspaceMode = "chat" | "sessions" | "ideas" | "threads" | "events" | "artifacts" | "settings";
type WorkspaceTab = "files" | "agents" | "activity" | "run";
type MonacoThemeChoice = "aceai" | "light" | "dark";
type InspectorGroupId = "run" | "context" | "work" | "signals";
type ProjectGroupedItem = {
  project_id: string;
  project_name: string;
};

type ProjectGroup<T extends ProjectGroupedItem> = {
  project_id: string;
  project_name: string;
  items: T[];
};

type ProviderOption = {
  label: string;
  value: string;
  auth_mode: string;
  api_key_env: string;
};

type ModelOption = {
  label: string;
  value: string;
};

type ToolPermissionConfig = {
  name: string;
  description: string;
  permission: "always" | "ask";
  enabled: boolean;
  tags: string[];
  max_calls_per_run?: number | null;
};

type SkillConfig = {
  name: string;
  description: string;
  location: string;
  source: string;
  builtin: boolean;
  enabled: boolean;
};

type GuiConfig = {
  provider: string;
  model: string;
  default_model: string;
  reasoning_level: string;
  compress_threshold: string;
  api_timeout_seconds: number;
  stream_start_timeout_seconds: number;
  stream_event_timeout_seconds: number;
  skill_selection_mode: string;
  enabled_skills: string[];
  disabled_providers: string[];
  api_key_set: boolean;
  api_key_env: string;
  config_path: string;
  providers: ProviderOption[];
  models: ModelOption[];
  models_by_provider: Record<string, ModelOption[]>;
  reasoning_options: string[];
  skills: SkillConfig[];
  tools: ToolPermissionConfig[];
};

const DEFAULT_INSPECTOR_GROUPS: Record<InspectorGroupId, boolean> = {
  run: true,
  context: false,
  work: true,
  signals: false
};

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
const GUI_QUERY_KEYS = {
  sessions: ["sessions"],
  ideas: ["ideas"],
  settings: ["settings"],
  references: (query: string) => ["references", query],
  file: (path: string) => ["file", path]
} as const;

export function App() {
  const queryClient = useQueryClient();
  const [serverUrl] = useState(() => localStorage.getItem("aceai.gui.ws") ?? defaultWebSocketUrl());
  const [joinRef, setJoinRef] = useState<string | null>(null);
  const [connectionState, setConnectionState] = useState<ConnectionState>("idle");
  const [input, setInput] = useState("");
  const [composerImages, setComposerImages] = useState<ComposerImageAttachment[]>([]);
  const [selectedIdeaIndex, setSelectedIdeaIndex] = useState(1);
  const [ideaDraft, setIdeaDraft] = useState("");
  const [newIdeaDraft, setNewIdeaDraft] = useState("");
  const [sessionQuery, setSessionQuery] = useState("");
  const [snapshot, setSnapshot] = useState<SnapshotPayload | null>(null);
  const [events, setEvents] = useState<SessionEvent[]>([]);
  const [activity, setActivity] = useState<SocketEnvelope[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [durationTick, setDurationTick] = useState(0);
  const [cancelArmed, setCancelArmed] = useState(false);
  const [queuedTurns, setQueuedTurns] = useState<QueuedTurn[]>([]);
  const [optimisticTurns, setOptimisticTurns] = useState<TranscriptItem[]>([]);
  const [liveRunId, setLiveRunId] = useState("");
  const [liveTimeline, setLiveTimeline] = useState<TimelineItem[]>([]);
  const [pendingApproval, setPendingApproval] = useState<ToolApprovalRequest | null>(null);
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
  const [selectedCommandIndex, setSelectedCommandIndex] = useState(0);
  const [workspaceMode, setWorkspaceMode] = useState<WorkspaceMode>("chat");
  const [workspaceOpen, setWorkspaceOpen] = useState(false);
  const [workspaceTab, setWorkspaceTab] = useState<WorkspaceTab>("files");
  const [openInspectorGroups, setOpenInspectorGroups] = useState(DEFAULT_INSPECTOR_GROUPS);
  const [settingsDraft, setSettingsDraft] = useState<GuiConfig | null>(null);
  const [settingsApiKey, setSettingsApiKey] = useState("");
  const apiBaseUrl = useMemo(() => apiBaseFromWebSocketUrl(serverUrl), [serverUrl]);
  const apiClient = useMemo(() => new ApiApi(new Configuration({ basePath: apiBasePath(apiBaseUrl) })), [apiBaseUrl]);
  const socketRef = useRef<WebSocket | null>(null);
  const activeTopicRef = useRef("session:new");
  const refCounter = useRef(0);
  const pending = useRef(new Map<string, PendingRequest>());
  const transcriptScrollRef = useRef<HTMLDivElement | null>(null);
  const stickToTranscriptEndRef = useRef(true);
  const composerRef = useRef<HTMLTextAreaElement | null>(null);
  const sessionSearchRef = useRef<HTMLInputElement | null>(null);
  const timelineRef = useRef<HTMLElement | null>(null);
  const statsRef = useRef<HTMLElement | null>(null);
  const eventsRef = useRef<HTMLElement | null>(null);
  const threadsRef = useRef<HTMLElement | null>(null);
  const workspaceDismissedAtRef = useRef<number | null>(null);
  const knownThreadIdsRef = useRef<Set<string>>(new Set());
  const activeReference = useMemo(() => activeReferencePrefix(input), [input]);
  const connected = connectionState === "connected";
  const sessionsQuery = useQuery({
    queryKey: GUI_QUERY_KEYS.sessions,
    queryFn: async () => {
      const payload = await apiClient.listSessionsApiSessionsGet();
      return payload.sessions.map(sessionListItemFromApi);
    },
    staleTime: 5000
  });
  const ideasQuery = useQuery({
    queryKey: GUI_QUERY_KEYS.ideas,
    queryFn: async () => {
      const payload = await apiClient.listIdeasApiIdeasGet();
      return payload.ideas.map(ideaItemFromApi);
    },
    staleTime: 10000
  });
  const settingsQuery = useQuery({
    queryKey: GUI_QUERY_KEYS.settings,
    queryFn: () => fetchJson<GuiConfig>(restApiUrl("/api/config")),
    staleTime: 10000
  });
  const referencesQuery = useQuery({
    queryKey: activeReference === null ? GUI_QUERY_KEYS.references("") : GUI_QUERY_KEYS.references(activeReference),
    queryFn: async ({ signal }) => {
      if (activeReference === null) return [];
      const payload = await apiClient.listReferencesApiReferencesGet({ q: activeReference }, { signal });
      return payload.items.map(referenceItemFromApi);
    },
    enabled: connected && activeReference !== null,
    staleTime: 2000
  });
  const sessions = sessionsQuery.data ?? [];
  const ideas = ideasQuery.data ?? [];
  const referenceItems = referencesQuery.data ?? [];
  const settings = settingsQuery.data ?? null;
  const sessionsLoading = sessionsQuery.isLoading;
  const deleteSessionMutation = useMutation({
    mutationFn: (sessionId: string) => apiClient.deleteSessionApiSessionsSessionIdDelete({ session_id: sessionId }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: GUI_QUERY_KEYS.sessions });
    }
  });
  const clearEmptySessionsMutation = useMutation({
    mutationFn: () => apiClient.deleteEmptySessionsApiSessionCleanupEmptyDelete(),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: GUI_QUERY_KEYS.sessions });
    }
  });
  const captureIdeaMutation = useMutation({
    mutationFn: (content: string) => apiClient.captureIdeaApiIdeasPost({ ideaCaptureRequest: { content } }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: GUI_QUERY_KEYS.ideas });
    }
  });
  const updateIdeaMutation = useMutation({
    mutationFn: (params: { index: number; content: string }) =>
      apiClient.updateIdeaApiIdeasIndexPut({
        index: params.index,
        ideaUpdateRequest: { content: params.content }
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: GUI_QUERY_KEYS.ideas });
    }
  });
  const deleteIdeaMutation = useMutation({
    mutationFn: (index: number) => apiClient.deleteIdeaApiIdeasIndexDelete({ index }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: GUI_QUERY_KEYS.ideas });
    }
  });
  const saveSettingsMutation = useMutation({
    mutationFn: (payload: { config: GuiConfig; apiKey: string }) =>
      fetchJson<GuiConfig>(restApiUrl("/api/config"), {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(configUpdatePayload(payload.config, payload.apiKey))
      }),
    onSuccess: (payload) => {
      queryClient.setQueryData(GUI_QUERY_KEYS.settings, payload);
      setSettingsDraft(payload);
      setSettingsApiKey("");
      setNotice("Settings saved.");
    }
  });
  const saveFileMutation = useMutation({
    mutationFn: (payload: { path: string; content: string }) =>
      apiClient.saveFileApiFilesPut({
        path: payload.path,
        fileSaveRequest: { content: payload.content }
      }),
    onSuccess: (payload) => {
      const nextFile = filePayloadFromApi(payload.file);
      queryClient.setQueryData(GUI_QUERY_KEYS.file(payload.file.path), nextFile);
      setOpenFile(nextFile);
      setFileDraft(payload.file.content);
      setFileEditMode(false);
      setNotice(`Saved ${payload.file.path}.`);
    }
  });
  const settingsSaving = saveSettingsMutation.isPending;

  const latestRun = useMemo(() => findLatestRun(events), [events]);
  const activeWorkRun = liveRunId || liveTimeline.find((item) => item.runId)?.runId || latestRun;
  const runReasoning = useMemo(() => buildRunReasoning(events), [events]);
  const runRetries = useMemo(() => buildRunRetries(events), [events]);
  const runWorkHistories = useMemo(
    () => buildRunWorkHistories(events, liveTimeline, activeWorkRun, isRunning),
    [events, liveTimeline, activeWorkRun, isRunning, durationTick]
  );
  const transcript = useMemo(() => buildTranscript(events, runWorkHistories, runReasoning, runRetries), [events, runWorkHistories, runReasoning, runRetries]);
  const liveAssistantWorkHistory = activeWorkRun ? runWorkHistories.get(activeWorkRun) : undefined;
  const visibleTranscript = useMemo(
    () => mergeOptimisticTranscript(transcript, optimisticTurns, liveRunId, liveAssistantWorkHistory),
    [transcript, optimisticTurns, liveRunId, liveAssistantWorkHistory]
  );
  const artifacts = useMemo(() => buildArtifacts(events), [events]);
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
  const visibleSessions = useMemo(() => filterSessions(sessions, sessionQuery), [sessions, sessionQuery]);
  const visibleSessionGroups = useMemo(() => groupByProject(visibleSessions), [visibleSessions]);
  const latestSession = sessions[0];
  const ideaGroups = useMemo(() => groupByProject(ideas), [ideas]);
  const emptySessionCount = useMemo(() => sessions.filter((session) => session.event_count === 0).length, [sessions]);
  const commandMatches = useMemo(() => matchingCommands(input), [input]);
  const connectionLabel = connectionState === "idle" ? "ready" : connectionState;
  const activeThread = snapshot?.threads.find((thread) => thread.thread_id === snapshot.active_thread_id);
  const hasActiveSession = connected || snapshot !== null;
  const showLaunchScreen = !hasActiveSession && workspaceMode === "chat";
  const launchProjectName = snapshot?.session.project_name ?? latestSession?.project_name ?? "current project";
  const showCommandMenu = connected && commandMatches.length > 0 && input.startsWith("/");
  const showReferenceMenu = connected && activeReference !== null && referenceItems.length > 0;
  const isBlockedForApproval = pendingApproval !== null || runtime.is_running_suspended;
  const hasWorkspaceObject = fileLoading || openFile !== null || inspectedArtifact !== null;
  const threadIdsKey = snapshot?.threads.map((thread) => thread.thread_id).join("|") ?? "";
  const composerStatus = isBlockedForApproval ? "approval" : isRunning ? "running" : connected ? "ready" : "offline";
  const usageTitle = observableUsage
    ? `Tokens: ${formatCompactNumber(observableUsage.total_tokens)} total, ${formatCompactNumber(observableUsage.input_tokens)} input, ${formatCompactNumber(observableUsage.output_tokens)} output, ${formatCompactNumber(observableUsage.cached_input_tokens)} cached, ${formatUsd(observableUsage.total_cost_usd)} cost`
    : "Token usage unavailable";

  useEffect(() => {
    if (!connected || isRunning || isBlockedForApproval || queuedTurns.length === 0) return;
    const [nextTurn, ...remainingTurns] = queuedTurns;
    setQueuedTurns(remainingTurns);
    void startQueuedMessage(0, nextTurn);
  }, [connected, isRunning, isBlockedForApproval, queuedTurns]);

  useEffect(() => {
    if (!isBlockedForApproval) return;
    revealWorkspace("activity", true);
  }, [isBlockedForApproval]);

  useEffect(() => {
    if (!snapshot) {
      knownThreadIdsRef.current = new Set();
      return;
    }
    const currentThreadIds = new Set(snapshot.threads.map((thread) => thread.thread_id));
    const hasNewThread = snapshot.threads.some((thread) => !knownThreadIdsRef.current.has(thread.thread_id));
    knownThreadIdsRef.current = currentThreadIds;
    if (snapshot.threads.length > 1 && hasNewThread) {
      revealWorkspace("agents", true);
    }
  }, [threadIdsKey, snapshot]);

  useEffect(() => {
    if (!stickToTranscriptEndRef.current) return;
    const transcriptElement = transcriptScrollRef.current;
    if (transcriptElement === null) return;
    window.requestAnimationFrame(() => {
      transcriptElement.scrollTop = transcriptElement.scrollHeight;
    });
  }, [visibleTranscript.length, events.length]);

  useEffect(() => {
    if (!connected || !isRunning) return;
    const timer = window.setInterval(() => void refreshSnapshot(), 2500);
    return () => window.clearInterval(timer);
  }, [connected, isRunning]);

  useEffect(() => {
    if (!isRunning) return;
    const timer = window.setInterval(() => setDurationTick((tick) => tick + 1), 1000);
    return () => window.clearInterval(timer);
  }, [isRunning]);

  useEffect(() => {
    if (!cancelArmed) return;
    const timer = window.setTimeout(() => setCancelArmed(false), 1400);
    return () => window.clearTimeout(timer);
  }, [cancelArmed]);

  useEffect(() => {
    function handleWindowKeyDown(event: globalThis.KeyboardEvent) {
      if (event.key !== "Escape" || !isRunning) return;
      event.preventDefault();
      requestCancelRun();
    }
    window.addEventListener("keydown", handleWindowKeyDown);
    return () => window.removeEventListener("keydown", handleWindowKeyDown);
  }, [cancelArmed, isRunning]);

  useEffect(() => {
    const sessionId = sessionIdFromUrl();
    if (sessionId !== null) {
      connectSession(sessionId, { replaceUrl: true });
    }
  }, []);

  useEffect(() => {
    function handlePopState() {
      const sessionId = sessionIdFromUrl();
      if (sessionId === null) {
        resetActiveSession();
        return;
      }
      connectSession(sessionId, { replaceUrl: true });
    }

    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  useEffect(() => {
    setSelectedReferenceIndex(0);
  }, [activeReference, referenceItems.length]);

  useEffect(() => {
    if (settings === null) return;
    setSettingsDraft(settings);
    setSettingsApiKey("");
  }, [settings]);

  useEffect(() => {
    const nextIdea = ideas.find((idea) => idea.index === selectedIdeaIndex) ?? ideas[0];
    setSelectedIdeaIndex(nextIdea?.index ?? 1);
    setIdeaDraft(nextIdea?.content ?? "");
  }, [ideas]);

  useEffect(() => {
    if (sessionsQuery.error) {
      setError(apiFailureMessage("Load sessions", sessionsQuery.error));
    }
  }, [sessionsQuery.error]);

  useEffect(() => {
    if (ideasQuery.error) {
      setError(apiFailureMessage("Load ideas", ideasQuery.error));
    }
  }, [ideasQuery.error]);

  useEffect(() => {
    if (settingsQuery.error) {
      setError(apiFailureMessage("Load settings", settingsQuery.error));
    }
  }, [settingsQuery.error]);

  useEffect(() => {
    return () => {
      socketRef.current?.close();
    };
  }, []);

  function revealWorkspace(tab: WorkspaceTab, urgent: boolean) {
    const dismissedAt = workspaceDismissedAtRef.current;
    if (!urgent && dismissedAt !== null && Date.now() - dismissedAt < WORKSPACE_REVEAL_SUPPRESSION_MS) return;
    setWorkspaceTab(tab);
    setWorkspaceOpen(true);
  }

  function openWorkspace(tab: WorkspaceTab) {
    workspaceDismissedAtRef.current = null;
    setWorkspaceTab(tab);
    setWorkspaceOpen(true);
  }

  function closeWorkspace() {
    workspaceDismissedAtRef.current = Date.now();
    setWorkspaceOpen(false);
  }

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

  async function refreshSessions() {
    await queryClient.invalidateQueries({ queryKey: GUI_QUERY_KEYS.sessions });
  }

  async function refreshSettings() {
    const result = await settingsQuery.refetch();
    if (result.error) {
      setError(apiFailureMessage("Load settings", result.error));
    }
  }

  function updateSettingsDraft(updater: (draft: GuiConfig) => GuiConfig) {
    setSettingsDraft((current) => {
      if (current === null) return current;
      return updater(current);
    });
  }

  function updateToolDraft(toolName: string, updater: (tool: ToolPermissionConfig) => ToolPermissionConfig) {
    updateSettingsDraft((draft) => ({
      ...draft,
      tools: draft.tools.map((tool) => tool.name === toolName ? updater(tool) : tool)
    }));
  }

  async function saveSettings() {
    if (settingsDraft === null) return;
    setError(null);
    try {
      await saveSettingsMutation.mutateAsync({ config: settingsDraft, apiKey: settingsApiKey });
    } catch (err) {
      setError(apiFailureMessage("Save settings", err));
    }
  }

  function connectSession(sessionId: string, options: { replaceUrl?: boolean; updateUrl?: boolean } = {}) {
    const nextTopic = `session:${sessionId}`;
    if (options.updateUrl !== false) {
      writeSessionIdToUrl(sessionId, { replace: options.replaceUrl === true });
    }
    const currentSocket = socketRef.current;
    if (
      activeTopicRef.current === nextTopic &&
      currentSocket !== null &&
      (currentSocket.readyState === WebSocket.CONNECTING || currentSocket.readyState === WebSocket.OPEN)
    ) {
      return;
    }
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
    setLiveRunId("");
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
          void refreshSessions();
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
    clearSessionIdFromUrl({ replace: false });
    connectSession("new", { updateUrl: false });
  }

  function openLatestSession() {
    if (latestSession === undefined) return;
    connectSession(latestSession.session_id);
    setWorkspaceMode("chat");
  }

  function resetActiveSession() {
    socketRef.current?.close();
    pending.current.clear();
    setSnapshot(null);
    setEvents([]);
    setActivity([]);
    setQueuedTurns([]);
    setOptimisticTurns([]);
    setLiveRunId("");
    setLiveTimeline([]);
    setPendingApproval(null);
    setSelectedDebugEventId(null);
    setSelectedArtifactId(null);
    setOpenFile(null);
    setFileDraft("");
    setFileEditMode(false);
    setConnectionState("idle");
    activeTopicRef.current = "session:new";
  }

  async function deleteSession(session: SessionListItem) {
    if (!window.confirm(`Delete "${session.title}"?`)) {
      return;
    }
    try {
      await deleteSessionMutation.mutateAsync(session.session_id);
      if (snapshot?.session.session_id === session.session_id) {
        clearSessionIdFromUrl({ replace: true });
        resetActiveSession();
      }
    } catch (err) {
      setError(apiFailureMessage("Delete session", err));
    }
  }

  async function clearEmptySessions() {
    if (emptySessionCount === 0) return;
    if (!window.confirm(`Delete ${emptySessionCount} empty sessions?`)) {
      return;
    }
    try {
      const payload = await clearEmptySessionsMutation.mutateAsync();
      if (snapshot && payload.sessionIds.includes(snapshot.session.session_id)) {
        clearSessionIdFromUrl({ replace: true });
        resetActiveSession();
      }
      setNotice(`Deleted ${payload.deleted} empty sessions.`);
    } catch (err) {
      setError(apiFailureMessage("Clean empty sessions", err));
    }
  }

  async function switchThread(threadId: string) {
    if (!connected || snapshot?.active_thread_id === threadId) return;
    try {
      const response = await sendCommand<SnapshotPayload>("switch_thread", { thread_id: threadId });
      setSnapshot(response);
      setEvents((current) => mergeSessionEventLogs(current, response.events));
      setQueuedTurns((current) => mergeSnapshotQueuedTurns(response, current));
      const responseRuntime = runtimeOf(response);
      setPendingApproval(responseRuntime.pending_approval);
      setIsRunning(snapshotIsRunning(responseRuntime));
      if (!snapshotIsRunning(responseRuntime)) {
        setLiveRunId("");
        setLiveTimeline([]);
      }
      setSelectedDebugEventId(observabilityOf(response)?.debug_events[0]?.event_id ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Thread switch failed");
    }
  }

  function handleAppEvent(payload: AppEventPayload) {
    if (payload.kind === "session") {
      appendSessionEvent(payload.event);
      setLiveRunId(payload.event.run_id);
      if (payload.event.kind === "tool_approval_requested" || payload.event.kind === "run_suspended") {
        setPendingApproval(eventApprovalRequest(payload.event.payload));
        setIsRunning(false);
        return;
      }
      if (payload.event.kind === "tool_approval_resolved") {
        setPendingApproval(null);
        setIsRunning(true);
        return;
      }
      if (payload.event.kind === "run_completed" || payload.event.kind === "run_failed") {
        setIsRunning(false);
        setPendingApproval(null);
        setLiveRunId("");
        setLiveTimeline([]);
      }
      return;
    }
    setLiveRunId(payload.event.run_id);
    const eventType = payload.event.event_type;
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
      window.setTimeout(() => {
        void refreshSnapshot();
        void refreshSessions();
      }, 250);
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

  function mergeSessionEventLogs(current: SessionEvent[], snapshotEvents: SessionEvent[]) {
    const snapshotAssistantRuns = new Set(
      snapshotEvents
        .filter((event) => event.kind === "assistant_message" || event.kind === "assistant_tool_call")
        .map(sessionRunKey)
    );
    const snapshotReasoningRuns = new Set(
      snapshotEvents
        .filter((event) => event.kind === "reasoning_summary")
        .map(sessionRunKey)
    );
    const snapshotIds = new Set(snapshotEvents.map((event) => event.event_id));
    const liveEvents = current.filter((event) => {
      if (snapshotIds.has(event.event_id)) return false;
      if (event.kind === "assistant_delta" && snapshotAssistantRuns.has(sessionRunKey(event))) return false;
      if (event.kind === "thinking_delta" && snapshotReasoningRuns.has(sessionRunKey(event))) return false;
      return true;
    });
    return [...snapshotEvents, ...liveEvents];
  }

  function sessionRunKey(event: SessionEvent) {
    return `${event.thread_id}:${event.run_id}`;
  }

  async function refreshSnapshot() {
    try {
      const response = await sendCommand<SnapshotPayload>("snapshot", {});
      setSnapshot(response);
      if (response.session.session_id !== "new") {
        writeSessionIdToUrl(response.session.session_id, { replace: true });
      }
      setEvents((current) => mergeSessionEventLogs(current, response.events));
      setQueuedTurns((current) => mergeSnapshotQueuedTurns(response, current));
      const responseRuntime = runtimeOf(response);
      setPendingApproval(responseRuntime.pending_approval);
      setIsRunning(snapshotIsRunning(responseRuntime));
      if (!snapshotIsRunning(responseRuntime)) {
        setLiveRunId("");
        setLiveTimeline([]);
      }
      setSelectedDebugEventId(observabilityOf(response)?.debug_events[0]?.event_id ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Snapshot failed");
    }
  }

  async function submitMessage(event: FormEvent) {
    event.preventDefault();
    if (!input && composerImages.length === 0) return;
    if (input.startsWith("/") && composerImages.length === 0) {
      executeComposerCommand(input);
      return;
    }
    const content = input;
    const attachments = composerImages.map(({ mime_type, data }) => ({ mime_type, data }));
    setInput("");
    setComposerImages([]);
    if (!runtime.active_thread_accepts_user_turn) {
      await steerMessage(content, attachments);
      return;
    }
    if (isBlockedForApproval) {
      setError("Choose Approve or Reject before sending another message.");
      return;
    }
    if (isRunning) {
      await enqueueComposerTurn(content, attachments);
      return;
    }
    await startMessage(content, attachments);
  }

  async function startMessage(content: string, attachments: ImageAttachmentPayload[] = []) {
    stickToTranscriptEndRef.current = true;
    appendOptimisticUserMessage(content, attachments);
    setLiveRunId("");
    setLiveTimeline([]);
    setIsRunning(true);
    try {
      const response = await sendCommand<{ session_id?: string }>("send_message", { content, attachments });
      if (response.session_id && response.session_id !== "new") {
        writeSessionIdToUrl(response.session_id, { replace: true });
      }
    } catch (err) {
      setIsRunning(false);
      setError(err instanceof Error ? err.message : "Send failed");
    }
  }

  async function cancelRun() {
    try {
      await sendCommand("cancel", {});
      setIsRunning(false);
      setCancelArmed(false);
      setPendingApproval(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Cancel failed");
    }
  }

  function requestCancelRun() {
    if (!isRunning) return;
    if (!cancelArmed) {
      setCancelArmed(true);
      setNotice("Press Esc again to cancel the current response.");
      return;
    }
    setNotice(null);
    void cancelRun();
  }

  async function enqueueComposerTurn(content: string, attachments: ImageAttachmentPayload[]) {
    stickToTranscriptEndRef.current = true;
    appendOptimisticUserMessage(content, attachments);
    setQueuedTurns((turns) => [...turns, { id: nextRef("queued"), attachments, content }]);
    try {
      const response = await sendCommand<QueuedTurnsResponse>("enqueue_message", { content, attachments });
      setQueuedTurns(snapshotQueuedTurnsFromPayloads(response.queued_turns));
      await refreshSnapshot();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Queue failed");
    }
  }

  function appendOptimisticUserMessage(content: string, attachments: ImageAttachmentPayload[]) {
    setOptimisticTurns((turns) => [
      ...turns,
      {
        id: nextRef("optimistic-user"),
        images: attachments,
        role: "user",
        text: content,
        time: new Date().toISOString()
      }
    ]);
  }

  function updateTranscriptStickiness() {
    const transcriptElement = transcriptScrollRef.current;
    if (transcriptElement === null) return;
    const distanceFromEnd = transcriptElement.scrollHeight - transcriptElement.scrollTop - transcriptElement.clientHeight;
    stickToTranscriptEndRef.current = distanceFromEnd < 96;
  }

  function appendLiveTimelineEvent(payload: AppEventPayload) {
    if (payload.kind !== "agent") return;
    const item = liveTimelineItem(payload);
    if (item === null) return;
    setLiveTimeline((items) => {
      return [item, ...items.filter((existingItem) => existingItem.id !== item.id)].slice(0, 8);
    });
  }

  async function handleComposerPaste(event: ClipboardEvent<HTMLTextAreaElement>) {
    const imageItems = Array.from(event.clipboardData.items).filter((item) => item.type.startsWith("image/"));
    if (imageItems.length === 0) return;
    event.preventDefault();
    try {
      const images = await Promise.all(imageItems.map(readClipboardImage));
      setComposerImages((items) => [...items, ...images]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Image paste failed");
    }
  }

  async function cancelQueuedTurn(index: number) {
    setQueuedTurns((turns) => turns.filter((_, turnIndex) => turnIndex !== index));
    try {
      const response = await sendCommand<QueuedTurnsResponse>("cancel_queued_message", { index });
      setQueuedTurns(snapshotQueuedTurnsFromPayloads(response.queued_turns));
      await refreshSnapshot();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Cancel queued message failed");
    }
  }

  async function steerQueuedTurn(index: number) {
    const queuedTurn = queuedTurns[index];
    if (!queuedTurn) return;
    setQueuedTurns((turns) => turns.filter((_, turnIndex) => turnIndex !== index));
    if (isRunning) {
      await startSteeredQueuedMessage(index, queuedTurn);
      return;
    }
    await startQueuedMessage(index, queuedTurn);
  }

  async function startQueuedMessage(index: number, queuedTurn: QueuedTurn) {
    stickToTranscriptEndRef.current = true;
    setLiveRunId("");
    setLiveTimeline([]);
    setIsRunning(true);
    try {
      await sendCommand("start_queued_message", { index });
    } catch (err) {
      setIsRunning(false);
      setError(err instanceof Error ? err.message : "Start queued message failed");
      setQueuedTurns((turns) => [queuedTurn, ...turns]);
    }
  }

  async function startSteeredQueuedMessage(index: number, queuedTurn: QueuedTurn) {
    stickToTranscriptEndRef.current = true;
    setLiveRunId("");
    setLiveTimeline([]);
    setIsRunning(true);
    try {
      await sendCommand("steer_queued_message", { index });
    } catch (err) {
      setIsRunning(false);
      setError(err instanceof Error ? err.message : "Steer queued message failed");
      setQueuedTurns((turns) => [queuedTurn, ...turns]);
    }
  }

  async function steerMessage(content: string, attachments: ImageAttachmentPayload[] = []) {
    stickToTranscriptEndRef.current = true;
    appendOptimisticUserMessage(content, attachments);
    setLiveRunId("");
    setLiveTimeline([]);
    setIsRunning(true);
    try {
      await sendCommand("steer_message", { content, attachments });
      await refreshSnapshot();
    } catch (err) {
      setIsRunning(false);
      setError(err instanceof Error ? err.message : "Steer failed");
    }
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
    if (event.key === "Escape") {
      if (isRunning) {
        event.preventDefault();
        event.stopPropagation();
        requestCancelRun();
        return;
      }
      setCancelArmed(false);
    }
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
    setSelectedReferenceIndex(0);
    if (item.kind === "file") {
      revealWorkspace("files", true);
      void openProjectFile(filePathFromReference(item.value));
    }
    window.requestAnimationFrame(() => composerRef.current?.focus());
  }

  async function openProjectFile(path: string, options: { edit?: boolean } = {}) {
    revealWorkspace("files", true);
    setFileLoading(true);
    setError(null);
    try {
      const payload = await queryClient.fetchQuery({
        queryKey: GUI_QUERY_KEYS.file(path),
        queryFn: async () => {
          const response = await apiClient.readFileApiFilesGet({ path });
          return filePayloadFromApi(response.file);
        },
        staleTime: 2000
      });
      setOpenFile(payload);
      setFileDraft(payload.content);
      setFileEditMode(options.edit === true);
      setSelectedArtifactId(null);
    } catch (err) {
      setError(apiFailureMessage("Open file", err));
    } finally {
      setFileLoading(false);
    }
  }

  async function saveOpenFile() {
    if (openFile === null) return;
    try {
      await saveFileMutation.mutateAsync({ path: openFile.path, content: fileDraft });
    } catch (err) {
      setError(apiFailureMessage("Save file", err));
    }
  }

  function inspectArtifact(artifact: ArtifactItem) {
    revealWorkspace("files", true);
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

  function toggleInspectorGroup(groupId: InspectorGroupId) {
    setOpenInspectorGroups((current) => ({
      ...current,
      [groupId]: !current[groupId]
    }));
  }

  async function saveIdea(content: string) {
    try {
      const payload = await captureIdeaMutation.mutateAsync(content);
      setNotice(`Saved idea ${payload.idea.index}.`);
      setSelectedIdeaIndex(payload.idea.index);
      setIdeaDraft(payload.idea.content);
      return true;
    } catch (err) {
      setError(apiFailureMessage("Save idea", err));
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
      const payload = await updateIdeaMutation.mutateAsync({ index: selectedIdea.index, content: ideaDraft });
      setNotice(`Updated idea ${payload.idea.index}.`);
    } catch (err) {
      setError(apiFailureMessage("Update idea", err));
    }
  }

  async function deleteSelectedIdea() {
    if (!selectedIdea) return;
    if (!window.confirm(`Delete idea ${selectedIdea.index}?`)) return;
    try {
      const payload = await deleteIdeaMutation.mutateAsync(selectedIdea.index);
      setNotice(`Deleted idea ${payload.idea.index}.`);
    } catch (err) {
      setError(apiFailureMessage("Delete idea", err));
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
      setWorkspaceMode("settings");
      void refreshSettings();
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
        void steerMessage(commandArg);
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
            {visibleSessionGroups.map((group) => (
              <section className="project-session-group" key={group.project_id}>
                <ProjectGroupHeader count={group.items.length} label={group.project_name} />
                {group.items.map((session) => {
                  const active = snapshot?.session.session_id === session.session_id;
                  return (
                    <div className={`session-row ${active ? "active online" : ""}`} key={session.session_id}>
                      <button className="session-open" onClick={() => connectSession(session.session_id)} title={`Open ${session.title}`}>
                        <span className="session-status" />
                        <div>
                          <strong>{session.title}</strong>
                          <span>{formatShortDate(session.updated_at)}</span>
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
              </section>
            ))}
          </div>
        </section>

      </aside>

      <section className="workspace">
        <header className="topbar">
          <div className="workspace-title">
            <StatusDot state={connectionState} />
            <div>
              <strong>{snapshot?.session.title || (hasActiveSession ? "New AceAI session" : "AceAI")}</strong>
              <span>
                {hasActiveSession ? connectionLabel : "Web GUI"}
                {snapshot ? ` / ${snapshot.session.project_name}` : !hasActiveSession ? ` / ${launchProjectName}` : ""}
              </span>
            </div>
          </div>
          {hasActiveSession ? <div className="topbar-center" aria-label="Workspace mode">
            <div className="mode-switch">
              <button
                className={`mode-pill ${workspaceMode === "chat" ? "active" : ""}`}
                onClick={() => setWorkspaceMode("chat")}
                title="Show chat view"
              >
                <MessageSquare size={13} />
                Chat
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
            <div className="topbar-usage" title={usageTitle} aria-label={usageTitle}>
              <Database size={13} />
              <span>{formatCompactNumber(observableUsage?.total_tokens ?? 0)} tokens</span>
              <span>{formatUsd(observableUsage?.total_cost_usd ?? 0)}</span>
            </div>
          </div> : <div className="topbar-center topbar-center-empty" />}
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
            {connected && isRunning ? (
              <button onClick={requestCancelRun} title={cancelArmed ? "Confirm cancel active run" : "Press twice to cancel active run"}>
                <CircleStop size={16} />
              </button>
            ) : (
              <button
                className={workspaceMode === "settings" ? "topbar-action-active" : ""}
                onClick={() => {
                  setWorkspaceMode("settings");
                  void refreshSettings();
                }}
                title="Settings"
              >
                <SlidersHorizontal size={16} />
              </button>
            )}
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
          className={`content-grid ${workspaceMode === "settings" ? "settings-mode" : ""} ${workspaceOpen ? "workspace-open" : "workspace-closed"} ${!hasActiveSession ? "no-session-mode" : ""}`}
          style={{ "--inspector-width": `${inspectorWidth}px` } as CSSProperties}
        >
          <section className="conversation-pane" aria-label="Transcript">
            {workspaceMode !== "settings" && hasActiveSession ? <div className="pane-header">
              <div>
                <span>Conversation</span>
                <strong>{isRunning ? "Streaming" : connected ? "Ready" : "Offline"}</strong>
              </div>
              <div className="run-chip">
                {isRunning ? <Sparkles size={14} /> : <Clock3 size={14} />}
                {isRunning ? "running" : latestRun ? "completed" : "idle"}
              </div>
            </div> : null}

            {showLaunchScreen ? (
              <LaunchScreen
                latestSessionTitle={latestSession?.title}
                onBrowseSessions={() => setWorkspaceMode("sessions")}
                onNewChat={createSession}
                onOpenLatest={openLatestSession}
                projectName={launchProjectName}
                sessionCount={sessions.length}
              />
            ) : null}

            {workspaceMode === "chat" && !showLaunchScreen ? <div className="transcript" onScroll={updateTranscriptStickiness} ref={transcriptScrollRef}>
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
                    {item.role === "assistant" && item.workHistory ? (
                      <MessageWorkHistory history={item.workHistory} onOpenFile={(path) => void openProjectFile(path)} />
                    ) : null}
                    {item.role === "assistant" && item.retries && item.retries.length > 0 ? (
                      <MessageRetryNotices items={item.retries} />
                    ) : null}
                    {item.role === "assistant" && item.reasoning ? (
                      <MessageReasoning text={item.reasoning} />
                    ) : null}
                    <MarkdownText text={item.text} onOpenFile={(path) => void openProjectFile(path)} />
                    {item.images && item.images.length > 0 ? <MessageImages images={item.images} /> : null}
                  </article>
                ))
              )}
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
                  {visibleSessionGroups.map((group) => (
                    <section className="project-workspace-group" key={group.project_id}>
                      <ProjectGroupHeader count={group.items.length} label={group.project_name} />
                      {group.items.map((session) => {
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
                                <span>{formatShortDate(session.updated_at)}</span>
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
                    </section>
                  ))}
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
                      {ideaGroups.map((group) => (
                        <section className="project-idea-group" key={group.project_id}>
                          <ProjectGroupHeader count={group.items.length} label={group.project_name} />
                          {group.items.map((idea) => (
                            <button
                              className={idea.index === selectedIdea?.index ? "active" : ""}
                              key={idea.idea_id}
                              onClick={() => selectIdea(idea)}
                            >
                              <span>{ideaTitle(idea.content)}</span>
                              <code>@idea:{idea.index}</code>
                            </button>
                          ))}
                        </section>
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

            {workspaceMode === "settings" ? (
              <SettingsWorkspace
                apiKey={settingsApiKey}
                current={settings}
                draft={settingsDraft}
                saving={settingsSaving}
                onApiKeyChange={setSettingsApiKey}
                onReload={() => void refreshSettings()}
                onSave={() => void saveSettings()}
                onUpdate={updateSettingsDraft}
                onUpdateTool={updateToolDraft}
              />
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
                      {turn.attachments.length > 0 ? <MessageImages images={turn.attachments} compact /> : null}
                      <button type="button" onClick={() => void steerQueuedTurn(index)}>
                        Send now
                      </button>
                      <button type="button" onClick={() => void cancelQueuedTurn(index)}>
                        Cancel
                      </button>
                    </div>
                  ))}
                </div>
              </section>
            ) : null}

            {workspaceMode !== "settings" && hasActiveSession ? <form className="composer" onSubmit={submitMessage}>
              <div className="composer-card">
                <div className="composer-input-area">
                <textarea
                  ref={composerRef}
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={handleComposerKeyDown}
                  placeholder={connected ? composerPlaceholder(isRunning, isBlockedForApproval) : "Choose a session first"}
                      disabled={!connected}
                  onPaste={(event) => void handleComposerPaste(event)}
                />
                {composerImages.length > 0 ? (
                  <div className="composer-attachments" aria-label="Image attachments">
                    {composerImages.map((image) => (
                      <div className="composer-attachment" key={image.id}>
                        <img alt="Pasted attachment" src={imageDataUrl(image)} />
                        <button
                          type="button"
                          onClick={() => setComposerImages((items) => items.filter((item) => item.id !== image.id))}
                          aria-label="Remove image"
                          title="Remove image"
                        >
                          <CircleStop size={13} />
                        </button>
                      </div>
                    ))}
                  </div>
                ) : null}
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
                      {cancelArmed ? "esc again to cancel" : composerStatus}
                    </span>
                    <button type="button" className="composer-model-button" title="Selected model">
                      <Sparkles size={15} />
                      {formatModelName(runtime.selected_model)}
                      <span>{formatReasoningLevel(runtime.reasoning_level)}</span>
                      <ChevronDown size={15} />
                    </button>
                    <button type="button" className="composer-icon-button" title="Voice input" aria-label="Voice input" disabled>
                      <Mic size={16} />
                    </button>
                    <button type="submit" className="composer-send-button" disabled={!connected || (!input && composerImages.length === 0)} title="Send">
                      <Send size={17} />
                    </button>
                  </div>
                </div>
              </div>
            </form> : null}
          </section>

          {workspaceMode !== "settings" && hasActiveSession && workspaceOpen ? <div
            aria-label="Resize workspace"
            className="split-resizer"
            onMouseDown={startInspectorResize}
            role="separator"
            title="Drag to resize workspace"
          /> : null}

          {workspaceMode !== "settings" && hasActiveSession ? <aside
            className={`inspector ${workspaceOpen ? "open" : "collapsed"} ${workspaceTab === "files" && hasWorkspaceObject ? "object-open" : ""}`}
            aria-label="Workspace"
          >
            {!workspaceOpen ? (
              <div className="workspace-rail" aria-label="Open workspace">
                <button type="button" onClick={() => openWorkspace("files")} title="Files" aria-label="Open files">
                  <FileText size={18} />
                </button>
                <button type="button" onClick={() => openWorkspace("agents")} title="Agents" aria-label="Open agents">
                  <GitBranch size={18} />
                </button>
                <button type="button" onClick={() => openWorkspace("activity")} title="Activity" aria-label="Open activity">
                  <Sparkles size={18} />
                </button>
                <button type="button" onClick={() => openWorkspace("run")} title={usageTitle} aria-label="Open run details">
                  <PanelRight size={18} />
                </button>
              </div>
            ) : (
              <>
                <div className="inspector-header">
                  <div>
                    <span>Workspace</span>
                    <strong>{workspaceTabTitle(workspaceTab)}</strong>
                  </div>
                  <button type="button" className="tiny-icon-button" onClick={closeWorkspace} title="Collapse workspace" aria-label="Collapse workspace">
                    <PanelRight size={14} />
                  </button>
                </div>
                <div className="workspace-tabs" role="tablist" aria-label="Workspace views">
                  <button type="button" className={workspaceTab === "files" ? "active" : ""} onClick={() => openWorkspace("files")}>
                    <FileText size={14} />
                    Files
                  </button>
                  <button type="button" className={workspaceTab === "agents" ? "active" : ""} onClick={() => openWorkspace("agents")}>
                    <GitBranch size={14} />
                    Agents
                  </button>
                  <button type="button" className={workspaceTab === "activity" ? "active" : ""} onClick={() => openWorkspace("activity")}>
                    <Sparkles size={14} />
                    Activity
                  </button>
                  <button type="button" className={workspaceTab === "run" ? "active" : ""} onClick={() => openWorkspace("run")}>
                    <PanelRight size={14} />
                    Run
                  </button>
                </div>

                {workspaceTab === "files" ? (
                  <section className="inspector-section object-inspector" aria-label="Workspace object">
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
                ) : null}

                {workspaceTab === "run" ? (
                  <div className="work-history" aria-label="Run details">
                    <InspectorGroup
                      icon={<PanelRight size={15} />}
                      open={openInspectorGroups.run}
                      onToggle={() => toggleInspectorGroup("run")}
                      summary={pendingApproval ? "approval needed" : isRunning ? "active run" : latestRun ? "settled" : "waiting"}
                      title="Run"
                    >
                      <section className="inspector-section" ref={statsRef}>
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
                    </InspectorGroup>
                  </div>
                ) : null}

                {workspaceTab === "agents" ? (
                  <div className="work-history" aria-label="Agents and context">
                    <InspectorGroup
                      icon={<GitBranch size={15} />}
                      open={openInspectorGroups.context}
                      onToggle={() => toggleInspectorGroup("context")}
                      summary={`${snapshot?.threads.length ?? 0} threads / ${ideas.length} ideas`}
                      title="Agents"
                    >
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
                              {ideaGroups.map((group) => (
                                <section className="project-idea-group" key={group.project_id}>
                                  <ProjectGroupHeader count={group.items.length} label={group.project_name} />
                                  {group.items.map((idea) => (
                                    <button
                                      className={idea.index === selectedIdea?.index ? "active" : ""}
                                      key={idea.idea_id}
                                      onClick={() => selectIdea(idea)}
                                    >
                                      <span>{ideaTitle(idea.content)}</span>
                                      <code>@idea:{idea.index}</code>
                                    </button>
                                  ))}
                                </section>
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
                    </InspectorGroup>
                  </div>
                ) : null}

                {workspaceTab === "activity" ? (
                  <div className="work-history" aria-label="Activity">
                <InspectorGroup
                  icon={<Sparkles size={15} />}
                  open={openInspectorGroups.work}
                  onToggle={() => toggleInspectorGroup("work")}
                  summary={`${timeline.length} steps / ${observableToolCalls.length} tools`}
                  title="Work"
                >
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
            </InspectorGroup>

            <InspectorGroup
              icon={<Activity size={15} />}
              open={openInspectorGroups.signals}
              onToggle={() => toggleInspectorGroup("signals")}
              summary={`${observableEventKinds.length} event kinds / ${observableDebugEvents.length} debug`}
              title="Signals"
            >
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
            </InspectorGroup>
              </div>
            ) : null}
              </>
            )}
          </aside> : null}
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

function LaunchScreen({
  latestSessionTitle,
  onBrowseSessions,
  onNewChat,
  onOpenLatest,
  projectName,
  sessionCount
}: {
  latestSessionTitle?: string;
  onBrowseSessions: () => void;
  onNewChat: () => void;
  onOpenLatest: () => void;
  projectName: string;
  sessionCount: number;
}) {
  const hasSessions = sessionCount > 0;
  return (
    <div className="launch-screen">
      <div className="launch-mark">
        <Bot size={32} />
      </div>
      <div className="launch-copy">
        <span>AceAI Web GUI</span>
        <h1>Ready when you are.</h1>
        <p>Project: {projectName}</p>
      </div>
      <div className="launch-actions">
        <button type="button" className="primary" onClick={onNewChat}>
          <Plus size={15} />
          New Chat
        </button>
        <button type="button" onClick={onBrowseSessions}>
          <Layers size={15} />
          Sessions
        </button>
        <button type="button" onClick={onOpenLatest} disabled={!hasSessions}>
          <Clock3 size={15} />
          {latestSessionTitle ? `Resume ${latestSessionTitle}` : "Resume latest"}
        </button>
      </div>
      <div className="launch-shortcuts" aria-label="Shortcuts">
        <div><kbd>Enter</kbd><span>ask</span></div>
        <div><kbd>S</kbd><span>sessions</span></div>
        <div><kbd>/</kbd><span>commands</span></div>
        <div><kbd>Esc</kbd><span>cancel</span></div>
      </div>
    </div>
  );
}

function workspaceTabTitle(tab: WorkspaceTab) {
  if (tab === "files") return "Files";
  if (tab === "agents") return "Agents";
  if (tab === "activity") return "Activity";
  return "Run";
}

function ProjectGroupHeader({ count, label }: { count: number; label: string }) {
  return (
    <div className="project-group-header">
      <span>{label}</span>
      <code>{count}</code>
    </div>
  );
}

function InspectorGroup({
  children,
  icon,
  onToggle,
  open,
  summary,
  title
}: {
  children: ReactNode;
  icon: ReactNode;
  onToggle: () => void;
  open: boolean;
  summary: string;
  title: string;
}) {
  return (
    <section className={`inspector-group ${open ? "open" : ""}`}>
      <button className="inspector-group-toggle" type="button" onClick={onToggle}>
        <span className="inspector-group-icon">{icon}</span>
        <span className="inspector-group-copy">
          <strong>{title}</strong>
          <small>{summary}</small>
        </span>
        {open ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
      </button>
      {open ? <div className="inspector-group-body">{children}</div> : null}
    </section>
  );
}

function StatusDot({ state }: { state: ConnectionState }) {
  if (state === "connected") return <Check className="status-icon good" size={17} />;
  if (state === "error") return <AlertTriangle className="status-icon bad" size={17} />;
  if (state === "closed") return <WifiOff className="status-icon muted-icon" size={17} />;
  return <span className={`status-dot ${state}`} />;
}

function MessageWorkHistory({ history, onOpenFile }: { history: RunWorkHistory; onOpenFile?: (path: string) => void }) {
  const listRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!history.isRunning) return;
    const list = listRef.current;
    if (list === null) return;
    list.scrollTop = list.scrollHeight;
  }, [history.isRunning, history.items.length]);
  if (history.items.length === 0) return null;
  return (
    <details className="message-work-history" open={history.isRunning}>
      <summary className="message-work-history-title">
        <span>{history.headline}</span>
        <small>{history.isRunning ? "live" : "details"}</small>
      </summary>
      <div className="transcript-history-list" ref={listRef}>
        {history.items.map((item) => (
          <div className={`transcript-history-row ${item.tone}`} key={item.id}>
            <span />
            <div>
              <strong>
                <FileLinkedText text={item.title} onOpenFile={onOpenFile} />
              </strong>
            </div>
          </div>
        ))}
      </div>
    </details>
  );
}

function MessageRetryNotices({ items }: { items: TimelineItem[] }) {
  return (
    <div className="message-retry-notices" aria-label="Model retry notices">
      {items.map((item) => (
        <div className="message-retry-notice" key={item.id}>
          <AlertTriangle size={14} />
          <span>{item.detail}</span>
        </div>
      ))}
    </div>
  );
}

function FileLinkedText({ text, onOpenFile }: { text: string; onOpenFile?: (path: string) => void }) {
  const match = filePathMatchFromText(text);
  if (match === null || onOpenFile === undefined) return <>{text}</>;
  return (
    <>
      {text.slice(0, match.start)}
      <button
        type="button"
        className="file-reference-button"
        onClick={() => onOpenFile(match.path)}
        title={`Preview ${match.path}`}
      >
        {match.path}
      </button>
      {text.slice(match.end)}
    </>
  );
}

function MessageReasoning({ text }: { text: string }) {
  return (
    <section className="message-reasoning" aria-label="Reasoning">
      <strong>Reasoning</strong>
      <MarkdownText text={text} compact />
    </section>
  );
}

function MarkdownText({ text, compact = false, onOpenFile }: { text: string; compact?: boolean; onOpenFile?: (path: string) => void }) {
  return (
    <div className={compact ? "markdown markdown-compact" : "markdown"}>
      <ReactMarkdown
        components={{
          code({ children, className, ...props }) {
            const rawValue = String(children);
            const value = rawValue.trim();
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
            const highlightedHtml = highlightedCodeHtml(rawValue, className);
            if (highlightedHtml !== null) {
              const codeClassName = ["hljs", className].filter(Boolean).join(" ");
              return <code {...props} className={codeClassName} dangerouslySetInnerHTML={{ __html: highlightedHtml }} />;
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
            const linkLabel = typeof href === "string" ? externalLinkLabel(href, children) : null;
            if (linkLabel !== null) {
              return (
                <a
                  className="external-link-card"
                  href={href}
                  rel="noreferrer"
                  target="_blank"
                  title={href}
                  {...props}
                >
                  <ExternalLink size={11} aria-hidden="true" />
                  <span className="external-link-host">{linkLabel.host}</span>
                  {linkLabel.path !== "" ? <span className="external-link-path">{linkLabel.path}</span> : null}
                </a>
              );
            }
            return <a href={href} rel="noreferrer" target="_blank" {...props}>{children}</a>;
          }
        }}
        remarkPlugins={MARKDOWN_PLUGINS}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}

function externalLinkLabel(href: string, children: ReactNode) {
  const childText = singleTextChild(children);
  if (childText !== href) return null;
  const match = href.match(/^https?:\/\/([^/?#]+)([^?#]*)?/);
  if (match === null) return null;
  return {
    host: match[1],
    path: match[2] ?? ""
  };
}

function singleTextChild(children: ReactNode) {
  if (typeof children === "string") return children;
  if (!Array.isArray(children)) return null;
  let value = "";
  for (const child of children) {
    if (typeof child !== "string") return null;
    value += child;
  }
  return value;
}

function MessageImages({ compact = false, images }: { compact?: boolean; images: ImageAttachmentPayload[] }) {
  return (
    <div className={compact ? "message-images compact" : "message-images"}>
      {images.map((image, index) => (
        <img
          alt={`Attached image ${index + 1}`}
          key={`${image.mime_type}-${index}`}
          src={imageDataUrl(image)}
        />
      ))}
    </div>
  );
}

function highlightedCodeHtml(value: string, className: string | undefined) {
  const code = value.endsWith("\n") ? value.slice(0, -1) : value;
  const language = markdownCodeLanguage(className);
  if (language !== null && hljs.getLanguage(language)) {
    return hljs.highlight(code, { language }).value;
  }
  if (code.includes("\n")) {
    return hljs.highlightAuto(code).value;
  }
  return null;
}

function markdownCodeLanguage(className: string | undefined) {
  const match = className?.match(/(?:^|\s)language-([^\s]+)/);
  return match?.[1] ?? null;
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

function buildTranscript(
  events: SessionEvent[],
  runWorkHistories: Map<string, RunWorkHistory>,
  runReasoning: Map<string, string>,
  runRetries: Map<string, TimelineItem[]>,
): TranscriptItem[] {
  const deltaDrafts = assistantDeltaDrafts(events);
  const finalizedRuns = new Set(
    events
      .filter((event) => event.kind === "assistant_message" || event.kind === "run_completed")
      .map((event) => event.run_id)
  );
  const insertedDraftRuns = new Set<string>();
  const items: TranscriptItem[] = [];
  for (const event of events) {
    const content = typeof event.payload.content === "string" ? event.payload.content : "";
    const images = imagesFromPayload(event.payload);
    if (event.kind === "user_message") {
      if (content || images.length > 0) {
        items.push({ id: event.event_id, images, role: "user", text: content, time: event.created_at });
      }
      const draft = assistantDraftItem(
        event.run_id,
        deltaDrafts,
        runWorkHistories,
        runReasoning,
        runRetries,
        finalizedRuns,
      );
      if (draft !== null) {
        items.push(draft);
        insertedDraftRuns.add(event.run_id);
      }
      continue;
    }
    if (event.kind === "assistant_message" || event.kind === "run_completed") {
      if (!content) continue;
      items.push({
        id: event.event_id,
        reasoning: runReasoning.get(event.run_id),
        retries: runRetries.get(event.run_id),
        role: "assistant",
        runId: event.run_id,
        text: content,
        time: event.created_at,
        workHistory: runWorkHistories.get(event.run_id)
      });
      continue;
    }
    if (event.kind === "run_failed" || event.kind === "error") {
      items.push({ id: event.event_id, role: "system", text: content || "Run failed", time: event.created_at });
    }
  }
  for (const [runId, workHistory] of runWorkHistories.entries()) {
    if (insertedDraftRuns.has(runId)) continue;
    const draft = assistantDraftItem(
      runId,
      deltaDrafts,
      runWorkHistories,
      runReasoning,
      runRetries,
      finalizedRuns,
    );
    if (draft !== null) items.push(draft);
  }
  return dedupeTranscript(items);
}

function assistantDraftItem(
  runId: string,
  deltaDrafts: Map<string, { text: string; time: string }>,
  runWorkHistories: Map<string, RunWorkHistory>,
  runReasoning: Map<string, string>,
  runRetries: Map<string, TimelineItem[]>,
  finalizedRuns: Set<string>,
): TranscriptItem | null {
  if (runId === "" || finalizedRuns.has(runId)) return null;
  const draft = deltaDrafts.get(runId);
  const workHistory = runWorkHistories.get(runId);
  const reasoning = runReasoning.get(runId);
  const retries = runRetries.get(runId);
  if (draft === undefined && !workHistory?.isRunning && (reasoning === undefined || reasoning === "") && retries === undefined) return null;
  return {
    id: `${runId}-assistant-draft`,
    reasoning,
    retries,
    role: "assistant",
    runId,
    text: draft?.text ?? "",
    time: draft?.time ?? new Date().toISOString(),
    workHistory
  };
}

function assistantDeltaDrafts(events: SessionEvent[]) {
  const drafts = new Map<string, { text: string; time: string }>();
  for (const event of events) {
    if (event.kind !== "assistant_delta" || event.run_id === "") continue;
    const content = typeof event.payload.content === "string" ? event.payload.content : "";
    if (content === "") continue;
    const draft = drafts.get(event.run_id);
    drafts.set(event.run_id, {
      text: `${draft?.text ?? ""}${content}`,
      time: event.created_at
    });
  }
  return drafts;
}

function imagesFromPayload(payload: Record<string, unknown>): ImageAttachmentPayload[] {
  const images = payload.images;
  if (!Array.isArray(images)) return [];
  return images.flatMap((image) => {
    if (!isRecord(image)) return [];
    const mimeType = image.mime_type;
    const data = image.data;
    if (typeof mimeType !== "string" || typeof data !== "string") return [];
    return [{ mime_type: mimeType, data }];
  });
}

function mergeOptimisticTranscript(
  persisted: TranscriptItem[],
  optimistic: TranscriptItem[],
  liveRunId: string,
  liveAssistantWorkHistory: RunWorkHistory | undefined,
) {
  const unmatched = optimistic.filter(
    (turn) =>
      !persisted.some((item) => transcriptItemsMatch(item, turn)),
  );
  const merged = [...persisted, ...unmatched];
  const hasLiveWorkHistory = liveAssistantWorkHistory !== undefined && liveAssistantWorkHistory.items.length > 0;
  const liveAssistantRunId = liveRunId || liveAssistantWorkHistory?.runId || "";
  if (liveAssistantRunId !== "" && liveAssistantWorkHistory !== undefined) {
    const liveIndex = merged.findIndex((item) => item.role === "assistant" && item.runId === liveAssistantRunId);
    if (liveIndex >= 0) {
      merged[liveIndex] = {
        ...merged[liveIndex],
        workHistory: liveAssistantWorkHistory ?? merged[liveIndex].workHistory
      };
      return merged;
    }
  }
  if (
    hasLiveWorkHistory &&
    !persisted.some((item) => (
      item.role === "assistant" &&
      liveAssistantRunId !== "" &&
      item.runId === liveAssistantRunId
    ))
  ) {
    merged.push({
      id: "live-assistant",
      role: "assistant",
      runId: liveAssistantRunId || undefined,
      text: "",
      time: new Date().toISOString(),
      workHistory: liveAssistantWorkHistory
    });
  }
  return merged;
}

function dedupeTranscript(items: TranscriptItem[]) {
  const deduped: TranscriptItem[] = [];
  for (const item of items) {
    const previous = deduped[deduped.length - 1];
    if (previous && transcriptItemsMatch(previous, item)) {
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

function buildRunReasoning(events: SessionEvent[]) {
  const reasoningByRun = new Map<string, string>();
  for (const event of events) {
    if (event.kind !== "thinking_delta" && event.kind !== "reasoning_summary") continue;
    const content = stringPayload(event.payload, "content");
    if (content === null || content === "") continue;
    reasoningByRun.set(event.run_id, `${reasoningByRun.get(event.run_id) ?? ""}${content}`);
  }
  return reasoningByRun;
}

function buildRunRetries(events: SessionEvent[]) {
  const retriesByRun = new Map<string, TimelineItem[]>();
  for (const event of events) {
    if (event.kind !== "llm_retrying" || event.run_id === "") continue;
    const content = stringPayload(event.payload, "content") ?? "Retrying model request";
    const retries = retriesByRun.get(event.run_id) ?? [];
    retries.push({
      content,
      detail: content,
      id: event.event_id,
      kind: event.kind,
      runId: event.run_id,
      title: "Retrying model",
      tone: "bad"
    });
    retriesByRun.set(event.run_id, retries);
  }
  return retriesByRun;
}

function buildRunWorkHistories(
  events: SessionEvent[],
  liveItems: TimelineItem[],
  activeRunId: string,
  isRunning: boolean,
) {
  const eventsByRun = new Map<string, SessionEvent[]>();
  for (const event of events) {
    if (!event.run_id) continue;
    const runEvents = eventsByRun.get(event.run_id) ?? [];
    runEvents.push(event);
    eventsByRun.set(event.run_id, runEvents);
  }
  for (const item of liveItems) {
    if (!item.runId || eventsByRun.has(item.runId)) continue;
    eventsByRun.set(item.runId, []);
  }

  const histories = new Map<string, RunWorkHistory>();
  for (const [runId, runEvents] of eventsByRun.entries()) {
    const liveRunItems = liveItems
      .filter((item) => item.runId === runId)
      .filter((item) => meaningfulLiveWorkHistoryKinds.has(item.kind ?? ""))
      .slice()
      .reverse();
    const toolCallItems = buildToolCallItems(runId, runEvents);
    const eventItems = runEvents.flatMap(workHistoryItemFromEvent);
    const runIsRunning = isRunning && runId === activeRunId;
    const items = uniqueTimelineItems([...toolCallItems, ...eventItems, ...liveRunItems]);
    const durationLabel = workHistoryDuration(runEvents, runId, runIsRunning);
    histories.set(runId, {
      durationLabel,
      headline: durationLabel,
      isRunning: runIsRunning,
      items,
      runId
    });
  }
  return histories;
}

function uniqueTimelineItems(items: TimelineItem[]) {
  const seen = new Set<string>();
  const uniqueItems: TimelineItem[] = [];
  for (const item of items) {
    if (seen.has(item.id)) continue;
    seen.add(item.id);
    uniqueItems.push(item);
  }
  return uniqueItems;
}

function workHistoryItemFromEvent(event: SessionEvent): TimelineItem[] {
  if (!workHistoryEventKinds.has(event.kind)) return [];
  const content = workHistoryEventContent(event);
  return [{
    content: content ?? undefined,
    id: event.event_id,
    kind: event.kind,
    runId: event.run_id,
    title: workHistoryEventTitle(event),
    detail: workHistoryEventDetail(event),
    tone: trajectoryTone(event.kind)
  }];
}

const workHistoryEventKinds = new Set([
  "tool_approval_requested",
  "tool_approval_resolved",
  "step_failed",
  "run_suspended",
  "run_failed",
  "error"
]);

const meaningfulLiveWorkHistoryKinds = new Set([
  "agent.tool.started",
  "agent.tool.approval_requested",
  "agent.run.suspended",
  "agent.tool.completed",
  "agent.tool.failed",
  "agent.run.failed"
]);

function workHistoryEventTitle(event: SessionEvent) {
  if (event.kind === "llm_retrying") return "Retrying model";
  if (event.kind === "tool_approval_requested") return "Approval requested";
  if (event.kind === "tool_approval_resolved") return "Approval resolved";
  if (event.kind === "step_failed") return "Step failed";
  if (event.kind === "run_suspended") return "Run suspended";
  if (event.kind === "run_failed" || event.kind === "error") return "Run failed";
  return event.kind;
}

function workHistoryEventDetail(event: SessionEvent) {
  return workHistoryEventContent(event) ?? formatTime(event.created_at) ?? event.kind;
}

function workHistoryEventContent(event: SessionEvent) {
  const content = event.payload.content;
  if (typeof content === "string" && content !== "") return content;
  const error = event.payload.error;
  if (typeof error === "string" && error !== "") return error;
  const finalAnswer = event.payload.final_answer;
  if (typeof finalAnswer === "string" && finalAnswer !== "") return _shortText(finalAnswer, 120);
  const toolName = event.payload.tool_name;
  if (typeof toolName === "string" && toolName !== "") return toolName;
  return null;
}

function workHistoryToolName(event: SessionEvent) {
  const toolName = event.payload.tool_name;
  if (typeof toolName === "string" && toolName !== "") return readableToolName(toolName);
  return "Tool";
}

function buildToolCallItems(runId: string, events: SessionEvent[]) {
  const toolEvents = events.filter((event) => (
    event.kind === "tool_started" ||
    event.kind === "tool_result" ||
    event.kind === "tool_completed" ||
    event.kind === "tool_failed"
  ));
  if (toolEvents.length === 0) return [];

  const items = new Map<string, TimelineItem>();
  for (const event of toolEvents) {
    const callId = stringPayload(event.payload, "tool_call_id");
    if (callId === null) continue;
    const existing = items.get(callId);
    const base = existing ?? toolCallItemFromEvent(runId, event, callId);
    if (event.kind === "tool_failed" || toolResultHasError(event)) {
      const summary = toolResultSummary(event, true);
      items.set(callId, {
        ...base,
        detail: summary ?? "Failed",
        tone: "bad"
      });
      continue;
    }
    if (event.kind === "tool_result" || event.kind === "tool_completed") {
      const summary = toolResultSummary(event, false);
      items.set(callId, {
        ...base,
        detail: summary ?? "Completed",
        tone: "good"
      });
      continue;
    }
    items.set(callId, base);
  }
  return Array.from(items.values());
}

function toolCallItemFromEvent(runId: string, event: SessionEvent, callId: string): TimelineItem {
  const toolName = stringPayload(event.payload, "tool_name") ?? "tool";
  const title = toolCallTitle(toolName, toolArguments(event));
  return {
    content: title,
    id: `${runId}-tool-${callId}`,
    kind: "tool_call",
    runId,
    title,
    detail: "Running",
    tone: "live"
  };
}

function toolArguments(event: SessionEvent) {
  const toolArguments = stringPayload(event.payload, "tool_arguments");
  const direct = jsonObjectFromString(toolArguments);
  if (direct !== null) return direct;
  const toolCall = event.payload.tool_call;
  if (!isRecord(toolCall)) return null;
  const argumentsValue = toolCall.arguments;
  if (typeof argumentsValue !== "string") return null;
  return jsonObjectFromString(argumentsValue);
}

function toolResultHasError(event: SessionEvent) {
  const toolResult = event.payload.tool_result;
  if (isRecord(toolResult) && typeof toolResult.error === "string" && toolResult.error !== "") return true;
  const error = event.payload.error;
  return typeof error === "string" && error !== "";
}

function toolCallTitle(toolName: string, args: Record<string, unknown> | null) {
  const displayName = sentenceCase(readableToolName(toolName));
  const summary = args === null ? null : summarizeRecord(args, 140);
  return summary === null ? displayName : `${displayName}: ${summary}`;
}

const primarySummaryKeys = ["task", "command", "path", "query", "name", "file_path", "cwd", "pattern", "replacement"];
const noisySummaryKeyFragments = ["job_id", "thread_id", "agent_id", "call_id", "session_id", "content", "output", "truncated_output", "metadata"];

function summarizeRecord(record: Record<string, unknown>, maxLength: number) {
  const parts: string[] = [];
  for (const key of primarySummaryKeys) {
    const value = record[key];
    const summary = summarizeValue(value, Math.max(32, Math.floor(maxLength / 2)));
    if (summary !== null) parts.push(key === "task" || key === "command" || key === "path" || key === "query" ? summary : `${key}: ${summary}`);
  }
  if (parts.length === 0) {
    for (const [key, value] of Object.entries(record)) {
      if (shouldHideSummaryKey(key)) continue;
      const summary = summarizeValue(value, 48);
      if (summary !== null) parts.push(`${key}: ${summary}`);
      if (parts.length >= 3) break;
    }
  }
  if (parts.length === 0) return null;
  return _shortText(parts.join(" · "), maxLength);
}

function summarizeOutput(value: unknown) {
  if (typeof value !== "string" || value === "") return null;
  const parsed = jsonObjectFromString(value);
  if (parsed !== null) return summarizeRecord(parsed, 160);
  if (looksLikeRawJson(value)) return null;
  return _shortText(value, 160);
}

function summarizeValue(value: unknown, maxLength: number): string | null {
  if (typeof value === "string" && value !== "") {
    if (looksLikeRawJson(value)) {
      const parsed = jsonObjectFromString(value);
      return parsed === null ? null : summarizeRecord(parsed, maxLength);
    }
    return _shortText(value, maxLength);
  }
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) return value.length === 0 ? null : `${value.length} items`;
  if (isRecord(value)) return summarizeRecord(value, maxLength);
  return null;
}

function shouldHideSummaryKey(key: string) {
  const normalized = key.toLowerCase();
  return noisySummaryKeyFragments.some((fragment) => normalized.includes(fragment));
}

function looksLikeRawJson(value: string) {
  const trimmed = value.trim();
  return trimmed.startsWith("{") || trimmed.startsWith("[");
}

function sentenceCase(value: string) {
  if (value === "") return value;
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function toolResultSummary(event: SessionEvent, failed: boolean) {
  const error = event.payload.error;
  if (typeof error === "string" && error !== "") return _shortText(error, 160);
  const toolResult = event.payload.tool_result;
  if (isRecord(toolResult)) {
    const resultError = toolResult.error;
    if (typeof resultError === "string" && resultError !== "") return _shortText(resultError, 160);
    const truncatedOutput = toolResult.truncated_output;
    const summarizedOutput = summarizeOutput(truncatedOutput);
    if (summarizedOutput !== null) return summarizedOutput;
    const output = toolResult.output;
    const summarizedRawOutput = summarizeOutput(output);
    if (summarizedRawOutput !== null) return summarizedRawOutput;
  }
  const output = event.payload.output;
  const summarizedOutput = summarizeOutput(output);
  if (summarizedOutput !== null) return summarizedOutput;
  const content = event.payload.content;
  const summarizedContent = summarizeOutput(content);
  if (summarizedContent !== null) return summarizedContent;
  return failed ? "Failed" : "Completed";
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

function groupByProject<T extends ProjectGroupedItem>(items: T[]): ProjectGroup<T>[] {
  const groups: ProjectGroup<T>[] = [];
  const projectIndexes = new Map<string, number>();
  for (const item of items) {
    const index = projectIndexes.get(item.project_id);
    if (index === undefined) {
      projectIndexes.set(item.project_id, groups.length);
      groups.push({
        project_id: item.project_id,
        project_name: item.project_name,
        items: [item]
      });
      continue;
    }
    groups[index].items.push(item);
  }
  return groups;
}

function matchingCommands(input: string) {
  if (!input.startsWith("/")) return [];
  const commandName = input.split(/\s+/, 1)[0];
  return COMPOSER_COMMANDS.filter((command) => command.name.startsWith(commandName)).slice(0, 10);
}

function readClipboardImage(item: DataTransferItem): Promise<ComposerImageAttachment> {
  const file = item.getAsFile();
  if (file === null) {
    return Promise.reject(new Error("Pasted image is empty."));
  }
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Image paste failed."));
    reader.onload = () => {
      if (typeof reader.result !== "string") {
        reject(new Error("Image paste did not produce a data URL."));
        return;
      }
      const parsed = imageFromDataUrl(reader.result);
      resolve({
        id: crypto.randomUUID(),
        ...parsed
      });
    };
    reader.readAsDataURL(file);
  });
}

function imageFromDataUrl(value: string): ImageAttachmentPayload {
  const match = /^data:([^;]+);base64,(.*)$/.exec(value);
  if (match === null) {
    throw new Error("Pasted image data URL is invalid.");
  }
  return {
    mime_type: match[1],
    data: match[2]
  };
}

function imageDataUrl(image: ImageAttachmentPayload) {
  return `data:${image.mime_type};base64,${image.data}`;
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    throw new Error(`Request failed (${response.status}).`);
  }
  return response.json() as Promise<T>;
}

function configUpdatePayload(config: GuiConfig, apiKey: string) {
  const tool_permissions: Record<string, string> = {};
  const tool_enabled: Record<string, boolean> = {};
  const tool_max_calls: Record<string, number> = {};
  for (const tool of config.tools) {
    tool_permissions[tool.name] = tool.permission;
    tool_enabled[tool.name] = tool.enabled;
    if (typeof tool.max_calls_per_run === "number") {
      tool_max_calls[tool.name] = tool.max_calls_per_run;
    }
  }
  return {
    provider: config.provider,
    model: config.model,
    default_model: config.default_model,
    reasoning_level: config.reasoning_level,
    compress_threshold: config.compress_threshold,
    api_timeout_seconds: config.api_timeout_seconds,
    stream_start_timeout_seconds: config.stream_start_timeout_seconds,
    stream_event_timeout_seconds: config.stream_event_timeout_seconds,
    skill_selection_mode: config.skill_selection_mode,
    enabled_skills: config.enabled_skills,
    disabled_providers: config.disabled_providers,
    api_key: apiKey === "" ? null : apiKey,
    tool_permissions,
    tool_enabled,
    tool_max_calls
  };
}

function apiFailureMessage(label: string, error: unknown) {
  if (error instanceof ResponseError && (error.response.status === 502 || error.response.status === 503)) {
    return `${label} failed: AceAI GUI backend is unavailable. Restart aceai-gui.`;
  }
  if (error instanceof ResponseError) {
    return `${label} failed (${error.response.status}).`;
  }
  if (error instanceof FetchError) {
    return `${label} failed: AceAI GUI backend is unavailable. Restart aceai-gui.`;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return `${label} failed.`;
}

function sessionListItemFromApi(item: SessionListItemPayload): SessionListItem {
  return {
    session_id: item.sessionId,
    project_id: item.projectId,
    project_name: item.projectName,
    title: item.title,
    created_at: item.createdAt,
    updated_at: item.updatedAt,
    path: item.path ?? undefined,
    event_count: item.eventCount,
    total_cost_usd: item.totalCostUsd,
    thread_count: item.threadCount,
    active_thread: threadMetadataFromApi(item.activeThread)
  };
}

function threadMetadataFromApi(item: ThreadMetadataPayload) {
  return {
    session_id: item.sessionId,
    thread_id: item.threadId,
    role: item.role,
    title: item.title,
    status: item.status,
    agent_id: item.agentId,
    parent_thread_id: item.parentThreadId,
    parent_run_id: item.parentRunId,
    parent_tool_call_id: item.parentToolCallId,
    metadata: item.metadata as Record<string, unknown>,
    created_at: item.createdAt,
    updated_at: item.updatedAt
  };
}

function referenceItemFromApi(item: ReferenceItemPayload): ReferenceItem {
  if (item.kind !== "file" && item.kind !== "idea") {
    throw new Error(`Unknown reference kind: ${item.kind}`);
  }
  return {
    kind: item.kind,
    value: item.value,
    label: item.label,
    description: item.description,
    idea_id: item.ideaId ?? undefined
  };
}

function filePayloadFromApi(item: GeneratedFilePayload): FilePayload {
  return {
    path: item.path,
    content: item.content,
    size: item.size,
    updated_at: item.updatedAt
  };
}

function ideaItemFromApi(item: IdeaItemPayload): IdeaItem {
  return {
    index: item.index,
    idea_id: item.ideaId,
    created_at: item.createdAt,
    project_id: item.projectId,
    project_name: item.projectName,
    workspace: item.workspace,
    content: item.content,
    source_session_id: item.sourceSessionId
  };
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

function filePathMatchFromText(value: string) {
  const match = value.match(/(?:^|[\s:])((?:\/|\.{1,2}\/)[^\s`"',;:!?）\])]+?\.(?:mdx?|txt|py|tsx?|jsx?|json|css|html|ya?ml|toml|z?sh|bash))/i);
  if (match === null || match.index === undefined) return null;
  const path = match[1];
  const start = match.index + match[0].lastIndexOf(path);
  return {
    end: start + path.length,
    path,
    start
  };
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
    queued_turns: [],
    pending_approval: null,
    is_running_suspended: false,
    active_thread_accepts_user_turn: true,
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

function mergeSnapshotQueuedTurns(snapshot: SnapshotPayload, current: QueuedTurn[]): QueuedTurn[] {
  return snapshotQueuedTurnsFromPayloads(runtimeOf(snapshot).queued_turns);
}

function snapshotQueuedTurnsFromPayloads(turns: QueuedTurnPayload[]): QueuedTurn[] {
  return turns.map((turn, index) => ({
    attachments: turn.images,
    id: `queued-${index}-${turn.content}`,
    content: turn.content
  }));
}

function transcriptItemsMatch(left: TranscriptItem, right: TranscriptItem) {
  return (
    left.role === right.role &&
    left.text === right.text &&
    imageAttachmentsMatch(left.images ?? [], right.images ?? [])
  );
}

function imageAttachmentsMatch(left: ImageAttachmentPayload[], right: ImageAttachmentPayload[]) {
  if (left.length !== right.length) return false;
  return left.every(
    (image, index) =>
      image.mime_type === right[index].mime_type && image.data === right[index].data,
  );
}

function eventApprovalRequest(payload: Record<string, unknown>): ToolApprovalRequest | null {
  const request = payload.request;
  if (isRecord(request)) {
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
  const call = payload.tool_call;
  if (!isRecord(call)) return null;
  if (typeof call.call_id !== "string") return null;
  if (typeof call.name !== "string") return null;
  if (typeof call.arguments !== "string") return null;
  const toolName = stringField(payload, "tool_name");
  const reason = stringField(payload, "content");
  if (toolName === null || reason === null) return null;
  return {
    call: {
      call_id: call.call_id,
      name: call.name,
      arguments: call.arguments
    },
    tool_name: toolName,
    reason,
    policy: ""
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
  if (eventType === "agent.llm.reasoning") {
    const reasoning = extractLiveReasoningText(fields);
    return liveTimelineRow(payload, "Reasoning", reasoning ?? "Reasoning", "live", reasoning ?? undefined);
  }
  if (eventType === "agent.tool.started") {
    const toolName = stringField(fields, "tool_name") ?? "tool";
    return liveTimelineRow(payload, toolCallTitle(toolName, liveToolArguments(fields)), "Running", "live");
  }
  if (eventType === "agent.tool.completed") {
    const toolName = stringField(fields, "tool_name") ?? "tool";
    return liveTimelineRow(payload, toolCallTitle(toolName, liveToolArguments(fields)), liveToolResultSummary(fields) ?? "Completed", "good");
  }
  if (eventType === "agent.tool.failed") {
    const toolName = stringField(fields, "tool_name") ?? "tool";
    return liveTimelineRow(payload, toolCallTitle(toolName, liveToolArguments(fields)), liveToolResultSummary(fields) ?? stringField(fields, "error") ?? "Failed", "bad");
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
  content?: string,
): TimelineItem {
  const id = payload.event.event_type.startsWith("agent.tool.")
    ? `${payload.event.run_id}-tool-${liveToolCallId(payload.event.payload) ?? payload.event.step_id}`
    : `${payload.event.event_type}-${payload.event.run_id}-${payload.event.step_id}`;
  return {
    id,
    content,
    kind: payload.event.event_type,
    runId: payload.event.run_id,
    title,
    detail,
    tone,
  };
}

function extractLiveReasoningText(payload: Record<string, unknown>) {
  const content = stringField(payload, "content");
  if (content !== null) return content;
  const segment = payload.segment;
  if (!isRecord(segment)) return null;
  const segmentContent = segment.content;
  return typeof segmentContent === "string" && segmentContent !== "" ? segmentContent : null;
}

function liveToolArguments(payload: Record<string, unknown>) {
  const toolCall = payload.tool_call;
  if (!isRecord(toolCall)) return null;
  const argumentsValue = toolCall.arguments;
  if (typeof argumentsValue !== "string") return null;
  return jsonObjectFromString(argumentsValue);
}

function liveToolCallId(payload: Record<string, unknown>) {
  const toolCallId = stringField(payload, "tool_call_id");
  if (toolCallId !== null) return toolCallId;
  const toolCall = payload.tool_call;
  if (!isRecord(toolCall)) return null;
  const callId = toolCall.call_id;
  return typeof callId === "string" && callId !== "" ? callId : null;
}

function liveToolResultSummary(payload: Record<string, unknown>) {
  const toolResult = payload.tool_result;
  if (!isRecord(toolResult)) {
    const error = stringField(payload, "error");
    return error === null ? null : _shortText(error, 160);
  }
  const error = toolResult.error;
  if (typeof error === "string" && error !== "") return _shortText(error, 160);
  const truncatedOutput = toolResult.truncated_output;
  const summarizedTruncated = summarizeOutput(truncatedOutput);
  if (summarizedTruncated !== null) return summarizedTruncated;
  const output = toolResult.output;
  return summarizeOutput(output);
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

function sessionIdFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const sessionId = params.get(SESSION_URL_PARAM);
  return sessionId === "" ? null : sessionId;
}

function writeSessionIdToUrl(sessionId: string, options: { replace: boolean }) {
  const url = new URL(window.location.href);
  url.searchParams.set(SESSION_URL_PARAM, sessionId);
  updateBrowserUrl(url, options);
}

function clearSessionIdFromUrl(options: { replace: boolean }) {
  const url = new URL(window.location.href);
  url.searchParams.delete(SESSION_URL_PARAM);
  updateBrowserUrl(url, options);
}

function updateBrowserUrl(url: URL, options: { replace: boolean }) {
  if (url.href === window.location.href) return;
  if (options.replace) {
    window.history.replaceState({}, "", url);
    return;
  }
  window.history.pushState({}, "", url);
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

function apiBasePath(apiBaseUrl: string) {
  return apiBaseUrl.replace(/\/+$/, "");
}

function restApiUrl(path: string) {
  return `${apiBasePath(window.location.origin)}${path}`;
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

function SettingsWorkspace({
  apiKey,
  current,
  draft,
  saving,
  onApiKeyChange,
  onReload,
  onSave,
  onUpdate,
  onUpdateTool
}: {
  apiKey: string;
  current: GuiConfig | null;
  draft: GuiConfig | null;
  saving: boolean;
  onApiKeyChange: (value: string) => void;
  onReload: () => void;
  onSave: () => void;
  onUpdate: (updater: (draft: GuiConfig) => GuiConfig) => void;
  onUpdateTool: (toolName: string, updater: (tool: ToolPermissionConfig) => ToolPermissionConfig) => void;
}) {
  if (draft === null) {
    return (
      <div className="workspace-panel settings-workspace loading">
        <div className="empty-state inline-empty">
          <strong>Loading settings</strong>
          <span>Reading the active AceAI config.</span>
        </div>
      </div>
    );
  }
  const modelOptions = draft.models_by_provider[draft.provider] ?? draft.models;
  const skills = draft.skills ?? [];
  const enabledSkillCount = skills.filter((skill) => skill.enabled).length;
  const toolGroups = groupToolsByTag(draft.tools);
  return (
    <div className="workspace-panel settings-workspace">
      <aside className="settings-rail">
        <div className="settings-rail-title">
          <strong>Settings</strong>
          <span>Project preferences</span>
        </div>
        <nav className="settings-nav" aria-label="Settings sections">
          <a href="#settings-model">Model</a>
          <a href="#settings-skills">Skills</a>
          <a href="#settings-context">Context</a>
          <a href="#settings-tools">Tools</a>
        </nav>
        <div className="settings-rail-meta">
          <span>Provider</span>
          <strong>{providerLabel(draft)}</strong>
          <span>Model</span>
          <strong>{draft.model}</strong>
          <span>Skills</span>
          <strong>{enabledSkillCount} enabled</strong>
          <span>Config</span>
          <code>{shortPath(draft.config_path)}</code>
        </div>
      </aside>

      <div className="settings-main">
        <header className="settings-toolbar">
          <div>
            <strong>Preferences</strong>
            <span>{draft.config_path}</span>
          </div>
          <div className="settings-actions">
            <button type="button" onClick={onReload}>
              <RefreshCw size={13} />
              Reload
            </button>
            <button type="button" className="primary" disabled={saving} onClick={onSave}>
              <Save size={13} />
              {saving ? "Saving" : "Save"}
            </button>
          </div>
        </header>

        <section className="settings-section" id="settings-model">
          <div className="settings-section-title">
            <strong>Model</strong>
            <span>Provider, model, credentials, and reasoning effort.</span>
          </div>
          <div className="preference-list">
            <PreferenceRow title="Provider" description="The backend used for new agent turns.">
              <select
                value={draft.provider}
                onChange={(event) => {
                  const provider = event.target.value;
                  const nextModels = draft.models_by_provider[provider] ?? [];
                  const nextModel = nextModels[0]?.value ?? draft.model;
                  onUpdate((value) => ({ ...value, provider, model: nextModel, default_model: nextModel }));
                }}
              >
                {draft.providers.map((provider) => (
                  <option key={provider.value} value={provider.value}>{provider.label}</option>
                ))}
              </select>
            </PreferenceRow>
            <PreferenceRow title="Model" description="Default model for subsequent runs.">
              <select
                value={draft.model}
                onChange={(event) => {
                  const model = event.target.value;
                  onUpdate((value) => ({ ...value, model, default_model: model }));
                }}
              >
                {modelOptions.map((model) => (
                  <option key={model.value} value={model.value}>{model.label}</option>
                ))}
              </select>
            </PreferenceRow>
            <PreferenceRow title="Reasoning" description="Effort level when the selected model supports it.">
              <select
                value={draft.reasoning_level}
                onChange={(event) => onUpdate((value) => ({ ...value, reasoning_level: event.target.value }))}
              >
                {draft.reasoning_options.map((level) => (
                  <option key={level} value={level}>{formatReasoningLevel(level)}</option>
                ))}
              </select>
            </PreferenceRow>
            <PreferenceRow title="API key" description={current?.api_key_set ? `Stored credential detected in ${draft.api_key_env}.` : `Expected environment key: ${draft.api_key_env}.`}>
              <input
                value={apiKey}
                onChange={(event) => onApiKeyChange(event.target.value)}
                placeholder={current?.api_key_set ? "Leave blank to keep stored key" : draft.api_key_env}
                type="password"
              />
            </PreferenceRow>
          </div>
        </section>

        <section className="settings-section" id="settings-skills">
          <div className="settings-section-title">
            <strong>Skills</strong>
            <span>Reusable instructions loaded into the active agent.</span>
          </div>
          <div className="skill-settings-list">
            {skills.length === 0 ? (
              <div className="settings-empty-row">No skills are available for this project.</div>
            ) : skills.map((skill) => (
              <div className="skill-settings-row" key={skill.name}>
                <label className="settings-switch" title={skill.enabled ? "Enabled" : "Disabled"}>
                  <input
                    checked={skill.enabled}
                    type="checkbox"
                    onChange={(event) => {
                      const enabled = event.target.checked;
                      onUpdate((value) => {
                        const nextSkills = value.skills.map((item) => item.name === skill.name ? { ...item, enabled } : item);
                        return {
                          ...value,
                          skill_selection_mode: "selected",
                          enabled_skills: nextSkills.filter((item) => item.enabled).map((item) => item.name),
                          skills: nextSkills
                        };
                      });
                    }}
                  />
                  <span />
                </label>
                <div>
                  <div className="skill-settings-title">
                    <strong>{skill.name}</strong>
                    <code>{skill.source}</code>
                  </div>
                  <p>{skill.description}</p>
                  <small>{skill.location}</small>
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="settings-section" id="settings-context">
          <div className="settings-section-title">
            <strong>Context</strong>
            <span>Compression and request timeout behavior.</span>
          </div>
          <div className="preference-list">
            <PreferenceRow title="Compress at" description="Context compression threshold, percentage or token count.">
              <input value={draft.compress_threshold} onChange={(event) => onUpdate((value) => ({ ...value, compress_threshold: event.target.value }))} />
            </PreferenceRow>
            <PreferenceRow title="API timeout" description="Timeout for non-streaming LLM requests, in seconds.">
              <input type="number" value={draft.api_timeout_seconds} onChange={(event) => onUpdate((value) => ({ ...value, api_timeout_seconds: Number(event.target.value) }))} />
            </PreferenceRow>
            <PreferenceRow title="Stream start" description="How long to wait for the first streaming event.">
              <input type="number" value={draft.stream_start_timeout_seconds} onChange={(event) => onUpdate((value) => ({ ...value, stream_start_timeout_seconds: Number(event.target.value) }))} />
            </PreferenceRow>
            <PreferenceRow title="Stream idle" description="Maximum idle gap between streaming events.">
              <input type="number" value={draft.stream_event_timeout_seconds} onChange={(event) => onUpdate((value) => ({ ...value, stream_event_timeout_seconds: Number(event.target.value) }))} />
            </PreferenceRow>
          </div>
        </section>

        <section className="settings-section" id="settings-tools">
          <div className="settings-section-title">
            <strong>Tools</strong>
            <span>Capability groups with enablement, approval policy, and per-run limits.</span>
          </div>
          <div className="settings-tool-groups">
            {toolGroups.map((group) => (
              <section className="settings-tool-group" key={group.tag}>
                <div className="settings-tool-group-title">
                  <strong>{group.tag}</strong>
                  <span>{group.tools.length} tools</span>
                </div>
                {group.tools.map((tool) => (
                  <div className="settings-tool-row" key={tool.name}>
                    <div className="settings-tool-identity">
                      <label className="settings-switch" title={tool.enabled ? "Enabled" : "Disabled"}>
                        <input checked={tool.enabled} type="checkbox" onChange={(event) => onUpdateTool(tool.name, (value) => ({ ...value, enabled: event.target.checked }))} />
                        <span />
                      </label>
                      <div>
                        <strong>{tool.name}</strong>
                        <p>{tool.description}</p>
                      </div>
                    </div>
                    <div className="settings-tool-controls">
                      <div className="segmented-control" aria-label={`${tool.name} permission`}>
                        <button
                          type="button"
                          className={tool.permission === "always" ? "active" : ""}
                          onClick={() => onUpdateTool(tool.name, (value) => ({ ...value, permission: "always" }))}
                        >
                          Always
                        </button>
                        <button
                          type="button"
                          className={tool.permission === "ask" ? "active" : ""}
                          onClick={() => onUpdateTool(tool.name, (value) => ({ ...value, permission: "ask" }))}
                        >
                          Ask
                        </button>
                      </div>
                      <input
                        aria-label={`${tool.name} maximum calls`}
                        min={1}
                        placeholder="No limit"
                        type="number"
                        value={tool.max_calls_per_run ?? ""}
                        onChange={(event) => onUpdateTool(tool.name, (value) => ({
                          ...value,
                          max_calls_per_run: event.target.value === "" ? null : Number(event.target.value)
                        }))}
                      />
                    </div>
                  </div>
                ))}
              </section>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}

function PreferenceRow({ children, description, title }: { children: ReactNode; description: string; title: string }) {
  return (
    <div className="preference-row">
      <div>
        <strong>{title}</strong>
        <span>{description}</span>
      </div>
      <div className="preference-control">{children}</div>
    </div>
  );
}

function providerLabel(config: GuiConfig) {
  return config.providers.find((provider) => provider.value === config.provider)?.label ?? config.provider;
}

function shortPath(path: string) {
  const parts = path.split("/");
  if (parts.length <= 3) return path;
  return `.../${parts.slice(-3).join("/")}`;
}

function groupToolsByTag(tools: ToolPermissionConfig[]) {
  const groups: { tag: string; tools: ToolPermissionConfig[] }[] = [];
  const indexes = new Map<string, number>();
  for (const tool of tools) {
    const tag = tool.tags[0] ?? "untagged";
    const index = indexes.get(tag);
    if (index === undefined) {
      indexes.set(tag, groups.length);
      groups.push({ tag, tools: [tool] });
      continue;
    }
    groups[index].tools.push(tool);
  }
  return groups;
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
