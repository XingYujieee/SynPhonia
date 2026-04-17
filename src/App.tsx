import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type CSSProperties,
  type MouseEvent as ReactMouseEvent,
} from "react";
import {
  THEME_STORAGE_KEY,
  cloneInitialFiles,
  cloneInitialFolders,
  cloneInitialMessages,
  cloneInitialSummaries,
} from "./constants";
import type {
  ChatMessage,
  ConversationMessage,
  CourseFile,
  KnowledgeBaseData,
  LiteSynphoniaProviderSettings,
  RealtimeTranscriptionEvent,
  SummaryCard,
  SummaryEmptyState,
  SummaryWindowDebugState,
  ThemeMode,
  ViewMode,
  WorkspaceCache,
  WorkspaceFolder,
  WorkspacePipelineStatus,
} from "./types";
import { askFileQuestion, uploadCourseFile } from "./services/api";
import {
  createNormalModeWorkspace,
  deleteNormalModeWorkspace,
  getNormalModeWorkspacePipelineStatus,
  listNormalModeWorkspaces,
  openNormalModeWorkspace,
  pauseNormalModeWorkspacePipeline,
  renameNormalModeWorkspace,
  starNormalModeWorkspace,
  startNormalModeWorkspacePipeline,
} from "./services/workspaceCache";
import {
  getLiteSynphoniaProviderSettings,
  saveLiteSynphoniaProviderSettings,
} from "./services/providerSettings";
import ChatPanel from "./components/ChatPanel";
import KnowledgeGraphPanel from "./components/KnowledgeGraphPanel";
import ModeSwitch from "./components/ModeSwitch";
import PanelToggleIcon from "./components/PanelToggleIcon";
import PdfPreview from "./components/PdfPreview";
import ReviewSidebar from "./components/ReviewSidebar";
import SummarySidebar from "./components/SummarySidebar";
import { isPdfCourseFile } from "./utils/courseFiles";
import settingsIcon from "./assets-settings.svg";
function getStoredTheme(): ThemeMode {
  const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
  return storedTheme === "dark" ? "dark" : "light";
}

const LEFT_PANEL_WIDTH_STORAGE_KEY = "course-pdf-left-panel-width";
const RIGHT_PANEL_WIDTH_STORAGE_KEY = "course-pdf-right-panel-width";
const DEFAULT_LEFT_PANEL_WIDTH = 296;
const DEFAULT_RIGHT_PANEL_WIDTH = 372;
const MIN_LEFT_PANEL_WIDTH = 248;
const MAX_LEFT_PANEL_WIDTH = 440;
const MIN_RIGHT_PANEL_WIDTH = 320;
const MAX_RIGHT_PANEL_WIDTH = 520;
const MIN_CENTER_WIDTH = 520;
const EMPTY_PROVIDER_SETTINGS: LiteSynphoniaProviderSettings = {
  configPath: "",
  deepgramApiKey: "",
  deepseekApiKey: "",
  siliconflowApiKey: "",
  transcriptionLanguage: "zh-CN",
  hasTranscriptionProvider: false,
  hasSummarizationProvider: false,
  hasEmbeddingProvider: false,
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function getStoredPanelWidth(
  storageKey: string,
  fallbackValue: number,
  minValue: number,
  maxValue: number,
): number {
  const storedValue = window.localStorage.getItem(storageKey);
  const parsedValue = storedValue ? Number(storedValue) : Number.NaN;

  if (!Number.isFinite(parsedValue)) {
    return fallbackValue;
  }

  return clamp(parsedValue, minValue, maxValue);
}

/**
 * TypeScript reference implementation for the current product surface.
 * The goal is to keep behavior aligned with the existing JS version while
 * providing a maintainable foundation for future frontend collaboration.
 */
export default function App() {
  const [theme, setTheme] = useState<ThemeMode>(getStoredTheme);
  const [viewMode, setViewMode] = useState<ViewMode>("normal");
  const [isLeftOpen, setIsLeftOpen] = useState(true);
  const [isRightOpen, setIsRightOpen] = useState(true);
  const [leftPanelWidth, setLeftPanelWidth] = useState(() =>
    getStoredPanelWidth(
      LEFT_PANEL_WIDTH_STORAGE_KEY,
      DEFAULT_LEFT_PANEL_WIDTH,
      MIN_LEFT_PANEL_WIDTH,
      MAX_LEFT_PANEL_WIDTH,
    ),
  );
  const [rightPanelWidth, setRightPanelWidth] = useState(() =>
    getStoredPanelWidth(
      RIGHT_PANEL_WIDTH_STORAGE_KEY,
      DEFAULT_RIGHT_PANEL_WIDTH,
      MIN_RIGHT_PANEL_WIDTH,
      MAX_RIGHT_PANEL_WIDTH,
    ),
  );
  const [resizeSide, setResizeSide] = useState<"left" | "right" | null>(null);

  const [folders, setFolders] =
    useState<WorkspaceFolder[]>(cloneInitialFolders);
  const [files, setFiles] = useState<CourseFile[]>(cloneInitialFiles);
  const [selectedFolderId, setSelectedFolderId] = useState("");
  const [currentFileId, setCurrentFileId] = useState("");
  const [activeWorkspaceId, setActiveWorkspaceId] = useState("");
  const [recentWorkspaces, setRecentWorkspaces] = useState<WorkspaceCache[]>(
    [],
  );
  const [summaries, setSummaries] = useState<SummaryCard[]>(
    cloneInitialSummaries,
  );
  const [summaryEmptyState, setSummaryEmptyState] = useState<
    SummaryEmptyState | undefined
  >();
  const [, setSummaryWindowDebugState] = useState<
    SummaryWindowDebugState | undefined
  >();
  const [expandedSummaryId, setExpandedSummaryId] = useState("");
  const [isWorkspaceLoading, setIsWorkspaceLoading] = useState(false);
  const [workspaceErrorMessage, setWorkspaceErrorMessage] = useState("");
  const [pipelineStatus, setPipelineStatus] =
    useState<WorkspacePipelineStatus | null>(null);
  const [isStartingPipeline, setIsStartingPipeline] = useState(false);

  const [chatMessages, setChatMessages] =
    useState<ChatMessage[]>(cloneInitialMessages);
  const [chatInput, setChatInput] = useState("");
  const [isAsking, setIsAsking] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [providerSettings, setProviderSettings] =
    useState<LiteSynphoniaProviderSettings>(EMPTY_PROVIDER_SETTINGS);
  const [isProviderSettingsLoading, setIsProviderSettingsLoading] =
    useState(false);
  const [isProviderSettingsSaving, setIsProviderSettingsSaving] =
    useState(false);
  const [providerSettingsFeedback, setProviderSettingsFeedback] = useState("");
  const [providerSettingsFeedbackTone, setProviderSettingsFeedbackTone] =
    useState<"success" | "error" | "">("");

  const [realtimeStatus, setRealtimeStatus] = useState<string>("idle");
  const [realtimeError, setRealtimeError] = useState("");
  const [isRealtimeActive, setIsRealtimeActive] = useState(false);
  const [realtimePartialText, setRealtimePartialText] = useState("");

  // ── 新增：转录全文 + 页码匹配 + 知识库 ──────────────
  const [fullTranscriptText, setFullTranscriptText] = useState("");
  const [currentMatchedPage, setCurrentMatchedPage] = useState<number | undefined>(undefined);
  const [isKnowledgeGraphOpen, setIsKnowledgeGraphOpen] = useState(false);
  const [knowledgeGraphData, setKnowledgeGraphData] =
    useState<KnowledgeBaseData | null>(null);
  const [isKnowledgeGraphLoading, setIsKnowledgeGraphLoading] = useState(false);
  // PPT 转换状态
  const [isPptConverting, setIsPptConverting] = useState(false);
  const [pptConversionError, setPptConversionError] = useState("");
  const [openWorkspaceMenuId, setOpenWorkspaceMenuId] = useState("");

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const appFrameRef = useRef<HTMLDivElement | null>(null);
  const handledPipelineCompletionRef = useRef("");
  const handledPipelineCycleRefreshRef = useRef("");
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const scriptNodeRef = useRef<ScriptProcessorNode | null>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  useEffect(() => {
    window.localStorage.setItem(
      LEFT_PANEL_WIDTH_STORAGE_KEY,
      String(Math.round(leftPanelWidth)),
    );
  }, [leftPanelWidth]);

  useEffect(() => {
    window.localStorage.setItem(
      RIGHT_PANEL_WIDTH_STORAGE_KEY,
      String(Math.round(rightPanelWidth)),
    );
  }, [rightPanelWidth]);

  useEffect(() => {
    function handlePointerDown(event: PointerEvent): void {
      const target = event.target;
      if (
        target instanceof Element &&
        target.closest(".workspace-home-menu-shell")
      ) {
        return;
      }

      setOpenWorkspaceMenuId("");
    }

    document.addEventListener("pointerdown", handlePointerDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
    };
  }, []);

  useEffect(() => {
    void refreshRecentWorkspaces();
  }, []);

  useEffect(() => {
    if (
      !activeWorkspaceId ||
      !window.desktopBridge?.getNormalWorkspacePipelineStatus
    ) {
      setPipelineStatus(null);
      handledPipelineCompletionRef.current = "";
      return;
    }

    let isCancelled = false;
    let timerId = 0;

    async function pollStatus(): Promise<void> {
      try {
        const nextStatus =
          await getNormalModeWorkspacePipelineStatus(activeWorkspaceId);
        if (isCancelled) {
          return;
        }

        setPipelineStatus(nextStatus);
        setRecentWorkspaces((previousWorkspaces) =>
          previousWorkspaces.map((workspace) =>
            workspace.workspaceId === activeWorkspaceId
              ? { ...workspace, status: nextStatus.processingStatus }
              : workspace,
          ),
        );

        if (nextStatus.runState === "running") {
          timerId = window.setTimeout(() => {
            void pollStatus();
          }, 1500);
        }
      } catch (error) {
        if (!isCancelled) {
          console.error("读取工作区转录状态失败", error);
        }
      }
    }

    void pollStatus();

    return () => {
      isCancelled = true;
      window.clearTimeout(timerId);
    };
  }, [activeWorkspaceId]);

  useEffect(() => {
    if (!activeWorkspaceId || !pipelineStatus?.finishedAtUtc) {
      handledPipelineCompletionRef.current = "";
      return;
    }

    const completionKey = `${activeWorkspaceId}:${pipelineStatus.runState}:${pipelineStatus.finishedAtUtc}`;
    if (handledPipelineCompletionRef.current === completionKey) {
      return;
    }

    handledPipelineCompletionRef.current = completionKey;

    if (pipelineStatus.runState === "succeeded") {
      void handleOpenWorkspace(activeWorkspaceId);
      void refreshRecentWorkspaces();
      return;
    }

    if (pipelineStatus.runState === "failed") {
      setWorkspaceErrorMessage(
        pipelineStatus.message || "LiteSynphonia 运行失败，请检查配置和日志。",
      );
      void refreshRecentWorkspaces();
    }
  }, [activeWorkspaceId, pipelineStatus]);

  useEffect(() => {
    if (!activeWorkspaceId) {
      handledPipelineCycleRefreshRef.current = "";
      return;
    }

    const completedCycles = pipelineStatus?.completedCycles || 0;
    if (!completedCycles) {
      return;
    }

    const marker = `${activeWorkspaceId}:${completedCycles}`;
    if (handledPipelineCycleRefreshRef.current === marker) {
      return;
    }

    handledPipelineCycleRefreshRef.current = marker;
    void handleOpenWorkspace(activeWorkspaceId);
  }, [activeWorkspaceId, pipelineStatus?.completedCycles]);

  useEffect(() => {
    setFolders((previousFolders) =>
      recentWorkspaces.map((workspace) => {
        const existingFolder = previousFolders.find(
          (folder) => folder.id === workspace.workspaceId,
        );

        return {
          id: workspace.workspaceId,
          name: workspace.workspaceName,
          expanded:
            existingFolder?.expanded ??
            (workspace.workspaceId === activeWorkspaceId ||
              workspace.workspaceId === selectedFolderId),
          createdAtUtc: workspace.createdAtUtc,
        };
      }),
    );
    setFiles((previousFiles) => {
      const workspaceFiles = recentWorkspaces.map((workspace) => {
        const fileId = getWorkspaceFileId(workspace.workspaceId);
        const existingFile = previousFiles.find((file) => file.id === fileId);

        return {
          id: fileId,
          folderId: workspace.workspaceId,
          name: workspace.sourceFileName,
          previewUrl: existingFile?.previewUrl || "",
          kind: workspace.sourceKind,
          type: existingFile?.type || "",
          size: existingFile?.size,
          extractedText: existingFile?.extractedText,
          workspace,
        };
      });

      return workspaceFiles;
    });
  }, [activeWorkspaceId, recentWorkspaces, selectedFolderId]);

  const currentFile = useMemo(
    () => files.find((file) => file.id === currentFileId) ?? null,
    [files, currentFileId],
  );
  const canPreviewCurrentFile = currentFile ? isPdfCourseFile(currentFile) : false;
  const activeWorkspace = currentFile?.workspace ?? null;
  const isPipelineRunning = pipelineStatus?.runState === "running";
  const inlineWorkspaceStatusMessage =
    workspaceErrorMessage ||
    (currentFile && activeWorkspace?.status === "failed"
      ? "当前工作区上一次转录失败，未生成可展示的总结或转录内容。请重新点击开始转录，并查看工作区目录中的 lite_synphonia.run.log。"
      : "");

  function sortWorkspacesByRecent(
    workspaces: ReadonlyArray<WorkspaceCache>,
  ): WorkspaceCache[] {
    return [...workspaces].sort((left, right) => {
      const starredDelta =
        Number(Boolean(right?.starred)) - Number(Boolean(left?.starred));
      if (starredDelta) {
        return starredDelta;
      }

      const rightStamp = Date.parse(
        right.lastOpenedAtUtc || right.updatedAtUtc || "",
      );
      const leftStamp = Date.parse(
        left.lastOpenedAtUtc || left.updatedAtUtc || "",
      );
      return rightStamp - leftStamp;
    });
  }

  function getWorkspaceFileId(workspaceId: string): string {
    return `workspace-${workspaceId}`;
  }

  function upsertWorkspace(
    previousWorkspaces: ReadonlyArray<WorkspaceCache>,
    workspace: WorkspaceCache,
  ): WorkspaceCache[] {
    return sortWorkspacesByRecent([
      workspace,
      ...previousWorkspaces.filter(
        (item) => item.workspaceId !== workspace.workspaceId,
      ),
    ]);
  }

  function upsertFile(
    previousFiles: ReadonlyArray<CourseFile>,
    nextFile: CourseFile,
  ): CourseFile[] {
    return [
      nextFile,
      ...previousFiles.filter((file) => file.id !== nextFile.id),
    ];
  }

  function getPipelineStatusLabel(
    status: WorkspacePipelineStatus | null,
  ): string {
    if (!status) {
      return "未开始转录";
    }

    if (status.runState === "running") {
      switch (status.processingStatus) {
        case "recording":
          return "正在录制";
        case "transcribing":
          return "正在转录";
        case "summarizing":
          return "正在总结";
        case "matching":
          return "正在匹配页码";
        default:
          return "处理中";
      }
    }

    if (status.runState === "succeeded") {
      return "已完成";
    }

    if (status.runState === "failed") {
      return "运行失败";
    }

    if (status.processingStatus === "paused") {
      return "已暂停";
    }

    if (status.processingStatus === "ready") {
      return "已完成";
    }

    return "未开始转录";
  }

  function resetWorkspacePresentationState(): void {
    setSummaries([]);
    setSummaryEmptyState(undefined);
    setSummaryWindowDebugState(undefined);
    setExpandedSummaryId("");
    setWorkspaceErrorMessage("");
    setFullTranscriptText("");
    setCurrentMatchedPage(undefined);
    setPptConversionError("");
    setRealtimePartialText("");
    setRealtimeError("");
  }

  async function stopRealtimeSession(workspaceId: string): Promise<void> {
    if (!workspaceId) {
      return;
    }

    stopMicCapture();

    try {
      const result =
        await window.desktopBridge?.stopRealtimeTranscription?.(workspaceId);
      if (result?.status) {
        setPipelineStatus(result.status);
      }
    } catch (error) {
      console.error("停止实时转写失败", error);
    } finally {
      setIsRealtimeActive(false);
      setRealtimeStatus("idle");
      setRealtimePartialText("");
    }
  }

  async function refreshRecentWorkspaces(): Promise<void> {
    if (!window.desktopBridge?.listNormalWorkspaces) {
      return;
    }

    try {
      const workspaces = await listNormalModeWorkspaces();
      setRecentWorkspaces(workspaces);
    } catch (error) {
      console.error("读取缓存工作区失败", error);
      setWorkspaceErrorMessage("读取缓存工作区失败，请稍后再试。");
    }
  }

  async function loadProviderSettings(): Promise<void> {
    if (!window.desktopBridge?.getLiteSynphoniaProviderSettings) {
      return;
    }

    setIsProviderSettingsLoading(true);
    setProviderSettingsFeedback("");
    setProviderSettingsFeedbackTone("");

    try {
      const nextSettings = await getLiteSynphoniaProviderSettings();
      setProviderSettings(nextSettings);
    } catch (error) {
      console.error("读取 LiteSynphonia 配置失败", error);
      setProviderSettingsFeedback(
        error instanceof Error
          ? error.message
          : "读取 LiteSynphonia 配置失败。",
      );
      setProviderSettingsFeedbackTone("error");
    } finally {
      setIsProviderSettingsLoading(false);
    }
  }

  function handleOpenSettings(): void {
    setIsSettingsOpen(true);
    void loadProviderSettings();
  }

  async function handleSaveProviderSettings(): Promise<void> {
    setIsProviderSettingsSaving(true);
    setProviderSettingsFeedback("");
    setProviderSettingsFeedbackTone("");

    try {
      const nextSettings = await saveLiteSynphoniaProviderSettings({
        deepgramApiKey: providerSettings.deepgramApiKey,
        deepseekApiKey: providerSettings.deepseekApiKey,
        siliconflowApiKey: providerSettings.siliconflowApiKey,
        transcriptionLanguage: providerSettings.transcriptionLanguage,
      });
      setProviderSettings(nextSettings);
      setProviderSettingsFeedback("LiteSynphonia 配置已保存。");
      setProviderSettingsFeedbackTone("success");
      setWorkspaceErrorMessage("");
    } catch (error) {
      console.error("保存 LiteSynphonia 配置失败", error);
      setProviderSettingsFeedback(
        error instanceof Error
          ? error.message
          : "保存 LiteSynphonia 配置失败。",
      );
      setProviderSettingsFeedbackTone("error");
    } finally {
      setIsProviderSettingsSaving(false);
    }
  }

  function handleToggleFolder(folderId: string): void {
    setFolders((previousFolders) =>
      previousFolders.map((folder) =>
        folder.id === folderId
          ? { ...folder, expanded: !folder.expanded }
          : folder,
      ),
    );
  }

  function handleSelectFile(fileId: string, folderId: string): void {
    setSelectedFolderId(folderId);

    const nextFile = files.find((file) => file.id === fileId);
    if (nextFile?.workspace?.workspaceId) {
      void handleOpenWorkspace(nextFile.workspace.workspaceId);
      return;
    }

    setCurrentFileId(fileId);
    setActiveWorkspaceId("");
    resetWorkspacePresentationState();
  }

  async function handleRenameWorkspace(workspaceId: string): Promise<void> {
    const currentWorkspace = recentWorkspaces.find(
      (workspace) => workspace.workspaceId === workspaceId,
    );

    if (!currentWorkspace) {
      setWorkspaceErrorMessage("没有找到要重命名的工作区。");
      return;
    }

    const nextWorkspaceName = window.prompt(
      "请输入新的工作区名称",
      currentWorkspace.workspaceName,
    );

    if (nextWorkspaceName === null) {
      return;
    }

    const trimmedName = nextWorkspaceName.trim();
    if (!trimmedName) {
      setWorkspaceErrorMessage("工作区名称不能为空。");
      return;
    }

    try {
      const renamedWorkspace = await renameNormalModeWorkspace({
        workspaceId,
        workspaceName: trimmedName,
      });

      setRecentWorkspaces((previousWorkspaces) =>
        upsertWorkspace(previousWorkspaces, renamedWorkspace),
      );
      setFiles((previousFiles) =>
        previousFiles.map((file) =>
          file.workspace?.workspaceId === workspaceId
            ? {
                ...file,
                workspace: renamedWorkspace,
              }
            : file,
        ),
      );
      setWorkspaceErrorMessage("");
    } catch (error) {
      console.error("重命名工作区失败", error);
      setWorkspaceErrorMessage(
        error instanceof Error ? error.message : "重命名工作区失败。",
      );
    }
  }

  async function handleDeleteWorkspace(workspaceId: string): Promise<void> {
    setOpenWorkspaceMenuId("");

    const currentWorkspace = recentWorkspaces.find(
      (workspace) => workspace.workspaceId === workspaceId,
    );

    if (!currentWorkspace) {
      setWorkspaceErrorMessage("没有找到要删除的工作区。");
      return;
    }

    const shouldDelete = window.confirm(
      `确定删除工作区“${currentWorkspace.workspaceName}”吗？缓存目录和副本文件也会一起删除。`,
    );

    if (!shouldDelete) {
      return;
    }

    try {
      const deletedWorkspaceId = await deleteNormalModeWorkspace(workspaceId);
      setRecentWorkspaces((previousWorkspaces) =>
        previousWorkspaces.filter(
          (workspace) => workspace.workspaceId !== deletedWorkspaceId,
        ),
      );
      setFiles((previousFiles) =>
        previousFiles.filter(
          (file) => file.workspace?.workspaceId !== deletedWorkspaceId,
        ),
      );
      setFolders((previousFolders) =>
        previousFolders.filter((folder) => folder.id !== deletedWorkspaceId),
      );

      if (
        activeWorkspaceId === deletedWorkspaceId ||
        selectedFolderId === deletedWorkspaceId
      ) {
        await handleCloseCurrentPreview();
        setSelectedFolderId("");
      }

      setWorkspaceErrorMessage("");
    } catch (error) {
      console.error("删除工作区失败", error);
      setWorkspaceErrorMessage(
        error instanceof Error ? error.message : "删除工作区失败。",
      );
    }
  }

  async function handleStarWorkspace(
    workspaceId: string,
    starred: boolean,
  ): Promise<void> {
    setOpenWorkspaceMenuId("");

    try {
      const updatedWorkspace = await starNormalModeWorkspace({
        workspaceId,
        starred,
      });

      setRecentWorkspaces((previousWorkspaces) =>
        upsertWorkspace(previousWorkspaces, updatedWorkspace),
      );
      setFiles((previousFiles) =>
        previousFiles.map((file) =>
          file.workspace?.workspaceId === workspaceId
            ? {
                ...file,
                workspace: updatedWorkspace,
              }
            : file,
        ),
      );
      setWorkspaceErrorMessage("");
    } catch (error) {
      console.error("更新工作区收藏状态失败", error);
      setWorkspaceErrorMessage(
        error instanceof Error ? error.message : "更新工作区收藏状态失败。",
      );
    }
  }

  function handleToggleWorkspaceMenu(
    event: ReactMouseEvent<HTMLButtonElement>,
    workspaceId: string,
  ): void {
    event.stopPropagation();
    setOpenWorkspaceMenuId((currentId) =>
      currentId === workspaceId ? "" : workspaceId,
    );
  }

  function handleToggleSummary(summaryId: string): void {
    setExpandedSummaryId((currentId) =>
      currentId === summaryId ? "" : summaryId,
    );
  }

  // ── Realtime streaming STT: mic capture via ScriptProcessorNode ──
  // Audio is captured continuously, converted to PCM16, and sent as
  // small chunks (default 100ms) via IPC to main process, which relays
  // to Deepgram's WebSocket. This is true streaming, not record-then-upload.

  const stopMicCapture = useCallback(() => {
    if (scriptNodeRef.current) {
      scriptNodeRef.current.disconnect();
      scriptNodeRef.current = null;
    }
    if (mediaStreamRef.current) {
      for (const track of mediaStreamRef.current.getTracks()) {
        track.stop();
      }
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current) {
      void audioContextRef.current.close();
      audioContextRef.current = null;
    }
  }, []);

  const startMicCapture = useCallback(
    async (
      workspaceId: string,
      chunkDurationMs: number = 100,
    ): Promise<boolean> => {
      const bridge = window.desktopBridge;
      if (!bridge?.pushRealtimeAudioChunk) {
        setRealtimeError("当前环境不支持实时音频采集桥接。");
        setRealtimeStatus("error");
        return false;
      }

      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            sampleRate: 16000,
            echoCancellation: true,
            noiseSuppression: true,
          },
        });
      } catch (err) {
        setRealtimeError(
          err instanceof DOMException && err.name === "NotAllowedError"
            ? "麦克风权限被拒绝，请在系统设置中允许此应用使用麦克风。"
            : `麦克风访问失败: ${err instanceof Error ? err.message : String(err)}`,
        );
        setRealtimeStatus("error");
        return false;
      }

      mediaStreamRef.current = stream;
      const sampleRate = 16000;
      const ctx = new AudioContext({ sampleRate });
      audioContextRef.current = ctx;

      // Ensure AudioContext is running (browsers may create it suspended)
      if (ctx.state === "suspended") {
        await ctx.resume();
      }

      const source = ctx.createMediaStreamSource(stream);

      // ScriptProcessorNode: bufferSize chosen to approximate chunkDurationMs
      const bufferSize = Math.pow(
        2,
        Math.max(
          8,
          Math.ceil(Math.log2((sampleRate * chunkDurationMs) / 1000)),
        ),
      );
      const processor = ctx.createScriptProcessor(bufferSize, 1, 1);
      scriptNodeRef.current = processor;

      processor.onaudioprocess = (e: AudioProcessingEvent) => {
        const float32 = e.inputBuffer.getChannelData(0);

        // Compute RMS for basic VAD
        let sumSq = 0;
        for (let i = 0; i < float32.length; i++) {
          sumSq += float32[i] * float32[i];
        }
        const rms = Math.sqrt(sumSq / float32.length);
        const hasSpeech = rms > 0.01;

        // Convert Float32 to PCM16 (Linear16)
        const pcm16 = new Int16Array(float32.length);
        for (let i = 0; i < float32.length; i++) {
          const s = Math.max(-1, Math.min(1, float32[i]));
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }

        bridge.pushRealtimeAudioChunk({
          workspaceId,
          audio: new Uint8Array(pcm16.buffer),
          sampleRate,
          chunkDurationMs: chunkDurationMs as 50 | 100 | 150,
          rms,
          hasSpeech,
        });
      };

      source.connect(processor);
      processor.connect(ctx.destination);
      return true;
    },
    [],
  );

  // Listen for realtime transcription events from main process
  useEffect(() => {
    const bridge = window.desktopBridge;
    if (!bridge?.onRealtimeTranscriptionEvent) return;

    const unsubscribe = bridge.onRealtimeTranscriptionEvent(
      (event: RealtimeTranscriptionEvent) => {
        if (event.workspaceId !== activeWorkspaceId) return;

        switch (event.type) {
          case "partial":
            setRealtimeStatus("streaming");
            setRealtimePartialText(event.partialText || "");
            break;
          case "final":
            setRealtimeStatus("streaming");
            setRealtimePartialText("");
            // 累积完整转录文本（用于"转录全文"Tab）
            if (event.finalSegment?.text) {
              setFullTranscriptText((prev) =>
                prev
                  ? prev + "\n" + event.finalSegment!.text
                  : event.finalSegment!.text,
              );
            }
            break;
          case "summary":
            if ((event as any).summaryItem) {
              const item = (event as any).summaryItem;
              setSummaries((prev) => [
                ...prev,
                {
                  id: item.id,
                  summary: item.summary,
                  transcript: item.transcript,
                },
              ]);
            }
            break;
          case "page_match":
            if (event.matchedPage && event.matchedPage > 0) {
              setCurrentMatchedPage(event.matchedPage);
            }
            break;
          case "status":
            setRealtimeStatus(event.status);
            break;
          case "error":
            setRealtimeError(event.lastError || "实时转写出错。");
            setRealtimeStatus("error");
            if (event.lastError?.includes("已停止自动重连")) {
              stopMicCapture();
              setIsRealtimeActive(false);
            }
            break;
          case "stopped":
            setRealtimeStatus("idle");
            setIsRealtimeActive(false);
            setRealtimePartialText("");
            break;
        }
      },
    );

    return unsubscribe;
  }, [activeWorkspaceId]);

  async function handleStartPipeline(): Promise<void> {
    if (!activeWorkspace) {
      setWorkspaceErrorMessage("请先选择一个工作区再开始转录。");
      return;
    }

    if (isRealtimeActive || isStartingPipeline) {
      return;
    }

    const bridge = window.desktopBridge;
    const useRealtime = !!bridge?.startRealtimeTranscription;

    setIsStartingPipeline(true);
    setWorkspaceErrorMessage("");
    setRealtimeError("");

    try {
      if (useRealtime) {
        setRealtimeStatus("connecting");
        setIsRealtimeActive(true);

        const result = await bridge.startRealtimeTranscription({
          workspaceId: activeWorkspace.workspaceId,
          chunkDurationMs: 200,
        });

        if (!result.ok) {
          setWorkspaceErrorMessage(result.message || "启动实时转写失败。");
          setIsRealtimeActive(false);
          setRealtimeStatus("error");
          return;
        }

        if (result.status) {
          setPipelineStatus(result.status);
        }

        const didStartMicCapture = await startMicCapture(
          activeWorkspace.workspaceId,
          200,
        );

        if (!didStartMicCapture) {
          try {
            const stopResult = await bridge.stopRealtimeTranscription(
              activeWorkspace.workspaceId,
            );
            if (stopResult.status) {
              setPipelineStatus(stopResult.status);
            }
          } catch (stopError) {
            console.error("启动失败后停止实时转写失败", stopError);
          }
          setIsRealtimeActive(false);
          return;
        }
      } else {
        const nextStatus = await startNormalModeWorkspacePipeline({
          workspaceId: activeWorkspace.workspaceId,
        });
        setPipelineStatus(nextStatus);
        setRecentWorkspaces((previousWorkspaces) =>
          previousWorkspaces.map((workspace) =>
            workspace.workspaceId === activeWorkspace.workspaceId
              ? { ...workspace, status: nextStatus.processingStatus }
              : workspace,
          ),
        );
      }
    } catch (error) {
      console.error("启动工作区转录失败", error);
      setWorkspaceErrorMessage(
        error instanceof Error ? error.message : "启动工作区转录失败。",
      );
      setIsRealtimeActive(false);
      setRealtimeStatus("error");
    } finally {
      setIsStartingPipeline(false);
    }
  }

  async function handlePausePipeline(): Promise<void> {
    if (!activeWorkspace) {
      setWorkspaceErrorMessage("请先选择一个工作区再暂停监听。");
      return;
    }

    const bridge = window.desktopBridge;

    try {
      if (isRealtimeActive && bridge?.stopRealtimeTranscription) {
        stopMicCapture();
        const result = await bridge.stopRealtimeTranscription(
          activeWorkspace.workspaceId,
        );
        if (result.status) {
          setPipelineStatus(result.status);
        }
        setIsRealtimeActive(false);
        setRealtimeStatus("idle");
        setRealtimePartialText("");
      } else if (isPipelineRunning) {
        const nextStatus = await pauseNormalModeWorkspacePipeline(
          activeWorkspace.workspaceId,
        );
        setPipelineStatus(nextStatus);
      }
      setWorkspaceErrorMessage("");
    } catch (error) {
      console.error("暂停工作区监听失败", error);
      setWorkspaceErrorMessage(
        error instanceof Error ? error.message : "暂停工作区监听失败。",
      );
    }
  }

  // Cleanup mic on unmount or workspace change
  useEffect(() => {
    return () => {
      stopMicCapture();
    };
  }, [activeWorkspaceId, stopMicCapture]);

  async function handleCloseCurrentPreview(): Promise<void> {
    if (isRealtimeActive && activeWorkspaceId) {
      await stopRealtimeSession(activeWorkspaceId);
    }

    setCurrentFileId("");
    setActiveWorkspaceId("");
    setPipelineStatus(null);
    resetWorkspacePresentationState();
  }

  async function handleOpenKnowledgeGraph(): Promise<void> {
    setIsKnowledgeGraphOpen(true);
    setIsKnowledgeGraphLoading(true);
    try {
      const bridge = window.desktopBridge;
      if (bridge?.getKnowledgeBaseData) {
        const result = await bridge.getKnowledgeBaseData();
        if (result.ok && result.data) {
          setKnowledgeGraphData(result.data);
        }
      }
    } catch (error) {
      console.error("加载知识库失败", error);
    } finally {
      setIsKnowledgeGraphLoading(false);
    }
  }

  async function handleConvertPptx(): Promise<void> {
    if (!activeWorkspace || isPptConverting) return;

    setIsPptConverting(true);
    setPptConversionError("");

    try {
      const bridge = window.desktopBridge;
      if (!bridge?.convertPptxToPdf) {
        throw new Error("当前环境不支持 PPT 转换。");
      }

      const result = await bridge.convertPptxToPdf({
        workspaceId: activeWorkspace.workspaceId,
      });

      if (!result.ok || !result.pdfBytes) {
        throw new Error(result.message || "PPT 转换失败。");
      }

      // 用转换后的 PDF bytes 生成预览 URL
      const pdfBlob = new Blob([new Uint8Array(result.pdfBytes)], {
        type: "application/pdf",
      });
      const pdfUrl = URL.createObjectURL(pdfBlob);

      setFiles((previousFiles) =>
        previousFiles.map((file) =>
          file.workspace?.workspaceId === activeWorkspace.workspaceId
            ? {
                ...file,
                kind: "pdf" as const,
                previewUrl: pdfUrl,
                workspace: result.workspace ?? file.workspace,
              }
            : file,
        ),
      );

      if (result.workspace) {
        setRecentWorkspaces((previousWorkspaces) =>
          upsertWorkspace(previousWorkspaces, result.workspace!),
        );
      }
    } catch (error) {
      console.error("PPT 转换失败", error);
      setPptConversionError(
        error instanceof Error ? error.message : "PPT 转换失败，请检查是否安装了 LibreOffice。",
      );
    } finally {
      setIsPptConverting(false);
    }
  }

  function activateExistingWorkspaceFile(workspaceId: string): boolean {
    const matchedFile = files.find(
      (file) => file.workspace?.workspaceId === workspaceId,
    );

    if (!matchedFile) {
      return false;
    }

    setFolders((previousFolders) =>
      previousFolders.map((folder) =>
        folder.id === workspaceId ? { ...folder, expanded: true } : folder,
      ),
    );
    setSelectedFolderId(workspaceId);
    setCurrentFileId(matchedFile.id);
    setActiveWorkspaceId(workspaceId);
    resetWorkspacePresentationState();
    return true;
  }

  async function handleOpenWorkspace(workspaceId: string): Promise<void> {
    setOpenWorkspaceMenuId("");

    if (
      isRealtimeActive &&
      activeWorkspaceId &&
      activeWorkspaceId !== workspaceId
    ) {
      await stopRealtimeSession(activeWorkspaceId);
    }

    setIsWorkspaceLoading(true);
    resetWorkspacePresentationState();

    try {
      if (!window.desktopBridge?.openNormalWorkspace) {
        const restored = activateExistingWorkspaceFile(workspaceId);
        if (restored) {
          return;
        }

        throw new Error("桌面工作区桥接未加载，请重启桌面应用后再试。");
      }

      const openedWorkspace = await openNormalModeWorkspace(workspaceId);
      const existingFile = files.find(
        (file) => file.id === openedWorkspace.file.id,
      );
      const nextFile =
        existingFile && existingFile.previewUrl
          ? {
              ...openedWorkspace.file,
              previewUrl: existingFile.previewUrl,
            }
          : openedWorkspace.file;

      if (
        existingFile?.previewUrl &&
        openedWorkspace.file.previewUrl &&
        existingFile.previewUrl !== openedWorkspace.file.previewUrl
      ) {
        URL.revokeObjectURL(openedWorkspace.file.previewUrl);
      }

      setFiles((previousFiles) => upsertFile(previousFiles, nextFile));
      setFolders((previousFolders) =>
        previousFolders.map((folder) =>
          folder.id === openedWorkspace.workspace.workspaceId
            ? { ...folder, expanded: true }
            : folder,
        ),
      );
      setSelectedFolderId(openedWorkspace.workspace.workspaceId);
      setCurrentFileId(nextFile.id);
      setActiveWorkspaceId(openedWorkspace.workspace.workspaceId);
      setSummaries(openedWorkspace.summaries);
      setSummaryEmptyState(openedWorkspace.summaryEmptyState);
      setSummaryWindowDebugState(openedWorkspace.summaryWindowState);
      setExpandedSummaryId("");
      setFullTranscriptText(openedWorkspace.fullTranscriptText);
      setRecentWorkspaces((previousWorkspaces) =>
        upsertWorkspace(previousWorkspaces, openedWorkspace.workspace),
      );

      // 读取页码匹配结果
      const bridge = window.desktopBridge;
      if (bridge?.getWorkspacePageMatch) {
        bridge
          .getWorkspacePageMatch(openedWorkspace.workspace.workspaceId)
          .then((result) => {
            if (result.ok && result.pageMatch?.currentPage != null) {
              setCurrentMatchedPage(result.pageMatch.currentPage);
            }
          })
          .catch(() => {});
      }
    } catch (error) {
      console.error("打开缓存工作区失败", error);
      const message =
        error instanceof Error && error.message
          ? error.message
          : "打开缓存工作区失败，请稍后再试。";
      setWorkspaceErrorMessage(message);
    } finally {
      setIsWorkspaceLoading(false);
    }
  }

  async function handleUploadFile(
    event: ChangeEvent<HTMLInputElement>,
  ): Promise<void> {
    const [file] = event.target.files ?? [];
    if (!file) {
      return;
    }

    try {
      if (isRealtimeActive && activeWorkspaceId) {
        await stopRealtimeSession(activeWorkspaceId);
      }

      const uploaded = await uploadCourseFile(file);
      const workspace = await createNormalModeWorkspace(file, []);

      if (!workspace) {
        throw new Error("当前环境未加载桌面工作区桥接，无法创建缓存工作区。");
      }

      const targetFolderId = workspace.workspaceId;
      const nextFileId = getWorkspaceFileId(workspace.workspaceId);
      const nextFile = {
        ...uploaded,
        id: nextFileId,
        folderId: targetFolderId,
        workspace,
      };

      setFiles((previousFiles) => upsertFile(previousFiles, nextFile));
      setCurrentFileId(nextFile.id);
      setSelectedFolderId(targetFolderId);
      setPipelineStatus(null);
      setFolders((previousFolders) =>
        previousFolders.map((folder) =>
          folder.id === targetFolderId ? { ...folder, expanded: true } : folder,
        ),
      );
      setActiveWorkspaceId(workspace.workspaceId);
      setSelectedFolderId(workspace.workspaceId);
      resetWorkspacePresentationState();
      setRecentWorkspaces((previousWorkspaces) =>
        upsertWorkspace(previousWorkspaces, workspace),
      );
    } catch (error) {
      console.error("上传失败", error);
      setWorkspaceErrorMessage(
        error instanceof Error
          ? error.message
          : "上传或创建缓存工作区失败，请稍后再试。",
      );
    } finally {
      event.target.value = "";
    }
  }

  async function handleSubmitQuestion(): Promise<void> {
    const question = chatInput.trim();
    if (!question || isAsking) {
      return;
    }

    setChatInput("");
    setChatMessages((previousMessages) => [
      ...previousMessages,
      { role: "user", text: question },
    ]);

    setIsAsking(true);

    // 构建多轮对话历史，传给后端
    const conversationHistory: ConversationMessage[] = chatMessages.map(
      (msg) => ({
        role: msg.role,
        content: msg.text,
      }),
    );

    try {
      const response = await askFileQuestion({
        question,
        currentFileName: currentFile?.name || "通用对话",
        workspaceId: activeWorkspace?.workspaceId,
        conversationHistory,
      });

      // 如果回答附带了页码匹配，自动跳转 PDF
      if (response.matchedPage != null && response.matchedPage > 0) {
        setCurrentMatchedPage(response.matchedPage);
      }

      setChatMessages((previousMessages) => [
        ...previousMessages,
        {
          role: "assistant",
          text: response.answer,
          citations: response.citations,
        },
      ]);
    } catch (error) {
      console.error("问答失败", error);
      setChatMessages((previousMessages) => [
        ...previousMessages,
        { role: "assistant", text: "问答出错，请稍后再试。" },
      ]);
    } finally {
      setIsAsking(false);
    }
  }

  function handlePreviewStageClick(): void {
    if (!currentFile) {
      fileInputRef.current?.click();
    }
  }

  function handleResizeStart(side: "left" | "right", pointerX: number): void {
    const frameRect = appFrameRef.current?.getBoundingClientRect();
    if (!frameRect) {
      return;
    }

    setResizeSide(side);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";

    function handlePointerMove(event: PointerEvent): void {
      const currentRect = appFrameRef.current?.getBoundingClientRect();
      if (!currentRect) {
        return;
      }

      if (side === "left") {
        const maxWidth = Math.min(
          MAX_LEFT_PANEL_WIDTH,
          currentRect.width -
            (isRightOpen ? rightPanelWidth : 0) -
            MIN_CENTER_WIDTH,
        );
        const nextWidth = clamp(
          event.clientX - currentRect.left,
          MIN_LEFT_PANEL_WIDTH,
          Math.max(MIN_LEFT_PANEL_WIDTH, maxWidth),
        );
        setLeftPanelWidth(nextWidth);
        return;
      }

      const maxWidth = Math.min(
        MAX_RIGHT_PANEL_WIDTH,
        currentRect.width -
          (isLeftOpen ? leftPanelWidth : 0) -
          MIN_CENTER_WIDTH,
      );
      const nextWidth = clamp(
        currentRect.right - event.clientX,
        MIN_RIGHT_PANEL_WIDTH,
        Math.max(MIN_RIGHT_PANEL_WIDTH, maxWidth),
      );
      setRightPanelWidth(nextWidth);
    }

    function stopResize(): void {
      setResizeSide((current) => (current === side ? null : current));
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", stopResize);
      window.removeEventListener("pointercancel", stopResize);
    }

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", stopResize);
    window.addEventListener("pointercancel", stopResize);

    handlePointerMove(new PointerEvent("pointermove", { clientX: pointerX }));
  }

  const frameStyle = {
    "--left-panel-width": `${leftPanelWidth}px`,
    "--right-panel-width": `${rightPanelWidth}px`,
  } as CSSProperties;

  return (
    <div
      className={`app-frame ${resizeSide ? "is-resizing" : ""}`}
      ref={appFrameRef}
      style={frameStyle}
    >
      <header className="app-topbar">
        <div className="topbar-cluster topbar-left">
          <div className="window-controls-slot" aria-hidden="true" />
          <button
            className="icon-button nav-toggle-button"
            type="button"
            onClick={() => setIsLeftOpen((current) => !current)}
            title={isLeftOpen ? "收起左侧栏" : "展开左侧栏"}
          >
            <PanelToggleIcon mirrored={!isLeftOpen} />
          </button>
        </div>

        <div className="topbar-center">
          <ModeSwitch value={viewMode} onChange={setViewMode} />
        </div>

        <div className="topbar-cluster topbar-right">
          <button
            className="icon-button nav-toggle-button"
            type="button"
            onClick={() => setIsRightOpen((current) => !current)}
            title={isRightOpen ? "收起问答栏" : "展开问答栏"}
          >
            <PanelToggleIcon mirrored={isRightOpen} />
          </button>
        </div>
      </header>

      <div
        className={`app-body ${isLeftOpen ? "left-open" : "left-closed"} ${
          isRightOpen ? "right-open" : "right-closed"
        } ${resizeSide ? "resizing" : ""}`}
      >
        <aside
          className={`sidebar-panel left-panel ${isLeftOpen ? "open" : "closed"}`}
        >
          <div className="sidebar-panel-inner">
            <div className="file-tree-panel sidebar-mode-shell">
              <div
                className={`sidebar-mode-pane ${
                  viewMode === "normal" ? "active" : "inactive"
                }`}
              >
                <SummarySidebar
                  summaries={summaries}
                  expandedSummaryId={expandedSummaryId}
                  emptyState={summaryEmptyState}
                  onToggleSummary={handleToggleSummary}
                  realtimePartialText={
                    isRealtimeActive ? realtimePartialText : ""
                  }
                  fullTranscriptText={fullTranscriptText}
                />
              </div>

              <div
                className={`sidebar-mode-pane ${
                  viewMode === "review" ? "active" : "inactive"
                }`}
              >
                <ReviewSidebar
                  folders={folders}
                  files={files}
                  selectedFolderId={selectedFolderId}
                  currentFileId={currentFileId}
                  onToggleFolder={handleToggleFolder}
                  onSelectFile={handleSelectFile}
                  onRenameWorkspace={(workspaceId) => {
                    void handleRenameWorkspace(workspaceId);
                  }}
                  onDeleteWorkspace={(workspaceId) => {
                    void handleDeleteWorkspace(workspaceId);
                  }}
                  onOpenKnowledgeBase={() => {
                    void handleOpenKnowledgeGraph();
                  }}
                />
              </div>
            </div>
            {isLeftOpen ? (
              <div
                className="panel-resizer panel-resizer-left"
                aria-hidden="true"
                onPointerDown={(event) => {
                  event.preventDefault();
                  handleResizeStart("left", event.clientX);
                }}
              />
            ) : null}
          </div>
        </aside>

        <main className="main-layer">
          <section className="preview-shell">
            <div className="preview-toolbar">
              <div className="preview-toolbar-main">
                <div className="preview-toolbar-context">
                  <span
                    className="panel-badge panel-badge-file"
                    title={currentFile ? currentFile.name : "未选择文件"}
                  >
                    {currentFile ? currentFile.name : "未选择文件"}
                  </span>
                  {activeWorkspace ? (
                    <span
                      className={`panel-badge pipeline-status-badge ${
                        pipelineStatus?.runState === "failed" ? "failed" : ""
                      }`}
                    >
                      {getPipelineStatusLabel(pipelineStatus)}
                    </span>
                  ) : null}
                </div>
                <div className="preview-toolbar-controls">
                  {activeWorkspace ? (
                    <button
                      className="secondary-button preview-action-button"
                      type="button"
                      disabled={
                        isStartingPipeline ||
                        isPipelineRunning ||
                        isRealtimeActive
                      }
                      onClick={() => {
                        void handleStartPipeline();
                      }}
                    >
                      {isStartingPipeline
                        ? "启动中"
                        : isPipelineRunning
                          ? "处理中"
                          : isRealtimeActive
                            ? "监听中"
                            : pipelineStatus?.processingStatus === "paused"
                              ? "继续"
                              : "开始"}
                    </button>
                  ) : null}
                  {isPipelineRunning || isRealtimeActive ? (
                    <button
                      className="secondary-button preview-action-button"
                      type="button"
                      onClick={() => {
                        void handlePausePipeline();
                      }}
                    >
                      暂停
                    </button>
                  ) : null}
                  {currentFile ? (
                    <button
                      className="secondary-button preview-action-button"
                      type="button"
                      onClick={() => {
                        void handleCloseCurrentPreview();
                      }}
                    >
                      关闭
                    </button>
                  ) : null}
                  <button
                    className="icon-button toolbar-icon-button"
                    type="button"
                    onClick={handleOpenSettings}
                    title="打开设置"
                  >
                    <img
                      className="toolbar-icon-image"
                      src={settingsIcon}
                      alt=""
                      aria-hidden="true"
                    />
                  </button>
                </div>
                <input
                  ref={fileInputRef}
                  id="file-input-ts"
                  type="file"
                  accept=".pdf,.ppt,.pptx"
                  hidden
                  onChange={handleUploadFile}
                />
              </div>
            </div>

            {currentFile && inlineWorkspaceStatusMessage ? (
              <div className="workspace-inline-status error">
                {inlineWorkspaceStatusMessage}
              </div>
            ) : null}
            {pptConversionError ? (
              <div className="workspace-inline-status error">
                {pptConversionError}
              </div>
            ) : null}
            {realtimeError && realtimeStatus === "error" ? (
              <div className="workspace-inline-status error">
                🎙 {realtimeError}
              </div>
            ) : null}
            {realtimeStatus === "connecting" && isRealtimeActive ? (
              <div className="workspace-inline-status">
                🎙 正在连接 Deepgram…
              </div>
            ) : null}

            <div className="preview-stage">
              {currentFile && canPreviewCurrentFile ? (
                <PdfPreview
                  file={currentFile}
                  targetPage={currentMatchedPage}
                  onPageChange={(page) => setCurrentMatchedPage(page)}
                />
              ) : currentFile ? (
                <div className="pdf-state">
                  <strong>
                    {currentFile.kind === "pdf"
                      ? "预览暂不可用"
                      : isPptConverting
                        ? "正在转换为 PDF…"
                        : "PPT 文件"}
                  </strong>
                  <p>
                    {currentFile.kind === "pdf"
                      ? "当前文件还没有可用的本地预览地址。"
                      : isPptConverting
                        ? "LibreOffice 正在将课件转换为 PDF，转换完成后会自动显示预览。"
                        : "PPT / PPTX 需要先转换为 PDF 才能预览。"}
                  </p>
                  {currentFile.kind !== "pdf" && !isPptConverting && (
                    <button
                      className="secondary-button"
                      type="button"
                      onClick={() => { void handleConvertPptx(); }}
                      style={{ marginTop: 12 }}
                    >
                      转换为 PDF 预览（需要 LibreOffice）
                    </button>
                  )}
                </div>
              ) : (
                <div className="workspace-home">
                  <button
                    className="upload-card workspace-upload-card"
                    type="button"
                    onClick={handlePreviewStageClick}
                  >
                    <div className="upload-icon">FILE</div>
                    <div className="upload-title">上传新课件</div>
                    <div className="upload-copy">
                      real-time 模式下首次上传会在软件缓存里自动创建工作区副本。
                    </div>
                  </button>

                  <div className="workspace-home-panel">
                    <div className="workspace-home-heading">
                      <span className="workspace-home-title">继续上次课程</span>
                      <span className="workspace-home-meta">
                        {recentWorkspaces.length
                          ? `${recentWorkspaces.length} 个缓存工作区`
                          : "还没有可恢复的工作区"}
                      </span>
                    </div>

                    {workspaceErrorMessage ? (
                      <div className="workspace-home-status error">
                        {workspaceErrorMessage}
                      </div>
                    ) : null}

                    {isWorkspaceLoading ? (
                      <div className="workspace-home-status">
                        正在读取缓存工作区…
                      </div>
                    ) : recentWorkspaces.length ? (
                      <div className="workspace-home-list">
                        {recentWorkspaces.map((workspace) => (
                          <article
                            className={`workspace-home-item ${
                              activeWorkspaceId === workspace.workspaceId
                                ? "active"
                                : ""
                            } ${workspace.starred ? "starred" : ""}`}
                            key={workspace.workspaceId}
                          >
                            <button
                              className="workspace-home-item-trigger"
                              type="button"
                              onClick={() => {
                                void handleOpenWorkspace(workspace.workspaceId);
                              }}
                            >
                              <span className="workspace-home-item-main">
                                <span className="workspace-home-item-copy">
                                  <span className="workspace-home-item-name-wrap">
                                    <span className="workspace-home-item-name">
                                      {workspace.workspaceName}
                                    </span>
                                    {workspace.starred ? (
                                      <span
                                        className="workspace-home-item-star"
                                        aria-hidden="true"
                                      >
                                        ★
                                      </span>
                                    ) : null}
                                  </span>
                                  <span className="workspace-home-item-meta">
                                    {workspace.sourceFileName} · 上次打开{" "}
                                    {new Date(
                                      workspace.lastOpenedAtUtc,
                                    ).toLocaleString()}
                                  </span>
                                </span>
                                <span className="workspace-home-item-status">
                                  {workspace.status}
                                </span>
                              </span>
                            </button>
                            <div className="workspace-home-menu-shell folder-menu-shell">
                              <button
                                className={`workspace-home-menu-trigger folder-menu-trigger ${
                                  openWorkspaceMenuId === workspace.workspaceId
                                    ? "active"
                                    : ""
                                }`}
                                type="button"
                                aria-label={`${workspace.workspaceName} 工作区操作`}
                                aria-expanded={
                                  openWorkspaceMenuId === workspace.workspaceId
                                }
                                onClick={(event) =>
                                  handleToggleWorkspaceMenu(
                                    event,
                                    workspace.workspaceId,
                                  )
                                }
                              >
                                ⋯
                              </button>

                              {openWorkspaceMenuId === workspace.workspaceId ? (
                                <div className="workspace-home-menu-popover folder-menu-popover">
                                  <button
                                    className={`folder-menu-item ${
                                      workspace.starred ? "is-starred" : ""
                                    }`}
                                    type="button"
                                    onClick={() => {
                                      void handleStarWorkspace(
                                        workspace.workspaceId,
                                        !workspace.starred,
                                      );
                                    }}
                                  >
                                    {workspace.starred
                                      ? "取消收藏"
                                      : "加入收藏"}
                                  </button>
                                  <button
                                    className="folder-menu-item danger"
                                    type="button"
                                    onClick={() => {
                                      void handleDeleteWorkspace(
                                        workspace.workspaceId,
                                      );
                                    }}
                                  >
                                    删除
                                  </button>
                                </div>
                              ) : null}
                            </div>
                          </article>
                        ))}
                      </div>
                    ) : (
                      <div className="workspace-home-status">
                        关闭当前课件后，后续可以从这里恢复之前的缓存工作区。
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </section>
        </main>

        <aside
          className={`sidebar-panel right-panel ${isRightOpen ? "open" : "closed"}`}
        >
          <div className="chat-panel-inner">
            <div className="file-tree-panel chat-panel-shell">
              <div className="chat-panel-pane">
                <ChatPanel
                  messages={chatMessages}
                  value={chatInput}
                  isAsking={isAsking}
                  onChange={setChatInput}
                  onSubmit={() => {
                    void handleSubmitQuestion();
                  }}
                />
              </div>
            </div>
          </div>
          {isRightOpen ? (
            <div
              className="panel-resizer panel-resizer-right"
              aria-hidden="true"
              onPointerDown={(event) => {
                event.preventDefault();
                handleResizeStart("right", event.clientX);
              }}
            />
          ) : null}
        </aside>
      </div>

      {isSettingsOpen ? (
        <div className="modal-overlay" onClick={() => setIsSettingsOpen(false)}>
          <div
            className="modal-content"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="modal-header">
              <h3 className="modal-title">设置</h3>
              <button
                className="icon-button"
                type="button"
                onClick={() => setIsSettingsOpen(false)}
              >
                ✕
              </button>
            </div>

            <div className="modal-body">
              <div className="settings-row">
                <span>主题模式</span>
                <select
                  value={theme}
                  onChange={(event) =>
                    setTheme(event.target.value as ThemeMode)
                  }
                >
                  <option value="light">浅色</option>
                  <option value="dark">深色</option>
                </select>
              </div>

              <div className="settings-section">
                <div className="settings-section-heading">
                  <div className="settings-section-title">LiteSynphonia</div>
                  <div className="settings-section-copy">
                    Deepgram 负责转录，DeepSeek 负责总结和问答，SiliconFlow
                    BGE 负责 PDF 页码匹配。
                  </div>
                </div>

                <div className="settings-field">
                  <label className="settings-label" htmlFor="deepgram-api-key">
                    Deepgram API Key
                  </label>
                  <input
                    id="deepgram-api-key"
                    className="settings-input"
                    type="password"
                    autoComplete="off"
                    spellCheck={false}
                    placeholder="请输入 Deepgram API Key"
                    value={providerSettings.deepgramApiKey}
                    onChange={(event) =>
                      setProviderSettings((currentSettings) => ({
                        ...currentSettings,
                        deepgramApiKey: event.target.value,
                      }))
                    }
                  />
                </div>

                <div className="settings-field">
                  <label className="settings-label" htmlFor="deepseek-api-key">
                    DeepSeek API Key
                  </label>
                  <input
                    id="deepseek-api-key"
                    className="settings-input"
                    type="password"
                    autoComplete="off"
                    spellCheck={false}
                    placeholder="请输入 DeepSeek API Key"
                    value={providerSettings.deepseekApiKey}
                    onChange={(event) =>
                      setProviderSettings((currentSettings) => ({
                        ...currentSettings,
                        deepseekApiKey: event.target.value,
                      }))
                    }
                  />
                </div>

                <div className="settings-field">
                  <label
                    className="settings-label"
                    htmlFor="siliconflow-api-key"
                  >
                    SiliconFlow API Key
                  </label>
                  <input
                    id="siliconflow-api-key"
                    className="settings-input"
                    type="password"
                    autoComplete="off"
                    spellCheck={false}
                    placeholder="请输入 SiliconFlow API Key"
                    value={providerSettings.siliconflowApiKey}
                    onChange={(event) =>
                      setProviderSettings((currentSettings) => ({
                        ...currentSettings,
                        siliconflowApiKey: event.target.value,
                      }))
                    }
                  />
                </div>

                <div className="settings-field">
                  <label className="settings-label">转录语言</label>
                  <div className="settings-lang-toggle">
                    {(
                      [
                        { value: "zh-CN", label: "中文" },
                        { value: "en-US", label: "英文" },
                      ] as const
                    ).map(({ value, label }) => (
                      <button
                        key={value}
                        type="button"
                        className={`settings-lang-btn ${
                          providerSettings.transcriptionLanguage === value
                            ? "active"
                            : ""
                        }`}
                        onClick={() =>
                          setProviderSettings((s) => ({
                            ...s,
                            transcriptionLanguage: value,
                          }))
                        }
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="settings-meta">
                  <span>
                    配置文件: {providerSettings.configPath || "尚未创建"}
                  </span>
                  <span>
                    状态:{" "}
                    {providerSettings.hasTranscriptionProvider &&
                    providerSettings.hasSummarizationProvider &&
                    providerSettings.hasEmbeddingProvider
                      ? "已就绪"
                      : "未完成"}
                  </span>
                </div>

                {isProviderSettingsLoading ? (
                  <div className="settings-feedback">
                    正在读取 LiteSynphonia 配置…
                  </div>
                ) : null}

                {providerSettingsFeedback ? (
                  <div
                    className={`settings-feedback ${
                      providerSettingsFeedbackTone
                        ? `is-${providerSettingsFeedbackTone}`
                        : ""
                    }`}
                  >
                    {providerSettingsFeedback}
                  </div>
                ) : null}

                <div className="settings-actions">
                  <button
                    className="secondary-button"
                    type="button"
                    disabled={isProviderSettingsSaving}
                    onClick={() => {
                      void handleSaveProviderSettings();
                    }}
                  >
                    {isProviderSettingsSaving ? "保存中" : "保存 API Key"}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {/* ── 知识库面板 ── */}
      {isKnowledgeGraphOpen ? (
        <KnowledgeGraphPanel
          data={knowledgeGraphData}
          isLoading={isKnowledgeGraphLoading}
          onClose={() => setIsKnowledgeGraphOpen(false)}
          onActivityOpen={(activityId) => {
            void handleOpenWorkspace(activityId);
            setIsKnowledgeGraphOpen(false);
          }}
        />
      ) : null}
    </div>
  );
}
