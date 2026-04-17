export type ThemeMode = "light" | "dark";
export type ViewMode = "normal" | "review";
export type ChatRole = "assistant" | "user";
export type CourseFileKind = "pdf" | "ppt" | "pptx" | "other";
export type WorkspaceProcessingStatus =
  | "initialized"
  | "idle"
  | "connecting"
  | "listening"
  | "streaming"
  | "partial_result"
  | "final_result"
  | "error"
  | "recording"
  | "transcribing"
  | "summarizing"
  | "matching"
  | "ready"
  | "paused"
  | "failed";
export type WorkspaceRunState =
  | "idle"
  | "running"
  | "succeeded"
  | "failed"
  | "error";
export type RealtimeChunkDurationMs = 50 | 100 | 150 | 200 | 250 | 300;
export type RealtimeTranscriptEventType =
  | "status"
  | "partial"
  | "final"
  | "summary"
  | "page_match"
  | "error"
  | "stopped";

export interface WorkspaceFolder {
  id: string;
  name: string;
  expanded: boolean;
  createdAtUtc?: string;
}

export interface CourseFile {
  id: string;
  folderId: string;
  name: string;
  previewUrl: string;
  sourceBytes?: Uint8Array;
  kind: CourseFileKind;
  type?: string;
  size?: number;
  extractedText?: string;
  workspace?: WorkspaceCache;
}

export interface ChatMessage {
  role: ChatRole;
  text: string;
  citations?: string[];
}

export interface SummaryCard {
  id: string;
  summary: string;
  transcript: string;
}

export interface TranscriptSegment {
  id: string;
  text: string;
  finalizedAtUtc: string;
}

export interface WorkspaceArtifacts {
  transcriptPath: string;
  summaryPath: string;
  sourceCopyPath: string;
  sidebarStatePath: string;
}

export interface WorkspaceCache {
  workspaceId: string;
  workspaceName: string;
  workspacePath: string;
  cacheRootPath: string;
  sourceFileName: string;
  sourceKind: CourseFileKind;
  starred?: boolean;
  status: WorkspaceProcessingStatus;
  createdAtUtc: string;
  updatedAtUtc: string;
  lastOpenedAtUtc: string;
  artifacts: WorkspaceArtifacts;
}

export interface NormalSidebarStateItem {
  id: string;
  summary: string;
  transcript: string;
  transcriptSegmentIds: string[];
  transcriptRange: {
    startTime: number | null;
    endTime: number | null;
  };
}

export interface SummaryWindowDebugState {
  triggerChars: number;
  overlapChars: number;
  pendingTranscript: string;
  pendingChars: number;
  generatedCount: number;
}

export interface RealtimeTranscriptSidebarState {
  schemaVersion: "2.0";
  mode: "normal";
  status: WorkspaceProcessingStatus;
  createdAtUtc: string;
  updatedAtUtc: string;
  items: NormalSidebarStateItem[];
  summaryWindow?: SummaryWindowDebugState;
  partialText: string;
  finalTranscriptText: string;
  finalSegments: TranscriptSegment[];
  streamConfig: {
    chunkDurationMs: RealtimeChunkDurationMs;
    sampleRate: number;
    channels: 1;
    encoding: "linear16";
  };
  lastError: string;
}

export interface NormalSidebarStateSnapshot {
  schemaVersion: "1.0" | "2.0";
  mode: "normal";
  status: WorkspaceProcessingStatus;
  createdAtUtc: string;
  updatedAtUtc: string;
  items: NormalSidebarStateItem[];
  summaryWindow?: SummaryWindowDebugState;
  partialText?: string;
  finalTranscriptText?: string;
  finalSegments?: TranscriptSegment[];
  streamConfig?: {
    chunkDurationMs: RealtimeChunkDurationMs;
    sampleRate: number;
    channels: 1;
    encoding: "linear16";
  };
  lastError?: string;
}

export interface SummaryEmptyState {
  title: string;
  copy: string;
  transcriptPreview?: string;
}

export interface TranscriptEmptyState {
  title: string;
  copy: string;
}

export interface CreateNormalWorkspacePayload {
  fileName: string;
  mimeType?: string;
  bytes: Uint8Array;
  sidebarState: NormalSidebarStateSnapshot;
}

export interface CreateNormalWorkspaceResult {
  ok: boolean;
  message?: string;
  workspace?: WorkspaceCache;
}

export interface ListNormalWorkspacesResult {
  ok: boolean;
  message?: string;
  workspaces: WorkspaceCache[];
}

export interface WorkspaceSourceSnapshot {
  fileName: string;
  mimeType: string;
  bytes: Uint8Array;
}

export interface OpenNormalWorkspaceResult {
  ok: boolean;
  message?: string;
  workspace?: WorkspaceCache;
  sidebarState?: NormalSidebarStateSnapshot;
  summaryEmptyState?: SummaryEmptyState;
  summaryWindowState?: SummaryWindowDebugState;
  transcriptState?: RealtimeTranscriptSidebarState;
  transcriptEmptyState?: TranscriptEmptyState;
  sourceFile?: WorkspaceSourceSnapshot;
}

export interface RenameNormalWorkspacePayload {
  workspaceId: string;
  workspaceName: string;
}

export interface StarNormalWorkspacePayload {
  workspaceId: string;
  starred: boolean;
}

export interface WorkspaceMutationResult {
  ok: boolean;
  message?: string;
  workspace?: WorkspaceCache;
  workspaceId?: string;
}

export interface StartNormalWorkspacePipelinePayload {
  workspaceId: string;
  chunkDurationMs?: RealtimeChunkDurationMs;
}

export interface WorkspacePipelineStatus {
  workspaceId: string;
  runState: WorkspaceRunState;
  processingStatus: WorkspaceProcessingStatus;
  startedAtUtc?: string;
  finishedAtUtc?: string;
  message?: string;
  exitCode?: number;
  completedCycles?: number;
}

export interface StartRealtimeTranscriptionPayload {
  workspaceId: string;
  chunkDurationMs: RealtimeChunkDurationMs;
  sampleRate?: number;
  keywords?: string[];
}

export interface PushRealtimeAudioChunkPayload {
  workspaceId: string;
  audio: Uint8Array;
  sampleRate: number;
  chunkDurationMs: RealtimeChunkDurationMs;
  rms: number;
  hasSpeech: boolean;
}

export interface RealtimeTranscriptionEvent {
  workspaceId: string;
  type: RealtimeTranscriptEventType;
  status: WorkspaceProcessingStatus;
  message?: string;
  partialText?: string;
  finalSegment?: TranscriptSegment;
  finalTranscriptText?: string;
  matchedPage?: number;
  lastError?: string;
}

export interface WorkspacePipelineStatusResult {
  ok: boolean;
  message?: string;
  status?: WorkspacePipelineStatus;
}

export type TranscriptionLanguage = "zh-CN" | "en-US";

export interface LiteSynphoniaProviderSettings {
  configPath: string;
  deepgramApiKey: string;
  deepseekApiKey: string;
  siliconflowApiKey: string;
  transcriptionLanguage: TranscriptionLanguage;
  hasTranscriptionProvider: boolean;
  hasSummarizationProvider: boolean;
  hasEmbeddingProvider: boolean;
}

export interface SaveLiteSynphoniaProviderSettingsPayload {
  deepgramApiKey: string;
  deepseekApiKey: string;
  siliconflowApiKey: string;
  transcriptionLanguage?: TranscriptionLanguage;
}

export interface LiteSynphoniaProviderSettingsResult {
  ok: boolean;
  message?: string;
  settings?: LiteSynphoniaProviderSettings;
}

export interface ConversationMessage {
  role: "user" | "assistant";
  content: string;
}

export interface AskFileQuestionPayload {
  question: string;
  currentFileName: string;
  workspaceId?: string;
  conversationHistory?: ConversationMessage[];
}

export interface AskFileQuestionResult {
  answer: string;
  citations: string[];
  matchedPage?: number;
}

export interface PageMatchEntry {
  pageIndex: number;
  confidence: number;
  startTime?: number;
  endTime?: number;
}

export interface KnowledgeBaseRelationDetail {
  relation_id: string;
  source_activity_id: string;
  target_activity_id: string;
  strength: string;
  state: string;
  reasons: string[];
  source_type: string;
}

export interface KnowledgeBaseFileEntry {
  file_type: string;
  label: string;
  path: string;
  exists: boolean;
  preview_mode: "inline_text" | "external_only";
  preview_text?: string;
  ppt_id?: string;
}

export interface KnowledgeBaseContentLineActivity {
  activity_id: string;
  title: string;
  date: string;
  start_time: string;
  end_time: string;
  summary: string;
  keywords: string[];
}

export interface KnowledgeBaseContentLine {
  content_line_id: string;
  title: string;
  activity_count: number;
  activities: KnowledgeBaseContentLineActivity[];
}

export interface KnowledgeBaseActivity {
  activity_id: string;
  title: string;
  activity_name: string;
  activity_intro: string;
  scene_type?: string | null;
  start_time: string;
  end_time: string;
  duration_minutes: number;
  transcript_text: string;
  summary_text: string;
  summary_of_summary: string;
  keywords: string[];
  keywords_of_keywords: string[];
  ppt_present: boolean;
  activity_dir?: string | null;
  transcript_file_path?: string | null;
  summary_file_path?: string | null;
  ppt_file_path?: string | null;
  ppt_id?: string | null;
  ppt_text_excerpt?: string | null;
  matched_slides: Array<Record<string, unknown>>;
  transcript_meta: Record<string, unknown>;
  summary_meta: Record<string, unknown>;
  relations: KnowledgeBaseRelationDetail[];
  content_line?: KnowledgeBaseContentLine | null;
  files: KnowledgeBaseFileEntry[];
}

export interface KnowledgeBaseCoreData {
  activities: KnowledgeBaseActivity[];
  selected_activity?: KnowledgeBaseActivity | null;
  content_lines: KnowledgeBaseContentLine[];
  counts: {
    activity_count: number;
    content_line_count: number;
    attachment_count: number;
  };
}

export interface KnowledgeBaseGraphNode {
  node_id: string;
  node_type: string;
  title: string;
  activity_name: string;
  activity_intro: string;
  summary_of_summary: string;
  scene_type?: string | null;
  start_time: string;
  keywords: string[];
  keywords_of_keywords: string[];
  has_ppt: boolean;
}

export interface KnowledgeBaseGraphEdge extends KnowledgeBaseRelationDetail {}

export interface KnowledgeBaseGraphView {
  nodes: KnowledgeBaseGraphNode[];
  edges: KnowledgeBaseGraphEdge[];
}

export interface KnowledgeBaseLegacyBundle {
  navigation: string[];
  history: Record<string, unknown>;
  relation_map: Record<string, unknown>;
  timeline_calendar: Record<string, unknown>;
  timeline_line_view: Record<string, unknown>;
  file_lookup: Record<string, unknown>;
  detail_panel?: Record<string, unknown> | null;
}

export interface KnowledgeBaseData {
  core_data: KnowledgeBaseCoreData;
  graph_view: KnowledgeBaseGraphView;
  legacy_view_bundle?: KnowledgeBaseLegacyBundle;
}

export interface PptConversionResult {
  ok: boolean;
  message?: string;
  pdfBytes?: Uint8Array;
}

export interface ApiConfig {
  baseUrl: string;
  useMock: boolean;
  endpoints: {
    upload: string;
    chat: string;
  };
}
