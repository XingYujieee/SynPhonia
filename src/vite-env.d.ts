/// <reference types="vite/client" />

import type {
  AskFileQuestionPayload,
  AskFileQuestionResult,
  CreateNormalWorkspacePayload,
  CreateNormalWorkspaceResult,
  KnowledgeBaseData,
  ListNormalWorkspacesResult,
  LiteSynphoniaProviderSettingsResult,
  OpenNormalWorkspaceResult,
  PushRealtimeAudioChunkPayload,
  RealtimeTranscriptionEvent,
  RenameNormalWorkspacePayload,
  SaveLiteSynphoniaProviderSettingsPayload,
  StarNormalWorkspacePayload,
  StartNormalWorkspacePipelinePayload,
  StartRealtimeTranscriptionPayload,
  WorkspaceMutationResult,
  WorkspacePipelineStatusResult,
} from "./types";

interface PageMatchResult {
  ok: boolean;
  message?: string;
  pageMatch?: {
    timeline: Array<{ pageIndex: number; startTime: number; endTime: number }>;
    segmentMatches: Array<{ pageIndex: number; confidence: number }>;
    currentPage: number | null;
  } | null;
}

interface PptxConversionIpcResult {
  ok: boolean;
  message?: string;
  pdfBytes?: number[];
  workspace?: import("./types").WorkspaceCache;
}

interface KnowledgeGraphIpcResult {
  ok: boolean;
  message?: string;
  graph?: {
    nodes: Array<Record<string, unknown>>;
    edges: Array<Record<string, unknown>>;
  };
}

interface KnowledgeBaseDataIpcResult {
  ok: boolean;
  message?: string;
  data?: KnowledgeBaseData;
  ingestResults?: Array<Record<string, unknown>>;
}

interface DesktopBridge {
  isDesktop: boolean;
  platform: string;
  createDocumentsWorkspace(folderName: string): Promise<unknown>;
  renameDocumentsWorkspace(payload: unknown): Promise<unknown>;
  pickExistingFolder(): Promise<unknown>;
  revealWorkspaceFolder(payload: unknown): Promise<unknown>;
  createNormalWorkspace(
    payload: CreateNormalWorkspacePayload,
  ): Promise<CreateNormalWorkspaceResult>;
  listNormalWorkspaces(): Promise<ListNormalWorkspacesResult>;
  openNormalWorkspace(workspaceId: string): Promise<OpenNormalWorkspaceResult>;
  renameNormalWorkspace(
    payload: RenameNormalWorkspacePayload,
  ): Promise<WorkspaceMutationResult>;
  starNormalWorkspace(
    payload: StarNormalWorkspacePayload,
  ): Promise<WorkspaceMutationResult>;
  deleteNormalWorkspace(workspaceId: string): Promise<WorkspaceMutationResult>;
  startNormalWorkspacePipeline(
    payload: StartNormalWorkspacePipelinePayload,
  ): Promise<WorkspacePipelineStatusResult>;
  pauseNormalWorkspacePipeline(
    workspaceId: string,
  ): Promise<WorkspacePipelineStatusResult>;
  getNormalWorkspacePipelineStatus(
    workspaceId: string,
  ): Promise<WorkspacePipelineStatusResult>;
  getLiteSynphoniaProviderSettings(): Promise<LiteSynphoniaProviderSettingsResult>;
  saveLiteSynphoniaProviderSettings(
    payload: SaveLiteSynphoniaProviderSettingsPayload,
  ): Promise<LiteSynphoniaProviderSettingsResult>;
  askCourseQuestion(
    payload: AskFileQuestionPayload,
  ): Promise<AskFileQuestionResult>;

  startRealtimeTranscription(
    payload: StartRealtimeTranscriptionPayload,
  ): Promise<WorkspacePipelineStatusResult>;
  pushRealtimeAudioChunk(payload: PushRealtimeAudioChunkPayload): Promise<void>;
  stopRealtimeTranscription(
    workspaceId: string,
  ): Promise<WorkspacePipelineStatusResult>;
  onRealtimeTranscriptionEvent(
    callback: (event: RealtimeTranscriptionEvent) => void,
  ): () => void;

  // ── 新增功能 ──────────────────────────────────────────
  getWorkspacePageMatch(workspaceId: string): Promise<PageMatchResult>;
  convertPptxToPdf(payload: {
    workspaceId: string;
  }): Promise<PptxConversionIpcResult>;
  getKnowledgeBaseData(payload?: {
    selectedActivityId?: string;
  }): Promise<KnowledgeBaseDataIpcResult>;
  getKnowledgeGraph(): Promise<KnowledgeGraphIpcResult>;
}

declare global {
  interface Window {
    desktopBridge?: DesktopBridge;
  }
}

export {};
