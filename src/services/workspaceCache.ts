import type {
  CourseFile,
  CourseFileKind,
  NormalSidebarStateItem,
  NormalSidebarStateSnapshot,
  RenameNormalWorkspacePayload,
  StarNormalWorkspacePayload,
  StartNormalWorkspacePipelinePayload,
  SummaryCard,
  SummaryEmptyState,
  SummaryWindowDebugState,
  WorkspaceCache,
  WorkspacePipelineStatus,
} from "../types";
import { detectCourseFileKind } from "../utils/courseFiles";

function buildSidebarStateSnapshot(
  summaries: ReadonlyArray<SummaryCard>,
): NormalSidebarStateSnapshot {
  const now = new Date().toISOString();

  return {
    schemaVersion: "1.0",
    mode: "normal",
    status: "initialized",
    createdAtUtc: now,
    updatedAtUtc: now,
    items: summaries.map((item) => ({
      id: item.id,
      summary: item.summary,
      transcript: item.transcript,
      transcriptSegmentIds: [],
      transcriptRange: {
        startTime: null,
        endTime: null,
      },
    })),
  };
}

function buildSummaryCards(
  items: ReadonlyArray<NormalSidebarStateItem>,
): SummaryCard[] {
  return items.map((item) => ({
    id: item.id,
    summary: item.summary,
    transcript: item.transcript,
  }));
}

function getSourceMimeType(
  sourceKind: CourseFileKind,
  mimeType: string,
): string {
  if (mimeType) {
    return mimeType;
  }

  if (sourceKind === "pdf") {
    return "application/pdf";
  }

  if (sourceKind === "pptx") {
    return "application/vnd.openxmlformats-officedocument.presentationml.presentation";
  }

  if (sourceKind === "ppt") {
    return "application/vnd.ms-powerpoint";
  }

  return "application/octet-stream";
}

function createWorkspacePreviewUrl(
  workspace: WorkspaceCache,
  sourceFile: { mimeType: string; bytes: Uint8Array },
): string {
  if (workspace.sourceKind !== "pdf" || !sourceFile.bytes.byteLength) {
    return "";
  }

  const pdfBytes = sourceFile.bytes.slice();
  const pdfBlob = new Blob([pdfBytes.buffer as ArrayBuffer], {
    type: getSourceMimeType(workspace.sourceKind, sourceFile.mimeType),
  });
  return URL.createObjectURL(pdfBlob);
}

function buildWorkspaceCourseFile(
  workspace: WorkspaceCache,
  sourceFile?: { fileName: string; mimeType: string; bytes: Uint8Array },
): CourseFile {
  const mimeType = sourceFile?.mimeType || "";
  const kind = detectCourseFileKind(workspace.sourceFileName, mimeType);

  return {
    id: `workspace-${workspace.workspaceId}`,
    folderId: workspace.workspaceId,
    name: workspace.sourceFileName,
    previewUrl: sourceFile
      ? createWorkspacePreviewUrl(workspace, sourceFile)
      : "",
    sourceBytes: sourceFile?.bytes,
    kind,
    type: getSourceMimeType(kind, mimeType),
    size: sourceFile?.bytes.byteLength,
    workspace,
  };
}

/**
 * Create the cache-backed workspace used by normal mode.
 * The renderer only prepares the binary payload and the current sidebar state;
 * all filesystem writes are delegated to the Electron main process.
 */
export async function createNormalModeWorkspace(
  file: File,
  summaries: ReadonlyArray<SummaryCard>,
): Promise<WorkspaceCache | null> {
  if (!window.desktopBridge?.createNormalWorkspace) {
    return null;
  }

  const bytes = new Uint8Array(await file.arrayBuffer());
  const result = await window.desktopBridge.createNormalWorkspace({
    fileName: file.name,
    mimeType: file.type,
    bytes,
    sidebarState: buildSidebarStateSnapshot(summaries),
  });

  if (!result.ok || !result.workspace) {
    throw new Error(result.message || "缓存工作区创建失败。");
  }

  return result.workspace;
}

export async function listNormalModeWorkspaces(): Promise<WorkspaceCache[]> {
  if (!window.desktopBridge?.listNormalWorkspaces) {
    return [];
  }

  const result = await window.desktopBridge.listNormalWorkspaces();
  if (!result.ok) {
    throw new Error(result.message || "读取缓存工作区列表失败。");
  }

  return result.workspaces;
}

export async function openNormalModeWorkspace(workspaceId: string): Promise<{
  workspace: WorkspaceCache;
  file: CourseFile;
  summaries: SummaryCard[];
  summaryEmptyState?: SummaryEmptyState;
  summaryWindowState?: SummaryWindowDebugState;
  fullTranscriptText: string;
}> {
  if (!window.desktopBridge?.openNormalWorkspace) {
    throw new Error("当前环境不支持打开缓存工作区。");
  }

  const result = await window.desktopBridge.openNormalWorkspace(workspaceId);
  if (!result.ok || !result.workspace) {
    throw new Error(result.message || "打开缓存工作区失败。");
  }

  return {
    workspace: result.workspace,
    file: buildWorkspaceCourseFile(result.workspace, result.sourceFile),
    summaries: buildSummaryCards(result.sidebarState?.items ?? []),
    summaryEmptyState: result.summaryEmptyState,
    summaryWindowState:
      result.summaryWindowState ?? result.sidebarState?.summaryWindow,
    fullTranscriptText: String(
      result.transcriptState?.finalTranscriptText ??
        result.sidebarState?.finalTranscriptText ??
        "",
    ).trim(),
  };
}

export async function renameNormalModeWorkspace(
  payload: RenameNormalWorkspacePayload,
): Promise<WorkspaceCache> {
  if (!window.desktopBridge?.renameNormalWorkspace) {
    throw new Error("当前环境不支持重命名缓存工作区。");
  }

  const result = await window.desktopBridge.renameNormalWorkspace(payload);
  if (!result.ok || !result.workspace) {
    throw new Error(result.message || "重命名缓存工作区失败。");
  }

  return result.workspace;
}

export async function starNormalModeWorkspace(
  payload: StarNormalWorkspacePayload,
): Promise<WorkspaceCache> {
  if (!window.desktopBridge?.starNormalWorkspace) {
    throw new Error("当前环境不支持收藏缓存工作区。");
  }

  const result = await window.desktopBridge.starNormalWorkspace(payload);
  if (!result.ok || !result.workspace) {
    throw new Error(result.message || "更新工作区收藏状态失败。");
  }

  return result.workspace;
}

export async function deleteNormalModeWorkspace(
  workspaceId: string,
): Promise<string> {
  if (!window.desktopBridge?.deleteNormalWorkspace) {
    throw new Error("当前环境不支持删除缓存工作区。");
  }

  const result = await window.desktopBridge.deleteNormalWorkspace(workspaceId);
  if (!result.ok || !result.workspaceId) {
    throw new Error(result.message || "删除缓存工作区失败。");
  }

  return result.workspaceId;
}

export async function startNormalModeWorkspacePipeline(
  payload: StartNormalWorkspacePipelinePayload,
): Promise<WorkspacePipelineStatus> {
  if (!window.desktopBridge?.startNormalWorkspacePipeline) {
    throw new Error("当前环境不支持启动 LiteSynphonia 流水线。");
  }

  const result =
    await window.desktopBridge.startNormalWorkspacePipeline(payload);
  if (!result.ok || !result.status) {
    throw new Error(result.message || "启动工作区转录失败。");
  }

  return result.status;
}

export async function pauseNormalModeWorkspacePipeline(
  workspaceId: string,
): Promise<WorkspacePipelineStatus> {
  if (!window.desktopBridge?.pauseNormalWorkspacePipeline) {
    throw new Error("当前环境不支持暂停 LiteSynphonia 流水线。");
  }

  const result =
    await window.desktopBridge.pauseNormalWorkspacePipeline(workspaceId);
  if (!result.ok || !result.status) {
    throw new Error(result.message || "暂停工作区转录失败。");
  }

  return result.status;
}

export async function getNormalModeWorkspacePipelineStatus(
  workspaceId: string,
): Promise<WorkspacePipelineStatus> {
  if (!window.desktopBridge?.getNormalWorkspacePipelineStatus) {
    throw new Error("当前环境不支持读取工作区转录状态。");
  }

  const result =
    await window.desktopBridge.getNormalWorkspacePipelineStatus(workspaceId);
  if (!result.ok || !result.status) {
    throw new Error(result.message || "读取工作区转录状态失败。");
  }

  return result.status;
}
