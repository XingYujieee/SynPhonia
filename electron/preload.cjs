const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("desktopBridge", {
  isDesktop: true,
  platform: process.platform,
  createDocumentsWorkspace(folderName) {
    return ipcRenderer.invoke("workspace:create-documents-folder", folderName);
  },
  renameDocumentsWorkspace(payload) {
    return ipcRenderer.invoke("workspace:rename-documents-folder", payload);
  },
  pickExistingFolder() {
    return ipcRenderer.invoke("workspace:pick-existing-folder");
  },
  revealWorkspaceFolder(payload) {
    return ipcRenderer.invoke("workspace:reveal-folder", payload);
  },
  createNormalWorkspace(payload) {
    return ipcRenderer.invoke("workspace:create-normal-workspace", payload);
  },
  listNormalWorkspaces() {
    return ipcRenderer.invoke("workspace:list-normal-workspaces");
  },
  openNormalWorkspace(workspaceId) {
    return ipcRenderer.invoke("workspace:open-normal-workspace", workspaceId);
  },
  renameNormalWorkspace(payload) {
    return ipcRenderer.invoke("workspace:rename-normal-workspace", payload);
  },
  starNormalWorkspace(payload) {
    return ipcRenderer.invoke("workspace:star-normal-workspace", payload);
  },
  deleteNormalWorkspace(workspaceId) {
    return ipcRenderer.invoke("workspace:delete-normal-workspace", workspaceId);
  },
  startNormalWorkspacePipeline(payload) {
    return ipcRenderer.invoke("workspace:start-normal-pipeline", payload);
  },
  pauseNormalWorkspacePipeline(workspaceId) {
    return ipcRenderer.invoke("workspace:pause-normal-pipeline", workspaceId);
  },
  getNormalWorkspacePipelineStatus(workspaceId) {
    return ipcRenderer.invoke(
      "workspace:get-normal-pipeline-status",
      workspaceId,
    );
  },
  getLiteSynphoniaProviderSettings() {
    return ipcRenderer.invoke("settings:get-lite-synphonia-provider-settings");
  },
  saveLiteSynphoniaProviderSettings(payload) {
    return ipcRenderer.invoke(
      "settings:save-lite-synphonia-provider-settings",
      payload,
    );
  },
  askCourseQuestion(payload) {
    return ipcRenderer.invoke("llm:ask-course-question", payload);
  },

  startRealtimeTranscription(payload) {
    return ipcRenderer.invoke(
      "workspace:start-realtime-transcription",
      payload,
    );
  },
  pushRealtimeAudioChunk(payload) {
    return ipcRenderer.invoke("workspace:push-realtime-audio-chunk", payload);
  },
  stopRealtimeTranscription(workspaceId) {
    return ipcRenderer.invoke(
      "workspace:stop-realtime-transcription",
      workspaceId,
    );
  },
  onRealtimeTranscriptionEvent(callback) {
    const handler = (_event, data) => callback(data);
    ipcRenderer.on("realtime-transcription-event", handler);
    return () => {
      ipcRenderer.removeListener("realtime-transcription-event", handler);
    };
  },

  // ── 页码匹配 ──────────────────────────────────────────
  getWorkspacePageMatch(workspaceId) {
    return ipcRenderer.invoke("workspace:get-page-match", workspaceId);
  },

  // ── PPT/PPTX → PDF 转换 ───────────────────────────────
  convertPptxToPdf(payload) {
    return ipcRenderer.invoke("workspace:convert-pptx-to-pdf", payload);
  },

  // ── 知识图谱 ──────────────────────────────────────────
  getKnowledgeBaseData(payload) {
    return ipcRenderer.invoke("knowledge:get-data", payload);
  },
  getKnowledgeGraph() {
    return ipcRenderer.invoke("knowledge:get-graph");
  },
});
