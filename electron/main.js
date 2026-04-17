import { app, BrowserWindow, dialog, ipcMain, shell } from "electron";
import { randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import { homedir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..");
const devServerUrl = process.env.VITE_DEV_SERVER_URL;
const appEntryHtml = process.env.APP_ENTRY_HTML || "index.html";
const appBuildDir = process.env.APP_BUILD_DIR || "dist";
const WORKSPACE_CACHE_DIRNAME = "workspace-cache";
const WORKSPACE_INDEX_FILE_NAME = "workspace-index.json";
const TRANSCRIPT_FILE_NAME = "transcription.full.json";
const SUMMARY_FILE_NAME = "summary.full.json";
const SIDEBAR_STATE_FILE_NAME = ".normal-mode-state.json";
const LITESYNPHONIA_INTERFACE_FILE_NAME = "interface_output.json";
const LITESYNPHONIA_RUN_LOG_FILE_NAME = "lite_synphonia.run.log";
const LITESYNPHONIA_MERGED_RESULTS_FILE_NAME = "merged_results.json";
const LITESYNPHONIA_REQUIREMENTS_PATH = path.join(
  projectRoot,
  "lite_synphonia",
  "requirements.txt",
);
const LITESYNPHONIA_VENV_DIR = path.join(projectRoot, ".lite_synphonia-venv");
const DEFAULT_RECORD_SECONDS = 20;
const SUMMARY_WINDOW_TRIGGER_CHARS = 200;
const SUMMARY_WINDOW_OVERLAP_CHARS = 20;
const SUMMARY_IDLE_FLUSH_MIN_CHARS = 120;
const SUMMARY_IDLE_TIMEOUT_MS = 8000;
const REALTIME_PDF_MATCH_MIN_SEGMENTS = 1;
const REALTIME_PDF_MATCH_MAX_SEGMENTS = 180;
const REALTIME_PDF_MATCH_DEBOUNCE_MS = 400;
const REALTIME_TRANSCRIPTION_KEYWORD_HINT_LIMIT = 12;
const REALTIME_TRANSCRIPTION_KEYWORD_TEXT_LIMIT = 24000;
const REALTIME_TRANSCRIPTION_KEYWORD_CACHE_FILE_NAME =
  ".pdf.transcription-keywords.json";
const KNOWLEDGE_BASE_V2_WORKSPACE_DIRNAME = "knowledge-base-v2";
const KNOWLEDGE_BASE_V2_ACTIVITY_INPUT_FILE_NAME = "workspace-activities.json";
const KNOWLEDGE_BASE_KEYWORD_LIMIT = 8;
const KNOWLEDGE_BASE_TOPIC_LIMIT = 3;
const REALTIME_PDF_MATCH_RESULT_FILE_NAME = "realtime.results.json";
const REALTIME_PDF_MATCH_TRANSCRIPTION_FILE_NAME =
  ".realtime.transcription.payload.json";
const PROVIDER_CONFIG_PATH = path.join(
  homedir(),
  ".config",
  "lite_synphonia",
  "providers.json",
);
const LEGACY_PROVIDER_CONFIG_PATH = path.join(
  homedir(),
  ".config",
  "mergesyn",
  "providers.json",
);
const APP_MANAGED_PROVIDER_NAMES = new Set([
  "deepgram",
  "deepseek",
  "siliconflow-embed",
]);
const VALID_TRANSCRIPTION_LANGUAGES = new Set(["zh-CN", "en-US"]);
const APP_MANAGED_PROVIDER_PRESETS = {
  deepgram: {
    name: "deepgram",
    baseUrl: "https://api.deepgram.com",
    modelId: "whisper-large",
    services: ["transcription"],
    timeoutSeconds: 60,
    maxRetries: 3,
    temperature: 0.1,
  },
  deepseek: {
    name: "deepseek",
    baseUrl: "https://api.deepseek.com/v1",
    modelId: "deepseek-chat",
    services: ["summarization"],
    timeoutSeconds: 60,
    maxRetries: 3,
    temperature: 0.1,
  },
  "siliconflow-embed": {
    name: "siliconflow-embed",
    baseUrl: "https://api.siliconflow.cn/v1",
    modelId: "BAAI/bge-large-zh-v1.5",
    services: ["embedding"],
    timeoutSeconds: 60,
    maxRetries: 3,
    temperature: 0,
  },
};
const WORKSPACE_STATUSES = new Set([
  "initialized",
  "recording",
  "transcribing",
  "summarizing",
  "matching",
  "ready",
  "paused",
  "failed",
]);
const WORKSPACE_RUN_STATES = new Set([
  "idle",
  "running",
  "succeeded",
  "failed",
]);
const WORKSPACE_INTERNAL_FILE_NAMES = new Set([
  TRANSCRIPT_FILE_NAME,
  SUMMARY_FILE_NAME,
  SIDEBAR_STATE_FILE_NAME,
  WORKSPACE_INDEX_FILE_NAME,
  LITESYNPHONIA_INTERFACE_FILE_NAME,
  LITESYNPHONIA_RUN_LOG_FILE_NAME,
  LITESYNPHONIA_MERGED_RESULTS_FILE_NAME,
  REALTIME_PDF_MATCH_TRANSCRIPTION_FILE_NAME,
]);
const workspacePipelineRuns = new Map();
const workspacePipelineControllers = new Map();
const WEBSOCKET_OPEN = 1;

function getDocumentsDisplayPath(folderName) {
  return `~/Documents/${folderName}`;
}

function getDocumentsFolderPath(folderName) {
  return path.join(app.getPath("documents"), folderName);
}

async function ensureDirectory(folderPath) {
  await fs.mkdir(folderPath, { recursive: true });
}

function getWorkspaceCacheRoot() {
  return path.join(app.getPath("userData"), WORKSPACE_CACHE_DIRNAME);
}

function getWorkspaceIndexPath(cacheRootPath) {
  return path.join(cacheRootPath, WORKSPACE_INDEX_FILE_NAME);
}

function normalizeTranscriptionLanguage(value, fallback = "zh-CN") {
  const normalized = String(value || "").trim();
  return VALID_TRANSCRIPTION_LANGUAGES.has(normalized) ? normalized : fallback;
}

function mapTranscriptionLanguageToSummaryLanguage(language) {
  const normalized = String(language || "").trim().toLowerCase();
  if (normalized.startsWith("en")) {
    return "en";
  }
  if (normalized.startsWith("zh")) {
    return "zh";
  }
  return "auto";
}

function sanitizePathSegment(value) {
  return String(value || "")
    .trim()
    .replace(/[<>:"/\\|?*\u0000-\u001f]/g, " ")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^[-.]+|[-.]+$/g, "")
    .slice(0, 72);
}

function buildWorkspaceFolderName(fileName) {
  const stamp = new Date()
    .toISOString()
    .replace(/[-:]/g, "")
    .replace(/\..+$/, "Z");
  const baseName =
    sanitizePathSegment(path.parse(fileName).name) || "workspace";
  return `${stamp}-${baseName}`;
}

function getDefaultWorkspaceDisplayName(
  sourceFileName,
  fallbackName = "未命名工作区",
) {
  const sourceName = String(sourceFileName || "").trim();
  const fallback = String(fallbackName || "").trim() || "未命名工作区";

  if (!sourceName) {
    return fallback;
  }

  const parsed = path.parse(sourceName);
  return parsed.name || sourceName || fallback;
}

function detectSourceKind(fileName, mimeType = "") {
  const lowerName = String(fileName || "").toLowerCase();
  const lowerType = String(mimeType || "").toLowerCase();

  if (lowerType.includes("pdf") || lowerName.endsWith(".pdf")) {
    return "pdf";
  }

  if (lowerName.endsWith(".pptx")) {
    return "pptx";
  }

  if (lowerName.endsWith(".ppt")) {
    return "ppt";
  }

  return "other";
}

function selectWorkspaceSourceEntry(
  childEntries,
  { sourceFileName = "", sourceKind = "other" } = {},
) {
  const files = childEntries
    .filter((child) => child.isFile())
    .sort((left, right) => left.name.localeCompare(right.name));

  const normalizedSourceFileName = String(sourceFileName || "").trim();
  if (normalizedSourceFileName) {
    const exactMatch = files.find((child) => child.name === normalizedSourceFileName);
    if (exactMatch) {
      return exactMatch;
    }
  }

  const visibleCandidates = files.filter(
    (child) =>
      !child.name.startsWith(".") &&
      !WORKSPACE_INTERNAL_FILE_NAMES.has(child.name),
  );

  if (sourceKind !== "other") {
    const kindMatch = visibleCandidates.find(
      (child) => detectSourceKind(child.name) === sourceKind,
    );
    if (kindMatch) {
      return kindMatch;
    }
  }

  const recognizedCourseFile = visibleCandidates.find(
    (child) => detectSourceKind(child.name) !== "other",
  );
  if (recognizedCourseFile) {
    return recognizedCourseFile;
  }

  return visibleCandidates[0] || null;
}

function toBuffer(binaryPayload) {
  if (binaryPayload instanceof Uint8Array) {
    return Buffer.from(binaryPayload);
  }

  if (Array.isArray(binaryPayload)) {
    return Buffer.from(binaryPayload);
  }

  return Buffer.alloc(0);
}

async function writeJsonFile(destination, payload) {
  const targetPath = path.resolve(destination);
  await ensureDirectory(path.dirname(targetPath));
  const tempPath = `${targetPath}.${process.pid}.${Date.now()}.${randomUUID()}.tmp`;
  const content = JSON.stringify(payload, null, 2);
  await fs.writeFile(tempPath, content, "utf8");
  await fs.rename(tempPath, targetPath);
}

async function pathExists(targetPath) {
  try {
    await fs.access(targetPath);
    return true;
  } catch (_error) {
    return false;
  }
}

async function readJsonFile(sourcePath, fallbackValue) {
  try {
    const raw = await fs.readFile(sourcePath, "utf8");
    return JSON.parse(raw);
  } catch (error) {
    if (error?.code === "ENOENT") {
      return fallbackValue;
    }

    throw error;
  }
}

async function backupCorruptedJsonFile(sourcePath) {
  if (!(await pathExists(sourcePath))) {
    return "";
  }

  const stamp = new Date()
    .toISOString()
    .replace(/[-:.]/g, "")
    .replace(/\..+$/, "Z");
  const backupPath = `${sourcePath}.corrupt-${stamp}`;
  await fs.copyFile(sourcePath, backupPath);
  return backupPath;
}

async function readProviderEntries() {
  const candidatePaths = [PROVIDER_CONFIG_PATH, LEGACY_PROVIDER_CONFIG_PATH];

  for (const candidatePath of candidatePaths) {
    try {
      const payload = await readJsonFile(candidatePath, null);
      const providers = Array.isArray(payload?.providers)
        ? payload.providers
        : Array.isArray(payload)
          ? payload
          : [];

      if (providers.length) {
        return {
          providerPath: candidatePath,
          entries: providers,
        };
      }
    } catch (_error) {
      // Ignore malformed/absent provider files here; the caller gets a single
      // deterministic remediation path instead of low-signal parse noise.
    }
  }

  return {
    providerPath: PROVIDER_CONFIG_PATH,
    entries: [],
  };
}

function normalizeProviderEntry(rawEntry) {
  if (!rawEntry || typeof rawEntry !== "object") {
    return null;
  }

  return {
    name: String(rawEntry.name || "").trim(),
    base_url: String(rawEntry.base_url || rawEntry.baseUrl || "").trim(),
    model_id: String(rawEntry.model_id || rawEntry.model || "").trim(),
    api_key: String(rawEntry.api_key || "").trim(),
    transcription_language: normalizeTranscriptionLanguage(
      rawEntry.transcription_language || rawEntry.transcriptionLanguage,
      "",
    ),
    services: Array.isArray(rawEntry.services)
      ? rawEntry.services
          .map((service) => String(service || "").trim())
          .filter(Boolean)
      : [],
    timeout_seconds: Number(rawEntry.timeout_seconds) || 60,
    max_retries: Number(rawEntry.max_retries) || 3,
    temperature:
      rawEntry.temperature === undefined
        ? 0.1
        : Number(rawEntry.temperature) || 0,
    created_at_utc: String(rawEntry.created_at_utc || "").trim(),
    updated_at_utc: String(rawEntry.updated_at_utc || "").trim(),
  };
}

function createManagedProviderEntry(preset, apiKey, previousEntry) {
  const now = new Date().toISOString();
  return {
    name: preset.name,
    base_url: preset.baseUrl,
    model_id: preset.modelId,
    api_key: apiKey,
    services: [...preset.services],
    timeout_seconds: previousEntry?.timeout_seconds || preset.timeoutSeconds,
    max_retries: previousEntry?.max_retries || preset.maxRetries,
    temperature:
      previousEntry?.temperature === undefined
        ? preset.temperature
        : previousEntry.temperature,
    created_at_utc: previousEntry?.created_at_utc || now,
    updated_at_utc: now,
  };
}

async function getLiteSynphoniaProviderSettingsSnapshot() {
  const { providerPath, entries } = await readProviderEntries();
  const normalizedEntries = entries.map(normalizeProviderEntry).filter(Boolean);
  const deepgramEntry =
    normalizedEntries.find((entry) => entry.name === "deepgram") ||
    normalizedEntries.find(
      (entry) => entry.api_key && entry.services.includes("transcription"),
    );
  const deepseekEntry =
    normalizedEntries.find((entry) => entry.name === "deepseek") ||
    normalizedEntries.find(
      (entry) => entry.api_key && entry.services.includes("summarization"),
    );
  const siliconflowEntry =
    normalizedEntries.find((entry) => entry.name === "siliconflow-embed") ||
    normalizedEntries.find(
      (entry) => entry.api_key && entry.services.includes("embedding"),
    );

  const transcriptionLanguage = normalizeTranscriptionLanguage(
    deepgramEntry?.transcription_language,
    "zh-CN",
  );

  return {
    configPath: providerPath,
    deepgramApiKey: deepgramEntry?.api_key || "",
    deepseekApiKey: deepseekEntry?.api_key || "",
    siliconflowApiKey: siliconflowEntry?.api_key || "",
    transcriptionLanguage,
    hasTranscriptionProvider: normalizedEntries.some(
      (entry) => entry.api_key && entry.services.includes("transcription"),
    ),
    hasSummarizationProvider: normalizedEntries.some(
      (entry) => entry.api_key && entry.services.includes("summarization"),
    ),
    hasEmbeddingProvider: normalizedEntries.some(
      (entry) => entry.api_key && entry.services.includes("embedding"),
    ),
  };
}

async function saveLiteSynphoniaProviderSettings(payload) {
  const deepgramApiKey = String(payload?.deepgramApiKey || "").trim();
  const deepseekApiKey = String(payload?.deepseekApiKey || "").trim();
  const siliconflowApiKey = String(payload?.siliconflowApiKey || "").trim();
  const transcriptionLanguage = normalizeTranscriptionLanguage(
    payload?.transcriptionLanguage,
    "zh-CN",
  );
  const { entries } = await readProviderEntries();
  const normalizedEntries = entries.map(normalizeProviderEntry).filter(Boolean);
  const existingEntriesByName = new Map(
    normalizedEntries.map((entry) => [entry.name, entry]),
  );
  const preservedEntries = normalizedEntries.filter(
    (entry) => !APP_MANAGED_PROVIDER_NAMES.has(entry.name),
  );
  const nextEntries = [...preservedEntries];

  if (deepgramApiKey) {
    const deepgramEntry = createManagedProviderEntry(
      APP_MANAGED_PROVIDER_PRESETS.deepgram,
      deepgramApiKey,
      existingEntriesByName.get("deepgram"),
    );
    deepgramEntry.transcription_language = transcriptionLanguage;
    nextEntries.push(deepgramEntry);
  }

  if (deepseekApiKey) {
    nextEntries.push(
      createManagedProviderEntry(
        APP_MANAGED_PROVIDER_PRESETS.deepseek,
        deepseekApiKey,
        existingEntriesByName.get("deepseek"),
      ),
    );
  }

  if (siliconflowApiKey) {
    nextEntries.push(
      createManagedProviderEntry(
        APP_MANAGED_PROVIDER_PRESETS["siliconflow-embed"],
        siliconflowApiKey,
        existingEntriesByName.get("siliconflow-embed"),
      ),
    );
  }

  const sortedEntries = nextEntries.sort((left, right) =>
    left.name.localeCompare(right.name, "zh-Hans-CN"),
  );
  await ensureDirectory(path.dirname(PROVIDER_CONFIG_PATH));
  await writeJsonFile(PROVIDER_CONFIG_PATH, {
    providers: sortedEntries,
    updated_at_utc: new Date().toISOString(),
  });

  return getLiteSynphoniaProviderSettingsSnapshot();
}

function takeTailText(text, maxChars) {
  const normalized = String(text || "").trim();
  if (!normalized || normalized.length <= maxChars) {
    return normalized;
  }

  return normalized.slice(-maxChars);
}

function formatSummaryItemsForPrompt(items) {
  if (!Array.isArray(items) || !items.length) {
    return "";
  }

  return items
    .slice(-6)
    .map((item, index) => {
      const summary = String(item?.summary || "").trim();
      if (!summary) {
        return "";
      }

      return `${index + 1}. ${summary}`;
    })
    .filter(Boolean)
    .join("\n");
}

function getPdfMatchingPayload(mergedResults) {
  if (!mergedResults || typeof mergedResults !== "object") {
    return null;
  }

  if (
    mergedResults.pdf_matching &&
    typeof mergedResults.pdf_matching === "object"
  ) {
    return mergedResults.pdf_matching;
  }

  return mergedResults;
}

function getPageMatchArrays(mergedResults) {
  const payload = getPdfMatchingPayload(mergedResults);
  if (!payload) {
    return { timeline: [], segmentMatches: [] };
  }

  const timeline = Array.isArray(payload.timeline)
    ? payload.timeline
    : Array.isArray(payload.page_timeline)
      ? payload.page_timeline
      : [];
  const segmentMatches = Array.isArray(payload.segment_matches)
    ? payload.segment_matches
    : Array.isArray(payload.segmentMatches)
      ? payload.segmentMatches
      : [];

  return { timeline, segmentMatches };
}

function buildPageMatchSnapshot(mergedResults) {
  const { timeline, segmentMatches } = getPageMatchArrays(mergedResults);

  const normalizedTimeline = timeline
    .map((entry) => ({
      pageIndex: Number(entry?.page_index ?? entry?.pageIndex ?? -1),
      startTime: Number(entry?.start_time ?? entry?.startTime ?? 0),
      endTime: Number(entry?.end_time ?? entry?.endTime ?? 0),
    }))
    .filter((entry) => Number.isFinite(entry.pageIndex) && entry.pageIndex >= 0);

  const normalizedSegmentMatches = segmentMatches
    .map((entry) => ({
      pageIndex: Number(entry?.page_index ?? entry?.pageIndex ?? -1),
      confidence: Number(entry?.confidence ?? 0),
    }))
    .filter((entry) => Number.isFinite(entry.pageIndex) && entry.pageIndex >= 0);

  const currentPage =
    normalizedTimeline.length > 0
      ? normalizedTimeline[normalizedTimeline.length - 1].pageIndex + 1
      : null;

  return {
    timeline: normalizedTimeline,
    segmentMatches: normalizedSegmentMatches,
    currentPage,
  };
}

// The chat panel uses the existing DeepSeek provider settings and reads the
// active workspace artifacts directly from cache. This keeps Q&A grounded in
// the current workspace without introducing a separate backend dependency.
async function askDeepSeekCourseQuestion(payload) {
  const question = String(payload?.question || "").trim();
  const workspaceId = String(payload?.workspaceId || "").trim();
  const currentFileName = String(payload?.currentFileName || "").trim();
  const conversationHistory = Array.isArray(payload?.conversationHistory)
    ? payload.conversationHistory
    : [];

  if (!question) {
    throw new Error("问题不能为空。");
  }

  const settings = await getLiteSynphoniaProviderSettingsSnapshot();
  const apiKey = String(settings.deepseekApiKey || "").trim();
  if (!apiKey) {
    throw new Error("请先在设置中填写 DeepSeek API Key。");
  }

  const contextSections = [];
  const citations = [];
  let matchedPage = null;

  if (workspaceId) {
    const cacheRootPath = getWorkspaceCacheRoot();
    const { workspace } = await getIndexedWorkspace(cacheRootPath, workspaceId);

    if (workspace) {
      const mergedResultsPath = path.join(
        workspace.workspacePath,
        LITESYNPHONIA_MERGED_RESULTS_FILE_NAME,
      );
      const [summaryArtifact, transcriptArtifact, mergedResults] =
        await Promise.all([
          readJsonFile(workspace.artifacts.summaryPath, null),
          readJsonFile(workspace.artifacts.transcriptPath, null),
          readJsonFile(mergedResultsPath, null),
        ]);

      const summaryText = String(summaryArtifact?.summaryText || "").trim();
      const summaryItems = formatSummaryItemsForPrompt(summaryArtifact?.items);
      const transcriptText = String(
        transcriptArtifact?.transcriptText || "",
      ).trim();

      if (summaryText || summaryItems) {
        contextSections.push(
          [
            "课程总结：",
            takeTailText(summaryText, 2400),
            summaryItems ? `分段总结：\n${summaryItems}` : "",
          ]
            .filter(Boolean)
            .join("\n\n"),
        );
        citations.push("summary.full.json");
      }

      if (transcriptText) {
        contextSections.push(
          `转录原文（最近片段）：\n${takeTailText(transcriptText, 8000)}`,
        );
        citations.push("transcription.full.json");
      }

      // 如果有页码匹配结果，把 PDF 页面内容也带入上下文
      const { segmentMatches } = getPageMatchArrays(mergedResults);
      if (segmentMatches.length) {
        const pageNums = segmentMatches
          .map((m) => m?.page_index ?? m?.pageIndex)
          .filter((p) => typeof p === "number")
          .map((p) => p + 1); // 转为 1-indexed
        if (pageNums.length) {
          const uniquePages = [...new Set(pageNums)].sort((a, b) => a - b);
          contextSections.push(
            `课件页码覆盖范围（根据转录内容匹配）：第 ${uniquePages.join("、")} 页`,
          );
          // 把最后一个匹配页作为推荐参考页
          matchedPage = uniquePages[uniquePages.length - 1];
          citations.push(`page-match (第 ${matchedPage} 页附近)`);
        }
      }
    }
  }

  // 构建多轮对话历史（最多保留最近 6 轮，避免 token 超限）
  const historyMessages = conversationHistory
    .slice(-12) // 最多 6 轮 = 12 条消息
    .map((msg) => ({
      role: msg.role === "assistant" ? "assistant" : "user",
      content: String(msg.content || "").trim(),
    }))
    .filter((msg) => msg.content);

  const systemMessage = {
    role: "system",
    content:
      contextSections.length
        ? "你是一个通用 AI 助手。当前如果提供了课件工作区上下文，应优先利用这些上下文回答；如果用户问题与课件无关，也可以直接进行普通对话。使用简体中文，回答简洁、直接。如果引用页码，请使用“第 X 页”的格式。"
        : "你是一个通用 AI 助手。使用简体中文，回答简洁、直接，支持连续多轮对话。",
  };

  const contextMessages = contextSections.length
    ? [
        {
          role: "user",
          content: [
            `当前工作区：${currentFileName || "未命名课件"}`,
            ...contextSections,
          ].join("\n\n"),
        },
        {
          role: "assistant",
          content: "好的，我已经了解当前工作区内容，可以继续提问。",
        },
      ]
    : [];

  const messages = [
    systemMessage,
    ...contextMessages,
    ...historyMessages,
    { role: "user", content: question },
  ];

  const response = await fetch(
    `${APP_MANAGED_PROVIDER_PRESETS.deepseek.baseUrl}/chat/completions`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: APP_MANAGED_PROVIDER_PRESETS.deepseek.modelId,
        temperature: 0.2,
        messages,
      }),
    },
  );

  if (!response.ok) {
    const errorText = takeTailText(await response.text(), 400);
    throw new Error(
      `DeepSeek 问答请求失败: ${response.status}${errorText ? ` - ${errorText}` : ""}`,
    );
  }

  const payloadJson = await response.json();
  const answer = String(
    payloadJson?.choices?.[0]?.message?.content || "",
  ).trim();

  if (!answer) {
    throw new Error("DeepSeek 没有返回可用答案。");
  }

  return {
    answer,
    citations,
    matchedPage,
  };
}

// ── PPT/PPTX → PDF 转换（LibreOffice headless）──────────────────
async function findLibreOfficeExecutable() {
  const candidates =
    process.platform === "darwin"
      ? [
          "/Applications/LibreOffice.app/Contents/MacOS/soffice",
          "/Applications/LibreOffice.app/Contents/MacOS/python",
        ]
      : process.platform === "win32"
        ? [
            "C:\\Program Files\\LibreOffice\\program\\soffice.exe",
            "C:\\Program Files (x86)\\LibreOffice\\program\\soffice.exe",
          ]
        : ["/usr/bin/libreoffice", "/usr/bin/soffice", "libreoffice", "soffice"];

  // macOS 特殊路径
  const macSoffice = "/Applications/LibreOffice.app/Contents/MacOS/soffice";
  if (process.platform === "darwin" && (await pathExists(macSoffice))) {
    return macSoffice;
  }

  for (const candidate of candidates) {
    if (
      candidate.startsWith("/") ||
      candidate.includes("\\")
    ) {
      if (await pathExists(candidate)) {
        return candidate;
      }
    }
  }

  // 尝试 PATH 中的 libreoffice
  return "libreoffice";
}

async function convertPptxToPdf(sourcePath, outputDir) {
  const soffice = await findLibreOfficeExecutable();

  return new Promise((resolve) => {
    const child = spawn(
      soffice,
      ["--headless", "--convert-to", "pdf", "--outdir", outputDir, sourcePath],
      { cwd: outputDir, stdio: ["ignore", "pipe", "pipe"] },
    );

    const stderrChunks = [];
    child.stderr?.on("data", (chunk) => {
      stderrChunks.push(chunk.toString("utf8"));
    });

    child.on("error", (err) => {
      resolve({
        ok: false,
        message: `LibreOffice 未找到或无法启动：${err.message}。请安装 LibreOffice 后再试。`,
      });
    });

    child.on("close", async (code) => {
      if (code !== 0) {
        resolve({
          ok: false,
          message:
            stderrChunks.join("").trim() ||
            `LibreOffice 转换失败，退出码 ${code}。`,
        });
        return;
      }

      // LibreOffice 输出文件名：把原文件名的扩展改为 .pdf
      const baseName = path.basename(sourcePath, path.extname(sourcePath));
      const pdfPath = path.join(outputDir, `${baseName}.pdf`);

      if (!(await pathExists(pdfPath))) {
        resolve({ ok: false, message: "LibreOffice 转换完成，但找不到输出 PDF 文件。" });
        return;
      }

      try {
        const pdfBytes = await fs.readFile(pdfPath);
        resolve({ ok: true, pdfPath, pdfBytes: new Uint8Array(pdfBytes) });
      } catch (err) {
        resolve({ ok: false, message: `读取转换结果失败：${err.message}` });
      }
    });
  });
}

// ── 页码匹配结果读取 ─────────────────────────────────────────────
async function readWorkspacePageMatchTimeline(workspace) {
  const mergedResultsPath = path.join(
    workspace.workspacePath,
    LITESYNPHONIA_MERGED_RESULTS_FILE_NAME,
  );
  const payload = await readJsonFile(mergedResultsPath, null);
  if (!payload) return null;
  return buildPageMatchSnapshot(payload);
}

function getKnowledgeBaseV2WorkspaceRoot() {
  return path.join(app.getPath("userData"), KNOWLEDGE_BASE_V2_WORKSPACE_DIRNAME);
}

function truncateLeadingText(text, maxChars) {
  const normalized = String(text || "").trim();
  if (!normalized || normalized.length <= maxChars) {
    return normalized;
  }

  return `${normalized.slice(0, maxChars).trim()}...`;
}

function normalizeKnowledgeBaseText(text) {
  return String(text || "").replace(/\s+/g, " ").trim();
}

function extractKnowledgeBaseTerms(text, limit = KNOWLEDGE_BASE_KEYWORD_LIMIT) {
  const normalizedText = normalizeKnowledgeBaseText(text);
  if (!normalizedText) {
    return [];
  }

  const asciiStopwords = new Set([
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "were",
    "have",
    "has",
    "into",
    "week",
    "class",
    "meeting",
    "teacher",
    "students",
    "session",
    "slides",
    "slide",
    "notes",
  ]);
  const zhStopwords = new Set([
    "我们",
    "你们",
    "他们",
    "老师",
    "学生",
    "内容",
    "活动",
    "问题",
    "进行",
    "可以",
    "以及",
    "这个",
    "那个",
    "部分",
  ]);

  const frequencies = new Map();
  const remember = (rawTerm) => {
    const value = String(rawTerm || "").trim();
    if (!value) {
      return;
    }
    const key = value.toLowerCase();
    frequencies.set(key, {
      value,
      count: (frequencies.get(key)?.count || 0) + 1,
    });
  };

  for (const token of normalizedText.match(/[A-Za-z][A-Za-z0-9_.+#-]{1,31}/g) || []) {
    const normalized = token.toLowerCase();
    if (normalized.length < 3 || asciiStopwords.has(normalized)) {
      continue;
    }
    remember(token);
  }

  const zhParts = normalizedText
    .split(/[，。；：、“”‘’（）()【】《》、,\s/]+/u)
    .map((part) => part.trim())
    .filter(Boolean);
  for (const part of zhParts) {
    if (
      part.length < 2 ||
      part.length > 10 ||
      !/^[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+$/u.test(part) ||
      zhStopwords.has(part)
    ) {
      continue;
    }
    remember(part);
  }

  return [...frequencies.values()]
    .sort((left, right) => right.count - left.count || right.value.length - left.value.length)
    .map((item) => item.value)
    .slice(0, limit);
}

function buildKnowledgeBaseSummaryText(summaryArtifact, sidebarState) {
  const directSummary = normalizeKnowledgeBaseText(summaryArtifact?.summaryText);
  const sidebarSummaries = joinTextBlocks(
    ...(Array.isArray(sidebarState?.items)
      ? sidebarState.items.map((item) => String(item?.summary || ""))
      : []),
  );
  return directSummary || sidebarSummaries || "";
}

function buildKnowledgeBaseTranscriptText(transcriptArtifact, sidebarState) {
  const directTranscript = normalizeKnowledgeBaseText(
    sidebarState?.finalTranscriptText || transcriptArtifact?.transcriptText,
  );
  if (directTranscript) {
    return directTranscript;
  }

  const segments = Array.isArray(transcriptArtifact?.segments)
    ? transcriptArtifact.segments
        .map((segment) => String(segment?.text || "").trim())
        .filter(Boolean)
    : [];
  return normalizeKnowledgeBaseText(segments.join(" "));
}

async function buildKnowledgeBaseActivityRecord(workspace) {
  const [sidebarState, transcriptArtifact, summaryArtifact, pageMatch, pdfKeywordHints] =
    await Promise.all([
      readJsonFile(workspace.artifacts.sidebarStatePath, null),
      readJsonFile(workspace.artifacts.transcriptPath, null),
      readJsonFile(workspace.artifacts.summaryPath, null),
      readWorkspacePageMatchTimeline(workspace).catch(() => null),
      extractPdfKeywordHints(workspace).catch(() => []),
    ]);

  const transcriptText = buildKnowledgeBaseTranscriptText(
    transcriptArtifact,
    sidebarState,
  );
  const summaryText = buildKnowledgeBaseSummaryText(summaryArtifact, sidebarState);

  if (!transcriptText && !summaryText) {
    return null;
  }

  const sourceName = getDefaultWorkspaceDisplayName(workspace.sourceFileName, workspace.workspaceName);
  const summaryOfSummary =
    truncateLeadingText(
      summaryText ||
        (Array.isArray(sidebarState?.items) && sidebarState.items.length
          ? String(sidebarState.items[sidebarState.items.length - 1]?.summary || "")
          : "") ||
        transcriptText,
      120,
    ) || sourceName;
  const keywords = mergeKeywordLists(
    Array.isArray(summaryArtifact?.keywords) ? summaryArtifact.keywords : [],
    extractKnowledgeBaseTerms(summaryText, KNOWLEDGE_BASE_KEYWORD_LIMIT),
    pdfKeywordHints,
    extractKnowledgeBaseTerms(transcriptText, 4),
    [sourceName],
  ).slice(0, KNOWLEDGE_BASE_KEYWORD_LIMIT);
  const keywordsOfKeywords = mergeKeywordLists(
    extractKnowledgeBaseTerms(summaryOfSummary, KNOWLEDGE_BASE_TOPIC_LIMIT),
    keywords.slice(0, KNOWLEDGE_BASE_TOPIC_LIMIT),
    [workspace.sourceKind === "other" ? "活动记录" : "课程内容"],
  ).slice(0, KNOWLEDGE_BASE_TOPIC_LIMIT);
  const sourceCopyPath = String(workspace.artifacts?.sourceCopyPath || "").trim();
  const hasSlides =
    Boolean(sourceCopyPath) &&
    ["pdf", "ppt", "pptx"].includes(String(workspace.sourceKind || ""));

  return {
    activity_id: workspace.workspaceId,
    start_time: workspace.createdAtUtc,
    end_time: workspace.updatedAtUtc || workspace.lastOpenedAtUtc || workspace.createdAtUtc,
    transcript_text: transcriptText,
    summary_text: summaryText || summaryOfSummary,
    summary_of_summary: summaryOfSummary,
    keywords,
    keywords_of_keywords: keywordsOfKeywords,
    ppt_present: hasSlides,
    ppt_file_path: hasSlides ? sourceCopyPath : undefined,
    activity_intro: summaryOfSummary,
    activity_name: workspace.workspaceName || sourceName,
    activity_dir: workspace.workspacePath,
    transcript_file_path: workspace.artifacts.transcriptPath,
    summary_file_path: workspace.artifacts.summaryPath,
    matched_slides:
      pageMatch?.timeline?.map((entry) => ({
        page_index: entry.pageIndex + 1,
        start_time: entry.startTime,
        end_time: entry.endTime,
      })) || [],
    ppt_text_excerpt: pdfKeywordHints.join("、") || undefined,
    scene_type: hasSlides ? "classroom" : "activity",
    transcript_meta: {
      workspace_status: workspace.status,
      segment_count: Array.isArray(sidebarState?.finalSegments)
        ? sidebarState.finalSegments.length
        : Array.isArray(transcriptArtifact?.segments)
          ? transcriptArtifact.segments.length
          : 0,
    },
    summary_meta: {
      summary_count: Array.isArray(sidebarState?.items) ? sidebarState.items.length : 0,
      workspace_status: workspace.status,
    },
  };
}

function buildEmptyKnowledgeBaseExport() {
  return {
    core_data: {
      activities: [],
      selected_activity: null,
      content_lines: [],
      counts: {
        activity_count: 0,
        content_line_count: 0,
        attachment_count: 0,
      },
    },
    graph_view: {
      nodes: [],
      edges: [],
    },
    legacy_view_bundle: {
      navigation: [],
      history: {
        navigation: [],
        statistics_cards: [],
        full_record_list: [],
        content_lines: [],
        attachment_records: [],
        pending_relations: [],
      },
      relation_map: {
        nodes: [],
        edges: [],
      },
      timeline_calendar: {
        mode: "chronological_calendar",
        dates: [],
      },
      timeline_line_view: {
        mode: "content_line_timeline",
        content_lines: [],
      },
      file_lookup: {
        activity_groups: [],
      },
      detail_panel: null,
    },
  };
}

async function exportKnowledgeBaseV2Data(selectedActivityId = "") {
  const cacheRootPath = getWorkspaceCacheRoot();
  const workspaces = await syncWorkspaceIndex(cacheRootPath);
  const activities = (
    await Promise.all(
      workspaces.map((workspace) => buildKnowledgeBaseActivityRecord(workspace)),
    )
  ).filter(Boolean);

  if (!activities.length) {
    return {
      ingestResults: [],
      data: buildEmptyKnowledgeBaseExport(),
    };
  }

  const kbWorkspacePath = getKnowledgeBaseV2WorkspaceRoot();
  await ensureDirectory(kbWorkspacePath);
  const activitiesPath = path.join(
    kbWorkspacePath,
    KNOWLEDGE_BASE_V2_ACTIVITY_INPUT_FILE_NAME,
  );
  await writeJsonFile(activitiesPath, activities);

  const pythonExecutable = await getLiteSynphoniaPythonExecutable();
  const exportScript = `
import json
import sys
from pathlib import Path

from lite_synphonia.knowledge_base.service import KnowledgeBaseService

workspace_path = Path(sys.argv[1]).expanduser().resolve()
activities_path = Path(sys.argv[2]).expanduser().resolve()
selected_activity_id = (sys.argv[3] or "").strip() or None

service = KnowledgeBaseService(workspace_path)
service.reset()
payload = json.loads(activities_path.read_text(encoding="utf-8"))
ingest_results = service.ingest_many(list(payload), base_dir=activities_path.parent)
exported = service.export_all_views(selected_activity_id=selected_activity_id)

print(json.dumps({
    "ingest_results": ingest_results,
    "data": exported,
}, ensure_ascii=False))
  `.trim();

  return await new Promise((resolve, reject) => {
    const child = spawn(
      pythonExecutable,
      ["-c", exportScript, kbWorkspacePath, activitiesPath, String(selectedActivityId || "")],
      {
        cwd: projectRoot,
        stdio: ["ignore", "pipe", "pipe"],
      },
    );

    const stdoutChunks = [];
    const stderrChunks = [];
    child.stdout?.on("data", (chunk) => {
      stdoutChunks.push(chunk.toString("utf8"));
    });
    child.stderr?.on("data", (chunk) => {
      stderrChunks.push(chunk.toString("utf8"));
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        reject(
          new Error(
            stderrChunks.join("").trim() ||
              `知识库 V2 导出失败，退出码 ${typeof code === "number" ? code : "unknown"}`,
          ),
        );
        return;
      }

      try {
        const payload = JSON.parse(stdoutChunks.join("").trim() || "{}");
        resolve({
          ingestResults: Array.isArray(payload?.ingest_results)
            ? payload.ingest_results
            : [],
          data: payload?.data || buildEmptyKnowledgeBaseExport(),
        });
      } catch (error) {
        reject(
          error instanceof Error
            ? error
            : new Error("解析知识库 V2 导出结果失败。"),
        );
      }
    });
  });
}

// ── 知识图谱 IPC ─────────────────────────────────────────────────
async function buildActivityKnowledgeGraph(cacheRootPath) {
  const workspaces = await syncWorkspaceIndex(cacheRootPath);
  const nodes = [];
  const edges = [];

  // 为每个工作区收集关键词
  const keywordMap = new Map(); // workspaceId -> Set<keyword>

  for (const workspace of workspaces) {
    const summaryArtifact = await readJsonFile(
      workspace.artifacts.summaryPath,
      null,
    );
    const keywords = Array.isArray(summaryArtifact?.keywords)
      ? summaryArtifact.keywords
          .map((k) => String(k || "").trim())
          .filter(Boolean)
      : [];

    if (!keywords.length && !String(summaryArtifact?.summaryText || "").trim()) {
      // 跳过没有任何内容的工作区
      continue;
    }

    nodes.push({
      id: workspace.workspaceId,
      label: workspace.workspaceName,
      keywords,
      createdAt: workspace.createdAtUtc,
    });

    keywordMap.set(workspace.workspaceId, new Set(keywords));
  }

  // 计算节点间共享关键词形成边
  for (let a = 0; a < nodes.length; a++) {
    for (let b = a + 1; b < nodes.length; b++) {
      const setA = keywordMap.get(nodes[a].id) || new Set();
      const setB = keywordMap.get(nodes[b].id) || new Set();
      const shared = [...setA].filter((kw) => setB.has(kw));
      if (shared.length >= 1) {
        edges.push({
          source: nodes[a].id,
          target: nodes[b].id,
          sharedKeywords: shared.slice(0, 8),
        });
      }
    }
  }

  return { nodes, edges };
}

async function getLiteSynphoniaPythonExecutable() {
  const candidates =
    process.platform === "win32"
      ? [path.join(LITESYNPHONIA_VENV_DIR, "Scripts", "python.exe")]
      : [
          path.join(LITESYNPHONIA_VENV_DIR, "bin", "python3"),
          path.join(LITESYNPHONIA_VENV_DIR, "bin", "python"),
        ];

  for (const candidate of candidates) {
    if (await pathExists(candidate)) {
      return candidate;
    }
  }

  return "python3";
}

async function checkLiteSynphoniaPythonDependencies(workspace) {
  const requiredModules = ["numpy", "sounddevice"];
  if (workspace.sourceKind === "pdf") {
    requiredModules.push("pypdf");
  }

  const checkScript = [
    "import importlib.util, json",
    `modules = ${JSON.stringify(requiredModules)}`,
    "missing = [name for name in modules if importlib.util.find_spec(name) is None]",
    "print(json.dumps({'missing': missing}, ensure_ascii=False))",
  ].join("; ");
  const pythonExecutable = await getLiteSynphoniaPythonExecutable();

  return await new Promise((resolve) => {
    const child = spawn(pythonExecutable, ["-c", checkScript], {
      cwd: projectRoot,
      stdio: ["ignore", "pipe", "pipe"],
    });

    const stdoutChunks = [];
    const stderrChunks = [];

    child.stdout?.on("data", (chunk) => {
      stdoutChunks.push(chunk.toString("utf8"));
    });
    child.stderr?.on("data", (chunk) => {
      stderrChunks.push(chunk.toString("utf8"));
    });

    child.on("error", (error) => {
      resolve({
        ok: false,
        message: `无法启动 python3 检查 LiteSynphonia 依赖: ${error.message}`,
      });
    });

    child.on("close", (code) => {
      if (code !== 0) {
        resolve({
          ok: false,
          message:
            stderrChunks.join("").trim() ||
            `Python 依赖检查失败，退出码 ${typeof code === "number" ? code : "unknown"}。`,
        });
        return;
      }

      try {
        const payload = JSON.parse(stdoutChunks.join("").trim() || "{}");
        const missingModules = Array.isArray(payload?.missing)
          ? payload.missing
              .map((item) => String(item || "").trim())
              .filter(Boolean)
          : [];

        if (!missingModules.length) {
          resolve({ ok: true });
          return;
        }

        resolve({
          ok: false,
          message: [
            "LiteSynphonia Python 依赖未安装完整。",
            `缺少模块: ${missingModules.join(", ")}`,
            `先执行: python3 -m venv ${LITESYNPHONIA_VENV_DIR}`,
            `然后执行: ${path.join(LITESYNPHONIA_VENV_DIR, process.platform === "win32" ? "Scripts" : "bin", process.platform === "win32" ? "python.exe" : "python3")} -m pip install -r ${LITESYNPHONIA_REQUIREMENTS_PATH}`,
          ].join("\n"),
        });
      } catch (error) {
        resolve({
          ok: false,
          message:
            error instanceof Error
              ? `解析 Python 依赖检查结果失败: ${error.message}`
              : "解析 Python 依赖检查结果失败。",
        });
      }
    });
  });
}

async function checkWorkspacePipelinePrerequisites(workspace) {
  const { providerPath, entries } = await readProviderEntries();
  const requiredServices =
    workspace.sourceKind === "pdf"
      ? ["transcription", "summarization", "embedding"]
      : ["transcription", "summarization"];

  const availableByService = new Map();
  for (const entry of entries) {
    const apiKey = String(entry?.api_key || "").trim();
    const services = Array.isArray(entry?.services) ? entry.services : [];

    if (!apiKey) {
      continue;
    }

    for (const service of services) {
      const nextService = String(service || "").trim();
      if (!nextService) {
        continue;
      }

      const serviceEntries = availableByService.get(nextService) || [];
      serviceEntries.push(String(entry?.name || "").trim() || "unnamed");
      availableByService.set(nextService, serviceEntries);
    }
  }

  const missingServices = requiredServices.filter(
    (service) => !(availableByService.get(service) || []).length,
  );

  if (!missingServices.length) {
    return { ok: true, providerPath };
  }

  const requiredKeys = [];
  if (missingServices.includes("transcription")) {
    requiredKeys.push("Deepgram API Key");
  }
  if (
    missingServices.includes("summarization") ||
    missingServices.includes("embedding")
  ) {
    if (missingServices.includes("summarization")) {
      requiredKeys.push("DeepSeek API Key");
    }
    if (missingServices.includes("embedding")) {
      requiredKeys.push("SiliconFlow API Key");
    }
  }

  return {
    ok: false,
    message: [
      "LiteSynphonia 配置未完成。",
      `缺少服务: ${missingServices.join(", ")}`,
      `请打开右上角设置，填写并保存: ${[...new Set(requiredKeys)].join("、")}`,
      "Deepgram 负责转录，DeepSeek 负责总结，SiliconFlow BGE 负责 PDF 页码匹配。",
      `配置文件路径: ${providerPath}`,
    ].join("\n"),
  };
}

function isWorkspaceStatus(value) {
  return WORKSPACE_STATUSES.has(String(value));
}

function isWorkspaceRunState(value) {
  return WORKSPACE_RUN_STATES.has(String(value));
}

function normalizeIsoString(value, fallbackValue) {
  const normalized = String(value || "").trim();
  if (!normalized) {
    return fallbackValue;
  }

  const parsed = new Date(normalized);
  return Number.isNaN(parsed.valueOf()) ? fallbackValue : parsed.toISOString();
}

function getMimeTypeForKind(sourceKind) {
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

function createWorkspacePipelineStatus(payload) {
  return {
    workspaceId: String(payload?.workspaceId || ""),
    runState: isWorkspaceRunState(payload?.runState)
      ? String(payload.runState)
      : "idle",
    processingStatus: isWorkspaceStatus(payload?.processingStatus)
      ? String(payload.processingStatus)
      : "initialized",
    startedAtUtc: payload?.startedAtUtc
      ? normalizeIsoString(payload.startedAtUtc, new Date().toISOString())
      : undefined,
    finishedAtUtc: payload?.finishedAtUtc
      ? normalizeIsoString(payload.finishedAtUtc, new Date().toISOString())
      : undefined,
    message: payload?.message ? String(payload.message) : "",
    exitCode:
      typeof payload?.exitCode === "number" && Number.isFinite(payload.exitCode)
        ? payload.exitCode
        : undefined,
    completedCycles:
      typeof payload?.completedCycles === "number" &&
      Number.isFinite(payload.completedCycles)
        ? payload.completedCycles
        : 0,
  };
}

function getWorkspacePipelineStatus(
  workspaceId,
  processingStatus = "initialized",
) {
  const cachedStatus = workspacePipelineRuns.get(workspaceId);
  if (cachedStatus) {
    return cachedStatus;
  }

  return createWorkspacePipelineStatus({
    workspaceId,
    runState: "idle",
    processingStatus,
  });
}

function setWorkspacePipelineStatus(workspaceId, nextState) {
  const current = workspacePipelineRuns.get(workspaceId);
  const status = createWorkspacePipelineStatus({
    ...current,
    ...nextState,
    workspaceId,
  });
  workspacePipelineRuns.set(workspaceId, status);
  return status;
}

function sortWorkspaces(workspaces) {
  return [...workspaces].sort((left, right) => {
    const starredDelta =
      Number(Boolean(right?.starred)) - Number(Boolean(left?.starred));
    if (starredDelta) {
      return starredDelta;
    }

    const rightStamp = Date.parse(
      right.lastOpenedAtUtc || right.updatedAtUtc || 0,
    );
    const leftStamp = Date.parse(
      left.lastOpenedAtUtc || left.updatedAtUtc || 0,
    );
    return rightStamp - leftStamp;
  });
}

function createWorkspaceRecord(payload) {
  const now = new Date().toISOString();
  const createdAtUtc = normalizeIsoString(payload?.createdAtUtc, now);
  const updatedAtUtc = normalizeIsoString(payload?.updatedAtUtc, createdAtUtc);
  const rawSourceKind = String(payload?.sourceKind || "").trim();
  const sourceKind =
    rawSourceKind && ["pdf", "ppt", "pptx", "other"].includes(rawSourceKind)
      ? rawSourceKind
      : detectSourceKind(payload?.sourceFileName || "", "");

  return {
    workspaceId: String(payload?.workspaceId || ""),
    workspaceName: getDefaultWorkspaceDisplayName(
      payload?.workspaceName || payload?.sourceFileName || "",
      path.basename(String(payload?.workspacePath || "")) || "未命名工作区",
    ),
    workspacePath: String(payload?.workspacePath || ""),
    cacheRootPath: String(payload?.cacheRootPath || ""),
    sourceFileName: String(payload?.sourceFileName || ""),
    sourceKind,
    starred: Boolean(payload?.starred),
    status: isWorkspaceStatus(payload?.status)
      ? String(payload.status)
      : "initialized",
    createdAtUtc,
    updatedAtUtc,
    lastOpenedAtUtc: normalizeIsoString(payload?.lastOpenedAtUtc, updatedAtUtc),
    artifacts: {
      transcriptPath: String(payload?.artifacts?.transcriptPath || ""),
      summaryPath: String(payload?.artifacts?.summaryPath || ""),
      sourceCopyPath: String(payload?.artifacts?.sourceCopyPath || ""),
      sidebarStatePath: String(payload?.artifacts?.sidebarStatePath || ""),
    },
  };
}

async function readWorkspaceIndex(cacheRootPath) {
  const indexPath = getWorkspaceIndexPath(cacheRootPath);
  let payload;

  try {
    payload = await readJsonFile(indexPath, {
      schemaVersion: "1.0",
      workspaces: [],
    });
  } catch (error) {
    if (error instanceof SyntaxError) {
      const backupPath = await backupCorruptedJsonFile(indexPath).catch(
        () => "",
      );
      console.warn(
        "[workspace-index] Detected malformed JSON. Rebuilding index from cache directories.",
        {
          indexPath,
          backupPath,
          error: error.message,
        },
      );
      return [];
    }

    throw error;
  }

  const workspaceList = Array.isArray(payload)
    ? payload
    : Array.isArray(payload?.workspaces)
      ? payload.workspaces
      : [];

  return workspaceList
    .map((item) => createWorkspaceRecord(item))
    .filter((item) => item.workspaceId && item.workspacePath);
}

async function writeWorkspaceIndex(cacheRootPath, workspaces) {
  await ensureDirectory(cacheRootPath);
  await writeJsonFile(getWorkspaceIndexPath(cacheRootPath), {
    schemaVersion: "1.0",
    workspaces: sortWorkspaces(workspaces),
  });
}

async function discoverWorkspaces(cacheRootPath) {
  await ensureDirectory(cacheRootPath);

  const directoryEntries = await fs.readdir(cacheRootPath, {
    withFileTypes: true,
  });
  const discovered = [];

  for (const entry of directoryEntries) {
    if (!entry.isDirectory()) {
      continue;
    }

    const workspacePath = path.join(cacheRootPath, entry.name);
    const transcriptPath = path.join(workspacePath, TRANSCRIPT_FILE_NAME);
    const summaryPath = path.join(workspacePath, SUMMARY_FILE_NAME);
    const sidebarStatePath = path.join(workspacePath, SIDEBAR_STATE_FILE_NAME);

    const [
      transcriptPayload,
      summaryPayload,
      sidebarPayload,
      stats,
      childEntries,
    ] = await Promise.all([
      readJsonFile(transcriptPath, null),
      readJsonFile(summaryPath, null),
      readJsonFile(sidebarStatePath, null),
      fs.stat(workspacePath),
      fs.readdir(workspacePath, { withFileTypes: true }),
    ]);

    const declaredSourceFileName =
      String(transcriptPayload?.sourceFileName || "").trim() ||
      String(summaryPayload?.sourceFileName || "").trim();
    const declaredSourceKind =
      detectSourceKind(
        declaredSourceFileName,
        transcriptPayload?.sourceKind || summaryPayload?.sourceKind || "",
      ) || "other";
    const sourceEntry = selectWorkspaceSourceEntry(childEntries, {
      sourceFileName: declaredSourceFileName,
      sourceKind: declaredSourceKind,
    });

    const sourceFileName = declaredSourceFileName || sourceEntry?.name || entry.name;
    const sourceKind =
      detectSourceKind(
        sourceFileName,
        transcriptPayload?.sourceKind || summaryPayload?.sourceKind || "",
      ) || "other";
    const updatedAtUtc = normalizeIsoString(
      transcriptPayload?.updatedAtUtc ||
        summaryPayload?.updatedAtUtc ||
        sidebarPayload?.updatedAtUtc,
      stats.mtime.toISOString(),
    );
    const createdAtUtc = normalizeIsoString(
      transcriptPayload?.createdAtUtc ||
        summaryPayload?.createdAtUtc ||
        sidebarPayload?.createdAtUtc,
      stats.birthtime.toISOString(),
    );

    discovered.push(
      createWorkspaceRecord({
        workspaceId:
          transcriptPayload?.workspaceId ||
          summaryPayload?.workspaceId ||
          entry.name,
        workspaceName: getDefaultWorkspaceDisplayName(
          sourceFileName,
          entry.name,
        ),
        workspacePath,
        cacheRootPath,
        sourceFileName,
        sourceKind,
        status:
          transcriptPayload?.status ||
          summaryPayload?.status ||
          sidebarPayload?.status ||
          "initialized",
        createdAtUtc,
        updatedAtUtc,
        lastOpenedAtUtc: updatedAtUtc,
        artifacts: {
          transcriptPath,
          summaryPath,
          sourceCopyPath: sourceEntry
            ? path.join(workspacePath, sourceEntry.name)
            : "",
          sidebarStatePath,
        },
      }),
    );
  }

  return discovered;
}

async function syncWorkspaceIndex(cacheRootPath) {
  const [indexedWorkspaces, discoveredWorkspaces] = await Promise.all([
    readWorkspaceIndex(cacheRootPath),
    discoverWorkspaces(cacheRootPath),
  ]);

  const mergedById = new Map();

  for (const workspace of indexedWorkspaces) {
    mergedById.set(workspace.workspaceId, workspace);
  }

  for (const workspace of discoveredWorkspaces) {
    const current = mergedById.get(workspace.workspaceId);
    mergedById.set(
      workspace.workspaceId,
      current
        ? createWorkspaceRecord({
            ...current,
            ...workspace,
            workspaceName: current.workspaceName || workspace.workspaceName,
            starred: Boolean(current?.starred || workspace?.starred),
            lastOpenedAtUtc:
              current.lastOpenedAtUtc || workspace.lastOpenedAtUtc,
            status: current.status || workspace.status,
          })
        : workspace,
    );
  }

  const syncedWorkspaces = sortWorkspaces([...mergedById.values()]);
  await writeWorkspaceIndex(cacheRootPath, syncedWorkspaces);
  return syncedWorkspaces;
}

async function getIndexedWorkspace(cacheRootPath, workspaceId) {
  const indexedWorkspaces = await syncWorkspaceIndex(cacheRootPath);
  const workspace = indexedWorkspaces.find(
    (item) => item.workspaceId === workspaceId,
  );
  return { indexedWorkspaces, workspace };
}

async function persistWorkspaceRecord(cacheRootPath, workspace) {
  const indexedWorkspaces = await syncWorkspaceIndex(cacheRootPath);
  const nextIndex = sortWorkspaces([
    workspace,
    ...indexedWorkspaces.filter(
      (item) => item.workspaceId !== workspace.workspaceId,
    ),
  ]);
  await writeWorkspaceIndex(cacheRootPath, nextIndex);
  return workspace;
}

function buildTranscriptSeed(workspaceId, sourceFileName, sourceKind) {
  const now = new Date().toISOString();

  return {
    schemaVersion: "1.0",
    workspaceId,
    status: "initialized",
    sourceFileName,
    sourceKind,
    transcriptText: "",
    segments: [],
    createdAtUtc: now,
    updatedAtUtc: now,
  };
}

function buildSummarySeed(workspaceId, sourceFileName, sourceKind) {
  const now = new Date().toISOString();

  return {
    schemaVersion: "1.0",
    workspaceId,
    status: "initialized",
    sourceFileName,
    sourceKind,
    summaryText: "",
    items: [],
    keywords: [],
    windowing: {
      triggerChars: SUMMARY_WINDOW_TRIGGER_CHARS,
      overlapChars: SUMMARY_WINDOW_OVERLAP_CHARS,
      pendingTranscript: "",
      pendingChars: 0,
      generatedCount: 0,
    },
    createdAtUtc: now,
    updatedAtUtc: now,
  };
}

function normalizeSidebarState(sidebarState) {
  const now = new Date().toISOString();
  const items = Array.isArray(sidebarState?.items)
    ? sidebarState.items.map((item) => ({
        id: String(item?.id || ""),
        summary: String(item?.summary || ""),
        transcript: String(item?.transcript || ""),
        transcriptSegmentIds: Array.isArray(item?.transcriptSegmentIds)
          ? item.transcriptSegmentIds.map((segmentId) => String(segmentId))
          : [],
        transcriptRange: {
          startTime:
            typeof item?.transcriptRange?.startTime === "number"
              ? item.transcriptRange.startTime
              : null,
          endTime:
            typeof item?.transcriptRange?.endTime === "number"
              ? item.transcriptRange.endTime
              : null,
        },
      }))
    : [];
  const pendingTranscript = String(
    sidebarState?.summaryWindow?.pendingTranscript || "",
  ).trim();
  const generatedCount =
    Number(sidebarState?.summaryWindow?.generatedCount) || items.length;

  return {
    schemaVersion: "1.0",
    mode: "normal",
    status: isWorkspaceStatus(sidebarState?.status)
      ? String(sidebarState.status)
      : "initialized",
    createdAtUtc: String(sidebarState?.createdAtUtc || now),
    updatedAtUtc: now,
    items,
    summaryWindow: {
      triggerChars:
        Number(sidebarState?.summaryWindow?.triggerChars) ||
        SUMMARY_WINDOW_TRIGGER_CHARS,
      overlapChars:
        Number(sidebarState?.summaryWindow?.overlapChars) ||
        SUMMARY_WINDOW_OVERLAP_CHARS,
      pendingTranscript,
      pendingChars: countVisibleChars(pendingTranscript),
      generatedCount,
    },
  };
}

function buildTranscriptText(transcriptionPayload) {
  const results = Array.isArray(transcriptionPayload?.results)
    ? transcriptionPayload.results
    : [];

  return results
    .map((item) => String(item?.text || "").trim())
    .filter(Boolean)
    .join(" ")
    .trim();
}

function countVisibleChars(text) {
  let count = 0;

  for (const char of Array.from(String(text || ""))) {
    if (!/\s/u.test(char)) {
      count += 1;
    }
  }

  return count;
}

function takeFirstVisibleChars(text, limit) {
  if (limit <= 0) {
    return "";
  }

  const chars = [];
  let visibleCount = 0;

  for (const char of Array.from(String(text || ""))) {
    chars.push(char);
    if (!/\s/u.test(char)) {
      visibleCount += 1;
      if (visibleCount >= limit) {
        break;
      }
    }
  }

  return chars.join("").trim();
}

function sliceAfterVisibleChars(text, offset) {
  if (offset <= 0) {
    return String(text || "").trim();
  }

  const remainingChars = [];
  let visibleCount = 0;
  let started = false;

  for (const char of Array.from(String(text || ""))) {
    if (started) {
      remainingChars.push(char);
      continue;
    }

    if (!/\s/u.test(char)) {
      visibleCount += 1;
    }

    if (visibleCount >= offset) {
      started = true;
    }
  }

  return remainingChars.join("").trim();
}

function takeLastVisibleChars(text, limit) {
  if (limit <= 0) {
    return "";
  }

  const sourceChars = Array.from(String(text || ""));
  const collected = [];
  let visibleCount = 0;

  for (let index = sourceChars.length - 1; index >= 0; index -= 1) {
    const char = sourceChars[index];
    collected.push(char);
    if (!/\s/u.test(char)) {
      visibleCount += 1;
      if (visibleCount >= limit) {
        break;
      }
    }
  }

  return collected.reverse().join("").trim();
}

function joinTextBlocks(...parts) {
  return parts
    .map((part) => String(part || "").trim())
    .filter(Boolean)
    .join("\n\n")
    .trim();
}

function mergeKeywordLists(...lists) {
  const seen = new Set();
  const merged = [];

  for (const list of lists) {
    if (!Array.isArray(list)) {
      continue;
    }

    for (const item of list) {
      const value = String(item || "").trim();
      if (!value || seen.has(value)) {
        continue;
      }

      seen.add(value);
      merged.push(value);
    }
  }

  return merged;
}

function normalizeRealtimeKeywordHints(keywords) {
  return mergeKeywordLists(keywords)
    .map((item) => String(item || "").trim())
    .filter((item) => item.length >= 2 && item.length <= 48)
    .slice(0, REALTIME_TRANSCRIPTION_KEYWORD_HINT_LIMIT);
}

async function extractPdfKeywordHints(workspace) {
  if (
    workspace?.sourceKind !== "pdf" ||
    !String(workspace?.artifacts?.sourceCopyPath || "").trim()
  ) {
    return [];
  }

  const sourcePath = path.resolve(workspace.artifacts.sourceCopyPath);
  const cachePath = path.join(
    workspace.workspacePath,
    REALTIME_TRANSCRIPTION_KEYWORD_CACHE_FILE_NAME,
  );

  try {
    const cached = await readJsonFile(cachePath, null);
    const cachedKeywords = normalizeRealtimeKeywordHints(cached?.keywords);
    if (cached?.sourcePath === sourcePath && cachedKeywords.length) {
      return cachedKeywords;
    }
  } catch (_error) {
    // Ignore malformed cache and rebuild it below.
  }

  const pythonExecutable = await getLiteSynphoniaPythonExecutable();
  const extractScript = `
import json
import re
import sys
from collections import Counter
from pathlib import Path

from lite_synphonia.pdf_matching.pdf_reader import read_pdf_document

PDF_PATH = Path(sys.argv[1]).expanduser().resolve()
TEXT_LIMIT = int(sys.argv[2])
KEYWORD_LIMIT = int(sys.argv[3])

ASCII_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was",
    "were", "have", "has", "had", "into", "onto", "your", "you", "our",
    "their", "his", "her", "its", "not", "but", "can", "will", "use",
}
ZH_STOP = {
    "我们", "你们", "他们", "这个", "那个", "一种", "进行", "可以", "以及",
    "如果", "因为", "所以", "或者", "然后", "就是", "一个", "一些", "主要",
    "相关", "内容", "部分", "其中", "通过", "实现", "使用", "进行", "问题",
}
PHRASE_SPLIT_RE = re.compile(
    r"[的了呢吗啊呀吧在是和与及并对将把被让向给由按为于等很都也还再更最所或及其]|"
    r"[^\\u3400-\\u4dbf\\u4e00-\\u9fff\\uf900-\\ufaffA-Za-z0-9_.+#-]+"
)

doc = read_pdf_document(PDF_PATH)
text = "\\n".join(page.text for page in doc.pages if page.text)[:TEXT_LIMIT]

ascii_freq = Counter()
ascii_display = {}
for token in re.findall(r"[A-Za-z][A-Za-z0-9_.+#-]{1,31}", text):
    normalized = token.lower()
    if len(normalized) < 3 or normalized in ASCII_STOP:
        continue
    if normalized.replace(".", "").replace("-", "").isdigit():
        continue
    ascii_freq[normalized] += 1
    ascii_display.setdefault(normalized, token)

zh_freq = Counter()
for phrase in PHRASE_SPLIT_RE.split(text):
    phrase = phrase.strip()
    if len(phrase) < 2 or len(phrase) > 10:
        continue
    if phrase in ZH_STOP:
        continue
    if not re.fullmatch(r"[\\u3400-\\u4dbf\\u4e00-\\u9fff\\uf900-\\ufaff]+", phrase):
        continue
    zh_freq[phrase] += 1

ranked_ascii = [
    ascii_display[key]
    for key, _ in sorted(
        ascii_freq.items(),
        key=lambda item: (-item[1], -len(item[0]), item[0]),
    )
]
ranked_zh = [
    key
    for key, _ in sorted(
        zh_freq.items(),
        key=lambda item: (-item[1], -len(item[0]), item[0]),
    )
]

keywords = []
seen = set()
for candidate in ranked_ascii + ranked_zh:
    value = str(candidate).strip()
    if not value or value in seen:
        continue
    seen.add(value)
    keywords.append(value)
    if len(keywords) >= KEYWORD_LIMIT:
        break

print(json.dumps({"keywords": keywords}, ensure_ascii=False))
  `.trim();

  const result = await new Promise((resolve) => {
    const child = spawn(
      pythonExecutable,
      [
        "-c",
        extractScript,
        sourcePath,
        String(REALTIME_TRANSCRIPTION_KEYWORD_TEXT_LIMIT),
        String(REALTIME_TRANSCRIPTION_KEYWORD_HINT_LIMIT),
      ],
      {
        cwd: projectRoot,
        stdio: ["ignore", "pipe", "pipe"],
      },
    );

    const stdoutChunks = [];
    const stderrChunks = [];
    child.stdout?.on("data", (chunk) => {
      stdoutChunks.push(chunk.toString("utf8"));
    });
    child.stderr?.on("data", (chunk) => {
      stderrChunks.push(chunk.toString("utf8"));
    });
    child.on("error", (error) => {
      resolve({
        ok: false,
        message: error.message,
      });
    });
    child.on("close", (code) => {
      if (code !== 0) {
        resolve({
          ok: false,
          message:
            stderrChunks.join("").trim() ||
            `提取 PDF 关键词失败，退出码 ${typeof code === "number" ? code : "unknown"}`,
        });
        return;
      }

      try {
        const payload = JSON.parse(stdoutChunks.join("").trim() || "{}");
        resolve({
          ok: true,
          keywords: normalizeRealtimeKeywordHints(payload?.keywords),
        });
      } catch (error) {
        resolve({
          ok: false,
          message:
            error instanceof Error
              ? error.message
              : "解析 PDF 关键词结果失败。",
        });
      }
    });
  });

  if (!result?.ok) {
    console.warn("[realtime-keywords]", result?.message || "提取 PDF 关键词失败。");
    return [];
  }

  const keywords = normalizeRealtimeKeywordHints(result.keywords);
  if (keywords.length) {
    await writeJsonFile(cachePath, {
      sourcePath,
      keywords,
      updatedAtUtc: new Date().toISOString(),
    }).catch(() => {});
  }

  return keywords;
}

function buildTranscriptArtifact(
  workspace,
  transcriptionPayload,
  transcriptText,
  processingStatus,
  previousArtifact,
) {
  const now = new Date().toISOString();
  const results = Array.isArray(transcriptionPayload?.results)
    ? transcriptionPayload.results
    : [];
  const previousSegments = Array.isArray(previousArtifact?.segments)
    ? previousArtifact.segments
    : [];
  const mergedTranscriptText = joinTextBlocks(
    previousArtifact?.transcriptText,
    transcriptText,
  );

  return {
    schemaVersion: "1.1",
    workspaceId: workspace.workspaceId,
    status: processingStatus,
    sourceFileName: workspace.sourceFileName,
    sourceKind: workspace.sourceKind,
    transcriptText: mergedTranscriptText,
    segments: [...previousSegments, ...results],
    createdAtUtc: previousArtifact?.createdAtUtc || workspace.createdAtUtc,
    updatedAtUtc: now,
    runtime: transcriptionPayload?.runtime || {},
    metrics: transcriptionPayload?.metrics || {},
    transcriptionQuality: transcriptionPayload?.transcription_quality || {},
    stageStatus: transcriptionPayload?.stage_status || {},
    liteSynphonia: transcriptionPayload || {},
  };
}

function buildSummaryArtifact(
  workspace,
  mergedSidebarItems,
  summaryWindowState,
  generatedSummaryResults,
  processingStatus,
  previousArtifact,
) {
  const now = new Date().toISOString();
  const mergedSummaryText = joinTextBlocks(
    previousArtifact?.summaryText,
    ...generatedSummaryResults.map((result) => result.summary),
  );

  return {
    schemaVersion: "1.2",
    workspaceId: workspace.workspaceId,
    status: processingStatus,
    sourceFileName: workspace.sourceFileName,
    sourceKind: workspace.sourceKind,
    summaryText: mergedSummaryText,
    items: mergedSidebarItems,
    keywords: mergeKeywordLists(
      previousArtifact?.keywords,
      ...generatedSummaryResults.map((result) => result.keywords),
    ),
    createdAtUtc: previousArtifact?.createdAtUtc || workspace.createdAtUtc,
    updatedAtUtc: now,
    runtime: {
      generatedWindowCount: generatedSummaryResults.length,
      totalWindowCount: mergedSidebarItems.length,
    },
    windowing: {
      triggerChars: summaryWindowState.triggerChars,
      overlapChars: summaryWindowState.overlapChars,
      pendingTranscript: summaryWindowState.pendingTranscript,
      pendingChars: summaryWindowState.pendingChars,
      generatedCount: summaryWindowState.generatedCount,
    },
    stageStatus: {
      stage: "summary",
      status: mergedSidebarItems.length ? "success" : "skipped",
      reason: mergedSidebarItems.length
        ? "Rolling summary windows updated successfully."
        : "Waiting for transcript buffer to reach the summary trigger size.",
      upstream_dependency: "transcription",
      quality_decision: "pass",
      details: {
        trigger_chars: summaryWindowState.triggerChars,
        overlap_chars: summaryWindowState.overlapChars,
        generated_count: summaryWindowState.generatedCount,
        pending_chars: summaryWindowState.pendingChars,
      },
    },
    liteSynphonia: previousArtifact?.liteSynphonia || {},
  };
}

function buildSummaryEmptyState(
  workspace,
  transcriptArtifact,
  summaryArtifact,
  sidebarState,
) {
  const sidebarItems = Array.isArray(sidebarState?.items)
    ? sidebarState.items
    : [];
  if (sidebarItems.length) {
    return undefined;
  }

  const transcriptText = String(
    transcriptArtifact?.transcriptText || "",
  ).trim();
  const blockedReason = String(
    transcriptArtifact?.stageStatus?.reason ||
      transcriptArtifact?.transcriptionQuality?.reason ||
      "",
  ).trim();
  const blockedStatus = String(
    transcriptArtifact?.stageStatus?.status || "",
  ).trim();
  const pendingChars = Number(sidebarState?.summaryWindow?.pendingChars) || 0;
  const triggerChars =
    Number(sidebarState?.summaryWindow?.triggerChars) ||
    SUMMARY_WINDOW_TRIGGER_CHARS;

  if (blockedStatus === "blocked" || blockedReason) {
    return {
      title: "本次未生成总结",
      copy: blockedReason
        ? `转录质量门禁已跳过总结阶段：${blockedReason}\n请确认麦克风录到了连续且有效的语音后重新开始转录。`
        : "本次转录结果未通过质量检查，系统已跳过总结阶段。",
      transcriptPreview: transcriptText || undefined,
    };
  }

  if (pendingChars > 0 && pendingChars < triggerChars) {
    return {
      title: "等待摘要触发",
      copy: `当前已累计 ${pendingChars} / ${triggerChars} 个有效字符。达到阈值后会触发一次摘要；如果累计至少 ${SUMMARY_IDLE_FLUSH_MIN_CHARS} 个有效字符并静音超过 ${Math.round(SUMMARY_IDLE_TIMEOUT_MS / 1000)} 秒，也会提前补刷一次摘要。`,
      transcriptPreview: transcriptText || undefined,
    };
  }

  if (
    workspace.status === "ready" &&
    !String(summaryArtifact?.summaryText || "").trim()
  ) {
    return {
      title: "暂无总结结果",
      copy: "本次流程已完成，但没有生成可展示的分段总结。请查看工作区日志和原始输出文件。",
      transcriptPreview: transcriptText || undefined,
    };
  }

  return undefined;
}

async function runLiteSynphoniaSummaryWindow(text, previousSummary) {
  const summaryText = String(text || "").trim();
  if (!summaryText) {
    return null;
  }

  let summaryLanguage = "auto";
  try {
    const { language } = await getDeepgramSettings();
    summaryLanguage = mapTranscriptionLanguageToSummaryLanguage(language);
  } catch (error) {
    console.warn(
      "[summary-window] 读取转录语言失败，回退 auto:",
      error instanceof Error ? error.message : String(error),
    );
  }

  const pythonExecutable = await getLiteSynphoniaPythonExecutable();
  const args = [
    "-m",
    "lite_synphonia",
    "summary-window",
    "--text",
    summaryText,
    "--provider",
    "deepseek",
    "--language",
    summaryLanguage,
  ];

  const prevSummary = String(previousSummary || "").trim();
  if (prevSummary) {
    args.push("--previous-summary", prevSummary);
  }

  const child = spawn(pythonExecutable, args, {
    cwd: projectRoot,
    stdio: ["ignore", "pipe", "pipe"],
  });

  const stdoutChunks = [];
  const stderrChunks = [];

  child.stdout?.on("data", (chunk) => {
    stdoutChunks.push(chunk.toString("utf8"));
  });
  child.stderr?.on("data", (chunk) => {
    stderrChunks.push(chunk.toString("utf8"));
  });

  const exitCode = await new Promise((resolve, reject) => {
    child.on("error", reject);
    child.on("close", resolve);
  });

  if (exitCode !== 0) {
    throw new Error(
      stderrChunks.join("").trim() ||
        `summary-window exited with code ${exitCode}.`,
    );
  }

  const raw = stdoutChunks.join("").trim();
  if (!raw) {
    throw new Error("summary-window returned an empty response.");
  }

  const payload = JSON.parse(raw);
  return {
    summary: String(payload?.summary || "").trim(),
    keywords: Array.isArray(payload?.keywords)
      ? payload.keywords
          .map((item) => String(item || "").trim())
          .filter(Boolean)
      : [],
    status: String(payload?.status || "").trim(),
    rawResponse: String(payload?.raw_response || "").trim(),
    errorMessage: String(payload?.error_message || "").trim(),
  };
}

function buildLiteSynphoniaRunLog(
  workspace,
  pipelineArgs,
  outputChunks,
  exitCode,
) {
  const renderedCommand = pipelineArgs.join(" ");
  const sections = [
    `[workspace] ${workspace.workspaceId}`,
    `[source] ${workspace.sourceFileName}`,
    `[command] ${renderedCommand}`,
  ];

  if (typeof exitCode === "number") {
    sections.push(`[exitCode] ${exitCode}`);
  }

  sections.push("", outputChunks.join(""));
  return sections.join("\n");
}

async function writeLiteSynphoniaRunLog(
  workspace,
  pipelineArgs,
  outputChunks,
  exitCode,
) {
  const logPath = path.join(
    workspace.workspacePath,
    LITESYNPHONIA_RUN_LOG_FILE_NAME,
  );
  const logContent = buildLiteSynphoniaRunLog(
    workspace,
    pipelineArgs,
    outputChunks,
    exitCode,
  );
  await fs.writeFile(logPath, logContent, "utf8");
  return logPath;
}

async function syncLiteSynphoniaArtifacts(workspace, processingStatus) {
  const transcriptionSourcePath = path.join(
    workspace.workspacePath,
    "transcription",
    "transcription.json",
  );
  const interfaceSourcePath = path.join(
    workspace.workspacePath,
    LITESYNPHONIA_INTERFACE_FILE_NAME,
  );

  const [transcriptionPayload, interfacePayload] = await Promise.all([
    readJsonFile(transcriptionSourcePath, null),
    readJsonFile(interfaceSourcePath, null),
  ]);

  if (!transcriptionPayload) {
    throw new Error("缺少 LiteSynphonia 转录结果文件。");
  }

  const transcriptText =
    String(interfacePayload?.transcription?.transcript_text || "").trim() ||
    buildTranscriptText(transcriptionPayload);
  const [
    previousTranscriptArtifact,
    previousSummaryArtifact,
    previousSidebarState,
  ] = await Promise.all([
    readJsonFile(workspace.artifacts.transcriptPath, null),
    readJsonFile(workspace.artifacts.summaryPath, null),
    readJsonFile(workspace.artifacts.sidebarStatePath, null),
  ]);
  const normalizedPreviousSidebarState = normalizeSidebarState(
    previousSidebarState || { items: [] },
  );
  let pendingTranscript = joinTextBlocks(
    normalizedPreviousSidebarState.summaryWindow?.pendingTranscript,
    transcriptText,
  );
  const sidebarItems = [];
  const generatedSummaryResults = [];
  let generatedCount =
    Number(normalizedPreviousSidebarState.summaryWindow?.generatedCount) ||
    normalizedPreviousSidebarState.items.length;

  while (countVisibleChars(pendingTranscript) >= SUMMARY_WINDOW_TRIGGER_CHARS) {
    const transcriptWindow = takeFirstVisibleChars(
      pendingTranscript,
      SUMMARY_WINDOW_TRIGGER_CHARS,
    );
    const summaryResult = await runLiteSynphoniaSummaryWindow(transcriptWindow);

    generatedCount += 1;
    generatedSummaryResults.push(summaryResult);
    sidebarItems.push({
      id: `summary-window-${generatedCount}`,
      summary: summaryResult?.summary || "",
      transcript: transcriptWindow,
      transcriptSegmentIds: [],
      transcriptRange: {
        startTime: null,
        endTime: null,
      },
    });

    const overflowTranscript = sliceAfterVisibleChars(
      pendingTranscript,
      SUMMARY_WINDOW_TRIGGER_CHARS,
    );
    const carryTranscript = takeLastVisibleChars(
      transcriptWindow,
      SUMMARY_WINDOW_OVERLAP_CHARS,
    );
    pendingTranscript = joinTextBlocks(carryTranscript, overflowTranscript);
  }

  const mergedSidebarItems = [
    ...normalizedPreviousSidebarState.items,
    ...sidebarItems,
  ];
  const summaryWindowState = {
    triggerChars: SUMMARY_WINDOW_TRIGGER_CHARS,
    overlapChars: SUMMARY_WINDOW_OVERLAP_CHARS,
    pendingTranscript,
    pendingChars: countVisibleChars(pendingTranscript),
    generatedCount,
  };
  console.log(
    `[summary-window] workspace=${workspace.workspaceId} pendingChars=${summaryWindowState.pendingChars} generated=${summaryWindowState.generatedCount} preview=${JSON.stringify(
      summaryWindowState.pendingTranscript.slice(0, 120),
    )}`,
  );
  const sidebarState = normalizeSidebarState({
    status: processingStatus,
    createdAtUtc:
      normalizedPreviousSidebarState.createdAtUtc || workspace.createdAtUtc,
    updatedAtUtc: new Date().toISOString(),
    items: mergedSidebarItems,
    summaryWindow: summaryWindowState,
  });
  const transcriptArtifact = buildTranscriptArtifact(
    workspace,
    transcriptionPayload,
    transcriptText,
    processingStatus,
    previousTranscriptArtifact,
  );
  const summaryArtifact = buildSummaryArtifact(
    workspace,
    mergedSidebarItems,
    summaryWindowState,
    generatedSummaryResults,
    processingStatus,
    previousSummaryArtifact,
  );

  await Promise.all([
    writeJsonFile(workspace.artifacts.transcriptPath, transcriptArtifact),
    writeJsonFile(workspace.artifacts.summaryPath, summaryArtifact),
    writeJsonFile(workspace.artifacts.sidebarStatePath, sidebarState),
  ]);

  return {
    transcriptArtifact,
    summaryArtifact,
    sidebarState,
  };
}

function inferProcessingStatusFromOutput(line) {
  const text = String(line || "");

  if (
    text.includes("麦克风预检") ||
    text.includes("skip-mic") ||
    text.includes("音频:")
  ) {
    return "recording";
  }

  if (text.includes("转录完成")) {
    return "summarizing";
  }

  if (text.includes("PDF 匹配中") || text.includes("匹配完成")) {
    return "matching";
  }

  if (text.includes("接口输出") || text.includes("全部完成")) {
    return "ready";
  }

  return null;
}

function createWorkspacePipelineController() {
  return {
    stopRequested: false,
    childProcess: null,
  };
}

async function finalizePausedWorkspace(cacheRootPath, workspaceId, workspace) {
  const finishedAtUtc = new Date().toISOString();
  const pausedWorkspace = createWorkspaceRecord({
    ...workspace,
    status: "paused",
    updatedAtUtc: finishedAtUtc,
  });
  await persistWorkspaceRecord(cacheRootPath, pausedWorkspace);
  const currentStatus = getWorkspacePipelineStatus(workspaceId, "paused");
  workspacePipelineControllers.delete(workspaceId);
  return setWorkspacePipelineStatus(workspaceId, {
    runState: "idle",
    processingStatus: "paused",
    finishedAtUtc,
    message: "已暂停监听。",
    completedCycles: currentStatus.completedCycles || 0,
  });
}

async function runNormalWorkspacePipelineCycle({
  cacheRootPath,
  workspaceId,
  workspace,
  recordSeconds,
}) {
  let controller = workspacePipelineControllers.get(workspaceId);
  if (!controller) {
    controller = createWorkspacePipelineController();
    workspacePipelineControllers.set(workspaceId, controller);
  }

  if (controller.stopRequested) {
    return finalizePausedWorkspace(cacheRootPath, workspaceId, workspace);
  }

  const cycleStartedAtUtc = new Date().toISOString();
  const activeWorkspace = createWorkspaceRecord({
    ...workspace,
    status: "recording",
    updatedAtUtc: cycleStartedAtUtc,
  });
  await persistWorkspaceRecord(cacheRootPath, activeWorkspace);

  const previousStatus = getWorkspacePipelineStatus(
    workspaceId,
    activeWorkspace.status,
  );
  setWorkspacePipelineStatus(workspaceId, {
    runState: "running",
    processingStatus: "recording",
    startedAtUtc: previousStatus.startedAtUtc || cycleStartedAtUtc,
    finishedAtUtc: undefined,
    message: `正在录制 ${recordSeconds} 秒音频...`,
    exitCode: undefined,
    completedCycles: previousStatus.completedCycles || 0,
  });

  let deepgramLanguage = "zh-CN";
  try {
    const settings = await getDeepgramSettings();
    deepgramLanguage = normalizeTranscriptionLanguage(
      settings.language,
      "zh-CN",
    );
  } catch (error) {
    console.warn(
      "[pipeline] 读取转录语言失败，回退 zh-CN:",
      error instanceof Error ? error.message : String(error),
    );
  }

  const pipelineArgs = [
    "-m",
    "lite_synphonia",
    "--activity-id",
    workspaceId,
    "--output-dir",
    activeWorkspace.workspacePath,
    "--seconds",
    String(recordSeconds),
    "--transcription-provider",
    "deepgram",
    "--transcription-model",
    "whisper-large",
    "--transcription-language",
    deepgramLanguage,
    "--summary-provider",
    "deepseek",
    "--skip-summary",
  ];

  if (
    activeWorkspace.sourceKind === "pdf" &&
    activeWorkspace.artifacts.sourceCopyPath
  ) {
    pipelineArgs.push(
      "--pdf-path",
      activeWorkspace.artifacts.sourceCopyPath,
      "--embedding-provider",
      "siliconflow-embed",
      "--embedding-model",
      "BAAI/bge-large-zh-v1.5",
      "--embedding-format",
      "openai",
    );
  }

  const pythonExecutable = await getLiteSynphoniaPythonExecutable();
  const spawnedCommand = [pythonExecutable, ...pipelineArgs];
  const childProcess = spawn(pythonExecutable, pipelineArgs, {
    cwd: projectRoot,
    stdio: ["ignore", "pipe", "pipe"],
  });

  controller.childProcess = childProcess;
  workspacePipelineControllers.set(workspaceId, controller);

  const stderrChunks = [];
  const outputChunks = [];

  function applyProgressStatus(nextProcessingStatus, message) {
    const progressStatus = setWorkspacePipelineStatus(workspaceId, {
      runState: "running",
      processingStatus: nextProcessingStatus,
      message,
    });

    if (nextProcessingStatus !== activeWorkspace.status) {
      void persistWorkspaceRecord(
        cacheRootPath,
        createWorkspaceRecord({
          ...activeWorkspace,
          status: nextProcessingStatus,
          updatedAtUtc: new Date().toISOString(),
        }),
      );
    }

    return progressStatus;
  }

  function handleOutputChunk(chunk, source) {
    const text = chunk.toString("utf8");
    outputChunks.push(`[${source}] ${text}`);
    if (source === "stderr") {
      stderrChunks.push(text);
    }

    for (const line of text.split(/\r?\n/)) {
      const trimmedLine = line.trim();
      if (!trimmedLine) {
        continue;
      }

      const inferredStatus = inferProcessingStatusFromOutput(trimmedLine);
      if (inferredStatus) {
        applyProgressStatus(inferredStatus, trimmedLine);
      } else {
        setWorkspacePipelineStatus(workspaceId, {
          message: trimmedLine,
        });
      }
    }
  }

  childProcess.stdout?.on("data", (chunk) => {
    handleOutputChunk(chunk, "stdout");
  });
  childProcess.stderr?.on("data", (chunk) => {
    handleOutputChunk(chunk, "stderr");
  });

  childProcess.on("error", async (error) => {
    const finishedAtUtc = new Date().toISOString();
    const logPath = await writeLiteSynphoniaRunLog(
      activeWorkspace,
      spawnedCommand,
      outputChunks,
      undefined,
    );
    const failedWorkspace = createWorkspaceRecord({
      ...activeWorkspace,
      status: "failed",
      updatedAtUtc: finishedAtUtc,
    });

    workspacePipelineControllers.delete(workspaceId);
    await persistWorkspaceRecord(cacheRootPath, failedWorkspace);
    setWorkspacePipelineStatus(workspaceId, {
      runState: "failed",
      processingStatus: "failed",
      finishedAtUtc,
      message: `LiteSynphonia 启动失败: ${error.message}\n日志文件: ${logPath}`,
    });
  });

  childProcess.on("close", async (code, signal) => {
    const currentController = workspacePipelineControllers.get(workspaceId);
    if (currentController) {
      currentController.childProcess = null;
      workspacePipelineControllers.set(workspaceId, currentController);
    }

    const finishedAtUtc = new Date().toISOString();
    const logPath = await writeLiteSynphoniaRunLog(
      activeWorkspace,
      spawnedCommand,
      outputChunks,
      typeof code === "number" ? code : undefined,
    );

    if (code === 0) {
      try {
        const latestStatus = getWorkspacePipelineStatus(
          workspaceId,
          activeWorkspace.status,
        );
        const completedCycles = (latestStatus.completedCycles || 0) + 1;
        const shouldContinue = !currentController?.stopRequested;
        const nextProcessingStatus = shouldContinue
          ? "recording"
          : currentController?.stopRequested
            ? "paused"
            : "ready";

        await syncLiteSynphoniaArtifacts(activeWorkspace, nextProcessingStatus);
        const completedWorkspace = createWorkspaceRecord({
          ...activeWorkspace,
          status: nextProcessingStatus,
          updatedAtUtc: finishedAtUtc,
        });
        await persistWorkspaceRecord(cacheRootPath, completedWorkspace);

        if (shouldContinue) {
          setWorkspacePipelineStatus(workspaceId, {
            runState: "running",
            processingStatus: "recording",
            finishedAtUtc: undefined,
            exitCode: undefined,
            completedCycles,
            message: `第 ${completedCycles} 段已完成，继续监听中...`,
          });
          setTimeout(() => {
            void runNormalWorkspacePipelineCycle({
              cacheRootPath,
              workspaceId,
              workspace: completedWorkspace,
              recordSeconds,
            });
          }, 250);
          return;
        }

        workspacePipelineControllers.delete(workspaceId);
        setWorkspacePipelineStatus(workspaceId, {
          runState: currentController?.stopRequested ? "idle" : "succeeded",
          processingStatus: nextProcessingStatus,
          finishedAtUtc,
          exitCode: code,
          completedCycles,
          message: currentController?.stopRequested
            ? "已暂停监听。"
            : "转录、总结和结果同步已完成。",
        });
      } catch (error) {
        const failedWorkspace = createWorkspaceRecord({
          ...activeWorkspace,
          status: "failed",
          updatedAtUtc: finishedAtUtc,
        });
        workspacePipelineControllers.delete(workspaceId);
        await persistWorkspaceRecord(cacheRootPath, failedWorkspace);
        setWorkspacePipelineStatus(workspaceId, {
          runState: "failed",
          processingStatus: "failed",
          finishedAtUtc,
          exitCode: code,
          message:
            error instanceof Error
              ? `工作区结果同步失败: ${error.message}\n日志文件: ${logPath}`
              : `工作区结果同步失败。\n日志文件: ${logPath}`,
        });
      }
      return;
    }

    if (currentController?.stopRequested) {
      workspacePipelineControllers.delete(workspaceId);
      const pausedWorkspace = createWorkspaceRecord({
        ...activeWorkspace,
        status: "paused",
        updatedAtUtc: finishedAtUtc,
      });
      await persistWorkspaceRecord(cacheRootPath, pausedWorkspace);
      setWorkspacePipelineStatus(workspaceId, {
        runState: "idle",
        processingStatus: "paused",
        finishedAtUtc,
        exitCode: typeof code === "number" ? code : undefined,
        completedCycles:
          getWorkspacePipelineStatus(workspaceId, "paused").completedCycles ||
          0,
        message: signal ? `已暂停监听（${signal}）。` : "已暂停监听。",
      });
      return;
    }

    const failedWorkspace = createWorkspaceRecord({
      ...activeWorkspace,
      status: "failed",
      updatedAtUtc: finishedAtUtc,
    });
    workspacePipelineControllers.delete(workspaceId);
    await persistWorkspaceRecord(cacheRootPath, failedWorkspace);
    const stderrMessage = stderrChunks.join("").trim();
    setWorkspacePipelineStatus(workspaceId, {
      runState: "failed",
      processingStatus: "failed",
      finishedAtUtc,
      exitCode: typeof code === "number" ? code : undefined,
      message: `${
        stderrMessage ||
        `LiteSynphonia 运行失败，退出码 ${
          typeof code === "number" ? code : "unknown"
        }。`
      }\n日志文件: ${logPath}`,
    });
  });
}

async function createMainWindow() {
  const mainWindow = new BrowserWindow({
    width: 1480,
    height: 940,
    minWidth: 1180,
    minHeight: 760,
    backgroundColor: "#0f1115",
    autoHideMenuBar: true,
    ...(process.platform === "darwin"
      ? {
          titleBarStyle: "hidden",
          trafficLightPosition: {
            x: 18,
            y: 15,
          },
        }
      : {}),
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, "preload.cjs"),
    },
  });

  if (devServerUrl) {
    await mainWindow.loadURL(new URL(appEntryHtml, devServerUrl).toString());
    return;
  }

  await mainWindow.loadFile(path.join(projectRoot, appBuildDir, appEntryHtml));
}

ipcMain.handle("workspace:create-documents-folder", async (_, folderName) => {
  const nextName = String(folderName || "").trim();
  if (!nextName) {
    return { ok: false, message: "工作区名称不能为空。" };
  }

  const folderPath = getDocumentsFolderPath(nextName);
  await ensureDirectory(folderPath);

  return {
    ok: true,
    folderPath,
    displayPath: getDocumentsDisplayPath(nextName),
  };
});

ipcMain.handle("workspace:rename-documents-folder", async (_, payload) => {
  const currentName = String(payload?.currentName || "").trim();
  const nextName = String(payload?.nextName || "").trim();

  if (!currentName || !nextName) {
    return { ok: false, message: "工作区名称不能为空。" };
  }

  const currentPath = getDocumentsFolderPath(currentName);
  const nextPath = getDocumentsFolderPath(nextName);

  if (currentPath !== nextPath) {
    try {
      await fs.rename(currentPath, nextPath);
    } catch (error) {
      if (error?.code === "ENOENT") {
        await ensureDirectory(nextPath);
      } else {
        return { ok: false, message: "重命名工作区文件夹失败。" };
      }
    }
  }

  return {
    ok: true,
    folderPath: nextPath,
    displayPath: getDocumentsDisplayPath(nextName),
  };
});

ipcMain.handle("workspace:pick-existing-folder", async () => {
  const result = await dialog.showOpenDialog({
    title: "选择已有工作区文件夹",
    properties: ["openDirectory"],
  });

  if (result.canceled || !result.filePaths.length) {
    return null;
  }

  const folderPath = result.filePaths[0];

  return {
    name: path.basename(folderPath),
    folderPath,
  };
});

ipcMain.handle("workspace:reveal-folder", async (_, payload) => {
  const sourceType = payload?.sourceType;
  const folderName = String(payload?.folderName || "").trim();
  let folderPath = String(payload?.folderPath || "").trim();

  if (sourceType === "documents") {
    if (!folderName) {
      return { ok: false, message: "缺少工作区名称。" };
    }

    folderPath = getDocumentsFolderPath(folderName);
    await ensureDirectory(folderPath);
  }

  if (!folderPath) {
    return { ok: false, message: "没有可显示的文件夹路径。" };
  }

  const openError = await shell.openPath(folderPath);
  if (openError) {
    return { ok: false, message: openError };
  }

  return { ok: true };
});

ipcMain.handle("settings:get-lite-synphonia-provider-settings", async () => {
  try {
    const settings = await getLiteSynphoniaProviderSettingsSnapshot();
    return { ok: true, settings };
  } catch (error) {
    return {
      ok: false,
      message:
        error instanceof Error
          ? error.message
          : "读取 LiteSynphonia 配置失败。",
    };
  }
});

ipcMain.handle(
  "settings:save-lite-synphonia-provider-settings",
  async (_, payload) => {
    try {
      const settings = await saveLiteSynphoniaProviderSettings(payload);
      return { ok: true, settings };
    } catch (error) {
      return {
        ok: false,
        message:
          error instanceof Error
            ? error.message
            : "保存 LiteSynphonia 配置失败。",
      };
    }
  },
);

ipcMain.handle("llm:ask-course-question", async (_, payload) => {
  return askDeepSeekCourseQuestion(payload);
});

ipcMain.handle("workspace:create-normal-workspace", async (_, payload) => {
  const fileName = String(payload?.fileName || "").trim();
  const mimeType = String(payload?.mimeType || "").trim();
  const sourceKind = detectSourceKind(fileName, mimeType);
  const fileBytes = toBuffer(payload?.bytes);

  if (!fileName) {
    return { ok: false, message: "缺少上传文件名。" };
  }

  if (!["pdf", "ppt", "pptx"].includes(sourceKind)) {
    return { ok: false, message: "当前只支持 PDF、PPT 或 PPTX 文件。" };
  }

  if (!fileBytes.length) {
    return { ok: false, message: "上传文件内容为空，无法创建缓存工作区。" };
  }

  const workspaceId = randomUUID();
  const workspaceName = getDefaultWorkspaceDisplayName(fileName);
  const workspaceFolderName = buildWorkspaceFolderName(fileName);
  const cacheRootPath = getWorkspaceCacheRoot();
  const workspacePath = path.join(cacheRootPath, workspaceFolderName);
  const now = new Date().toISOString();
  const safeSourceFileName =
    sanitizePathSegment(path.parse(fileName).name) +
    path.extname(fileName).toLowerCase();

  const sourceCopyPath = path.join(
    workspacePath,
    safeSourceFileName || fileName,
  );
  const transcriptPath = path.join(workspacePath, TRANSCRIPT_FILE_NAME);
  const summaryPath = path.join(workspacePath, SUMMARY_FILE_NAME);
  const sidebarStatePath = path.join(workspacePath, SIDEBAR_STATE_FILE_NAME);

  await ensureDirectory(workspacePath);

  await Promise.all([
    fs.writeFile(sourceCopyPath, fileBytes),
    writeJsonFile(
      transcriptPath,
      buildTranscriptSeed(workspaceId, fileName, sourceKind),
    ),
    writeJsonFile(
      summaryPath,
      buildSummarySeed(workspaceId, fileName, sourceKind),
    ),
    writeJsonFile(
      sidebarStatePath,
      normalizeSidebarState(payload?.sidebarState),
    ),
  ]);

  const workspace = createWorkspaceRecord({
    workspaceId,
    workspaceName,
    workspacePath,
    cacheRootPath,
    sourceFileName: fileName,
    sourceKind,
    status: "initialized",
    createdAtUtc: now,
    updatedAtUtc: now,
    lastOpenedAtUtc: now,
    artifacts: {
      transcriptPath,
      summaryPath,
      sourceCopyPath,
      sidebarStatePath,
    },
  });

  const currentIndex = await syncWorkspaceIndex(cacheRootPath);
  const nextIndex = sortWorkspaces([
    workspace,
    ...currentIndex.filter(
      (item) => item.workspaceId !== workspace.workspaceId,
    ),
  ]);
  await writeWorkspaceIndex(cacheRootPath, nextIndex);

  return {
    ok: true,
    workspace,
  };
});

ipcMain.handle("workspace:list-normal-workspaces", async () => {
  const cacheRootPath = getWorkspaceCacheRoot();
  const workspaces = await syncWorkspaceIndex(cacheRootPath);

  return {
    ok: true,
    workspaces,
  };
});

ipcMain.handle("workspace:open-normal-workspace", async (_, workspaceId) => {
  const nextWorkspaceId = String(workspaceId || "").trim();
  if (!nextWorkspaceId) {
    return { ok: false, message: "缺少工作区标识。" };
  }

  const cacheRootPath = getWorkspaceCacheRoot();
  const indexedWorkspaces = await syncWorkspaceIndex(cacheRootPath);
  const matchedWorkspace = indexedWorkspaces.find(
    (item) => item.workspaceId === nextWorkspaceId,
  );

  if (!matchedWorkspace) {
    return { ok: false, message: "没有找到对应的缓存工作区。" };
  }

  const openedAtUtc = new Date().toISOString();
  const workspace = createWorkspaceRecord({
    ...matchedWorkspace,
    lastOpenedAtUtc: openedAtUtc,
  });
  const nextIndex = sortWorkspaces([
    workspace,
    ...indexedWorkspaces.filter(
      (item) => item.workspaceId !== workspace.workspaceId,
    ),
  ]);

  await writeWorkspaceIndex(cacheRootPath, nextIndex);

  const [sidebarState, sourceBytes, transcriptArtifact, summaryArtifact] =
    await Promise.all([
      readJsonFile(
        workspace.artifacts.sidebarStatePath,
        normalizeSidebarState({ items: [] }),
      ),
      workspace.artifacts.sourceCopyPath
        ? fs
            .readFile(workspace.artifacts.sourceCopyPath)
            .catch(() => Buffer.alloc(0))
        : Promise.resolve(Buffer.alloc(0)),
      readJsonFile(workspace.artifacts.transcriptPath, null),
      readJsonFile(workspace.artifacts.summaryPath, null),
    ]);
  const normalizedSidebarState = normalizeSidebarState(sidebarState);

  return {
    ok: true,
    workspace,
    sidebarState: normalizedSidebarState,
    summaryWindowState: normalizedSidebarState.summaryWindow,
    summaryEmptyState: buildSummaryEmptyState(
      workspace,
      transcriptArtifact,
      summaryArtifact,
      normalizedSidebarState,
    ),
    sourceFile: {
      fileName: workspace.sourceFileName,
      mimeType: getMimeTypeForKind(workspace.sourceKind),
      bytes: new Uint8Array(sourceBytes),
    },
  };
});

ipcMain.handle("workspace:rename-normal-workspace", async (_, payload) => {
  const workspaceId = String(payload?.workspaceId || "").trim();
  const workspaceName = getDefaultWorkspaceDisplayName(
    payload?.workspaceName || "",
  );

  if (!workspaceId) {
    return { ok: false, message: "缺少工作区标识。" };
  }

  const cacheRootPath = getWorkspaceCacheRoot();
  const indexedWorkspaces = await syncWorkspaceIndex(cacheRootPath);
  const matchedWorkspace = indexedWorkspaces.find(
    (item) => item.workspaceId === workspaceId,
  );

  if (!matchedWorkspace) {
    return { ok: false, message: "没有找到对应的工作区。" };
  }

  const workspace = createWorkspaceRecord({
    ...matchedWorkspace,
    workspaceName,
    updatedAtUtc: new Date().toISOString(),
  });
  const nextIndex = sortWorkspaces([
    workspace,
    ...indexedWorkspaces.filter(
      (item) => item.workspaceId !== workspace.workspaceId,
    ),
  ]);
  await writeWorkspaceIndex(cacheRootPath, nextIndex);

  return {
    ok: true,
    workspace,
  };
});

ipcMain.handle("workspace:star-normal-workspace", async (_, payload) => {
  const workspaceId = String(payload?.workspaceId || "").trim();
  const starred = Boolean(payload?.starred);

  if (!workspaceId) {
    return { ok: false, message: "缺少工作区标识。" };
  }

  const cacheRootPath = getWorkspaceCacheRoot();
  const indexedWorkspaces = await syncWorkspaceIndex(cacheRootPath);
  const matchedWorkspace = indexedWorkspaces.find(
    (item) => item.workspaceId === workspaceId,
  );

  if (!matchedWorkspace) {
    return { ok: false, message: "没有找到对应的工作区。" };
  }

  const workspace = createWorkspaceRecord({
    ...matchedWorkspace,
    starred,
    updatedAtUtc: new Date().toISOString(),
  });
  const nextIndex = sortWorkspaces([
    workspace,
    ...indexedWorkspaces.filter((item) => item.workspaceId !== workspaceId),
  ]);
  await writeWorkspaceIndex(cacheRootPath, nextIndex);

  return {
    ok: true,
    workspace,
  };
});

ipcMain.handle("workspace:delete-normal-workspace", async (_, workspaceId) => {
  const nextWorkspaceId = String(workspaceId || "").trim();

  if (!nextWorkspaceId) {
    return { ok: false, message: "缺少工作区标识。" };
  }

  const cacheRootPath = getWorkspaceCacheRoot();
  const indexedWorkspaces = await syncWorkspaceIndex(cacheRootPath);
  const matchedWorkspace = indexedWorkspaces.find(
    (item) => item.workspaceId === nextWorkspaceId,
  );

  if (!matchedWorkspace) {
    return { ok: false, message: "没有找到对应的工作区。" };
  }

  await fs.rm(matchedWorkspace.workspacePath, {
    recursive: true,
    force: true,
  });

  const nextIndex = indexedWorkspaces.filter(
    (item) => item.workspaceId !== matchedWorkspace.workspaceId,
  );
  await writeWorkspaceIndex(cacheRootPath, nextIndex);

  return {
    ok: true,
    workspaceId: matchedWorkspace.workspaceId,
  };
});

ipcMain.handle("workspace:start-normal-pipeline", async (_, payload) => {
  const workspaceId = String(payload?.workspaceId || "").trim();
  const recordSeconds = Math.min(
    600,
    Math.max(1, Number(payload?.recordSeconds) || DEFAULT_RECORD_SECONDS),
  );

  if (!workspaceId) {
    return { ok: false, message: "缺少工作区标识。" };
  }

  const cacheRootPath = getWorkspaceCacheRoot();
  const { workspace } = await getIndexedWorkspace(cacheRootPath, workspaceId);

  if (!workspace) {
    return { ok: false, message: "没有找到对应的工作区。" };
  }

  const currentPipelineStatus = getWorkspacePipelineStatus(
    workspaceId,
    workspace.status,
  );
  if (currentPipelineStatus.runState === "running") {
    return {
      ok: false,
      message: "当前工作区已有正在执行的转录任务。",
      status: currentPipelineStatus,
    };
  }

  const dependencyResult =
    await checkLiteSynphoniaPythonDependencies(workspace);
  if (!dependencyResult.ok) {
    return {
      ok: false,
      message: dependencyResult.message,
    };
  }

  const prerequisiteResult =
    await checkWorkspacePipelinePrerequisites(workspace);
  if (!prerequisiteResult.ok) {
    return {
      ok: false,
      message: prerequisiteResult.message,
    };
  }

  const startedAtUtc = new Date().toISOString();
  const initialStatus = setWorkspacePipelineStatus(workspaceId, {
    runState: "running",
    processingStatus: "recording",
    startedAtUtc,
    finishedAtUtc: undefined,
    message: `正在录制 ${recordSeconds} 秒音频...`,
    exitCode: undefined,
    completedCycles: currentPipelineStatus.completedCycles || 0,
  });
  workspacePipelineControllers.set(
    workspaceId,
    createWorkspacePipelineController(),
  );
  void runNormalWorkspacePipelineCycle({
    cacheRootPath,
    workspaceId,
    workspace,
    recordSeconds,
  });

  return {
    ok: true,
    status: initialStatus,
  };
});

ipcMain.handle("workspace:pause-normal-pipeline", async (_, workspaceId) => {
  const nextWorkspaceId = String(workspaceId || "").trim();

  if (!nextWorkspaceId) {
    return { ok: false, message: "缺少工作区标识。" };
  }

  const cacheRootPath = getWorkspaceCacheRoot();
  const { workspace } = await getIndexedWorkspace(
    cacheRootPath,
    nextWorkspaceId,
  );

  if (!workspace) {
    return { ok: false, message: "没有找到对应的工作区。" };
  }

  const currentStatus = getWorkspacePipelineStatus(
    nextWorkspaceId,
    workspace.status,
  );
  const controller = workspacePipelineControllers.get(nextWorkspaceId);

  if (!controller || currentStatus.runState !== "running") {
    return {
      ok: true,
      status: currentStatus,
    };
  }

  controller.stopRequested = true;
  if (controller.childProcess && !controller.childProcess.killed) {
    try {
      controller.childProcess.kill("SIGTERM");
    } catch (_error) {
      // Ignore kill races; close handler will reconcile final state.
    }
  }
  workspacePipelineControllers.set(nextWorkspaceId, controller);

  const nextStatus = setWorkspacePipelineStatus(nextWorkspaceId, {
    runState: "running",
    processingStatus: currentStatus.processingStatus,
    message: "将在当前片段结束后暂停监听。",
    completedCycles: currentStatus.completedCycles || 0,
  });

  return {
    ok: true,
    status: nextStatus,
  };
});

ipcMain.handle(
  "workspace:get-normal-pipeline-status",
  async (_, workspaceId) => {
    const nextWorkspaceId = String(workspaceId || "").trim();

    if (!nextWorkspaceId) {
      return { ok: false, message: "缺少工作区标识。" };
    }

    const cacheRootPath = getWorkspaceCacheRoot();
    const { workspace } = await getIndexedWorkspace(
      cacheRootPath,
      nextWorkspaceId,
    );

    if (!workspace) {
      return { ok: false, message: "没有找到对应的工作区。" };
    }

    return {
      ok: true,
      status: getWorkspacePipelineStatus(nextWorkspaceId, workspace.status),
    };
  },
);

// ── Realtime streaming STT via Deepgram WebSocket ──────────────────
// Architecture: renderer captures mic audio via AudioWorklet, sends
// PCM16 chunks through IPC. Main process maintains a single WebSocket
// per workspace to Deepgram's streaming endpoint. Partial/final results
// are relayed back to renderer via webContents.send().

const realtimeSessions = new Map();

function emitRealtimeEvent(workspaceId, event) {
  for (const win of BrowserWindow.getAllWindows()) {
    win.webContents.send("realtime-transcription-event", {
      workspaceId,
      ...event,
    });
  }
}

async function getDeepgramSettings() {
  const settings = await getLiteSynphoniaProviderSettingsSnapshot();
  return {
    apiKey: settings.deepgramApiKey,
    language: settings.transcriptionLanguage || "zh-CN",
  };
}

async function getDeepgramApiKey() {
  return (await getDeepgramSettings()).apiKey;
}

function buildDeepgramStreamingUrl(
  sampleRate,
  keywords,
  model = "nova-2",
  language = "zh-CN",
) {
  const params = new URLSearchParams({
    model,
    language,
    smart_format: "true",
    punctuate: "true",
    interim_results: "true",
    // Deepgram live transcription requires utterance_end_ms >= 1000.
    utterance_end_ms: "1000",
    vad_events: "true",
    endpointing: "120",
    encoding: "linear16",
    sample_rate: String(sampleRate),
    channels: "1",
  });
  if (Array.isArray(keywords) && keywords.length > 0) {
    for (const kw of keywords) {
      params.append("keywords", kw);
    }
  }
  return `wss://api.deepgram.com/v1/listen?${params.toString()}`;
}

const REALTIME_TRANSCRIPTION_MODEL = "nova-2";

function createRealtimeSession(
  workspaceId,
  chunkDurationMs,
  sampleRate,
  keywords,
) {
  return {
    workspaceId,
    chunkDurationMs,
    sampleRate,
    keywords: keywords || [],
    model: REALTIME_TRANSCRIPTION_MODEL,
    ws: null,
    partialText: "",
    finalTranscriptText: "",
    finalSegments: [],
    timedSegments: [],
    timelineCursorSec: 0,
    matching: {
      enabled: false,
      workspacePath: "",
      pdfPath: "",
      resultPath: "",
      transcriptionPayloadPath: "",
      cacheDir: "",
      providerName: "siliconflow-embed",
      modelId: APP_MANAGED_PROVIDER_PRESETS["siliconflow-embed"].modelId,
      format: "openai",
      timer: null,
      running: false,
      pending: false,
      lastMatchedPage: null,
      lastError: "",
    },
    reconnectAttempts: 0,
    maxReconnectAttempts: 5,
    shouldReconnect: true,
    permanentErrorMessage: "",
    pendingChunks: [],
    keepAliveTimer: null,
  };
}

function extractTimedSegmentsFromTranscriptArtifact(transcriptArtifact) {
  const rawSegments = Array.isArray(transcriptArtifact?.segments)
    ? transcriptArtifact.segments
    : [];

  return rawSegments
    .map((segment, index) => ({
      segmentId: String(segment?.id || `history-${index}`),
      text: String(segment?.text || "").trim(),
      startTime: Number(segment?.t0 ?? segment?.start ?? Number.NaN),
      endTime: Number(segment?.t1 ?? segment?.end ?? Number.NaN),
    }))
    .filter(
      (segment) =>
        segment.text &&
        Number.isFinite(segment.startTime) &&
        Number.isFinite(segment.endTime),
    )
    .map((segment) => ({
      ...segment,
      endTime: Math.max(segment.startTime, segment.endTime),
    }))
    .slice(-REALTIME_PDF_MATCH_MAX_SEGMENTS);
}

function resolveRealtimeSegmentTiming(session, text, rawMessage) {
  const safeText = String(text || "").trim();
  if (!safeText) {
    return null;
  }

  let startTime = Number(rawMessage?.start);
  let endTime = Number(rawMessage?.duration);
  if (Number.isFinite(startTime) && Number.isFinite(endTime)) {
    endTime = startTime + Math.max(0, endTime);
  } else {
    const words = Array.isArray(rawMessage?.channel?.alternatives?.[0]?.words)
      ? rawMessage.channel.alternatives[0].words
      : [];
    if (words.length) {
      const first = words[0];
      const last = words[words.length - 1];
      startTime = Number(first?.start);
      endTime = Number(last?.end);
    }
  }

  if (!Number.isFinite(startTime) || !Number.isFinite(endTime)) {
    const fallbackStart = Number(session.timelineCursorSec) || 0;
    const estimatedDuration = Math.min(
      8,
      Math.max(0.8, safeText.replace(/\s+/g, "").length / 12),
    );
    startTime = fallbackStart;
    endTime = fallbackStart + estimatedDuration;
  }

  const previousEnd =
    session.timedSegments.length > 0
      ? Number(session.timedSegments[session.timedSegments.length - 1].endTime)
      : 0;
  const normalizedStart = Math.max(previousEnd, Number(startTime) || 0);
  const normalizedEnd = Math.max(normalizedStart + 0.25, Number(endTime) || 0);
  session.timelineCursorSec = normalizedEnd;

  return {
    startTime: normalizedStart,
    endTime: normalizedEnd,
  };
}

function appendRealtimeTimedSegment(session, segment, rawMessage) {
  const timing = resolveRealtimeSegmentTiming(session, segment?.text, rawMessage);
  if (!timing) {
    return;
  }

  session.timedSegments.push({
    segmentId: String(segment?.id || randomUUID()),
    text: String(segment?.text || "").trim(),
    startTime: timing.startTime,
    endTime: timing.endTime,
  });

  if (session.timedSegments.length > REALTIME_PDF_MATCH_MAX_SEGMENTS) {
    session.timedSegments.splice(
      0,
      session.timedSegments.length - REALTIME_PDF_MATCH_MAX_SEGMENTS,
    );
  }
}

function clearRealtimePdfMatchTimer(session) {
  if (session?.matching?.timer) {
    clearTimeout(session.matching.timer);
    session.matching.timer = null;
  }
}

async function upsertWorkspacePdfMatchingPayload(workspacePath, pdfMatchingPayload) {
  const mergedResultsPath = path.join(
    workspacePath,
    LITESYNPHONIA_MERGED_RESULTS_FILE_NAME,
  );
  const existingPayload = await readJsonFile(mergedResultsPath, null);
  const previous =
    existingPayload && typeof existingPayload === "object" ? existingPayload : {};

  const stageStatus =
    previous?.stage_status && typeof previous.stage_status === "object"
      ? previous.stage_status
      : {};

  const nextPayload = {
    ...previous,
    pdf_matching: pdfMatchingPayload,
    stage_status: {
      ...stageStatus,
      pdf_matching: pdfMatchingPayload?.stage_status || {},
    },
  };

  await writeJsonFile(mergedResultsPath, nextPayload);
}

async function runRealtimePdfMatching(session) {
  if (!session?.matching?.enabled) {
    return;
  }

  if (session.timedSegments.length < REALTIME_PDF_MATCH_MIN_SEGMENTS) {
    return;
  }

  const transcriptionPayload = {
    results: session.timedSegments.map((segment) => ({
      text: segment.text,
      t0: Number(segment.startTime),
      t1: Number(segment.endTime),
    })),
  };
  await writeJsonFile(
    session.matching.transcriptionPayloadPath,
    transcriptionPayload,
  );

  const pythonExecutable = await getLiteSynphoniaPythonExecutable();
  const args = [
    "-m",
    "lite_synphonia",
    "pdf-match",
    "--pdf-path",
    session.matching.pdfPath,
    "--transcription-json",
    session.matching.transcriptionPayloadPath,
    "--output-json",
    session.matching.resultPath,
    "--embedding-provider",
    session.matching.providerName,
    "--embedding-model",
    session.matching.modelId,
    "--embedding-format",
    session.matching.format,
    "--pdf-cache-dir",
    session.matching.cacheDir,
  ];

  const runResult = await new Promise((resolve) => {
    const child = spawn(pythonExecutable, args, {
      cwd: projectRoot,
      stdio: ["ignore", "pipe", "pipe"],
    });

    const stdoutChunks = [];
    const stderrChunks = [];
    child.stdout?.on("data", (chunk) => stdoutChunks.push(chunk.toString("utf8")));
    child.stderr?.on("data", (chunk) => stderrChunks.push(chunk.toString("utf8")));
    child.on("error", (error) => {
      resolve({
        ok: false,
        message: `启动 PDF 匹配命令失败: ${error.message}`,
      });
    });
    child.on("close", (code) => {
      if (code === 0) {
        resolve({
          ok: true,
          stdout: stdoutChunks.join("").trim(),
        });
        return;
      }

      resolve({
        ok: false,
        message:
          stderrChunks.join("").trim() ||
          `PDF 匹配命令失败，退出码 ${typeof code === "number" ? code : "unknown"}。`,
      });
    });
  });

  if (!runResult.ok) {
    throw new Error(runResult.message || "PDF 匹配执行失败。");
  }

  const pdfMatchingPayload = await readJsonFile(session.matching.resultPath, null);
  if (!pdfMatchingPayload) {
    throw new Error("PDF 匹配命令未生成结果文件。");
  }

  await upsertWorkspacePdfMatchingPayload(
    session.matching.workspacePath,
    pdfMatchingPayload,
  );
  const snapshot = buildPageMatchSnapshot({
    pdf_matching: pdfMatchingPayload,
  });
  if (
    snapshot.currentPage &&
    snapshot.currentPage > 0 &&
    snapshot.currentPage !== session.matching.lastMatchedPage
  ) {
    session.matching.lastMatchedPage = snapshot.currentPage;
    emitRealtimeEvent(session.workspaceId, {
      type: "page_match",
      status: "streaming",
      matchedPage: snapshot.currentPage,
      message: `页码匹配已定位到第 ${snapshot.currentPage} 页。`,
    });
  }
}

function scheduleRealtimePdfMatching(session) {
  if (!session?.matching?.enabled) {
    return;
  }

  if (session.timedSegments.length < REALTIME_PDF_MATCH_MIN_SEGMENTS) {
    return;
  }

  clearRealtimePdfMatchTimer(session);
  session.matching.timer = setTimeout(() => {
    session.matching.timer = null;
    void flushRealtimePdfMatching(session);
  }, REALTIME_PDF_MATCH_DEBOUNCE_MS);
}

async function flushRealtimePdfMatching(session) {
  if (!session?.matching?.enabled) {
    return;
  }

  if (session.matching.running) {
    session.matching.pending = true;
    return;
  }

  session.matching.running = true;
  try {
    await runRealtimePdfMatching(session);
    session.matching.lastError = "";
  } catch (error) {
    const nextError =
      error instanceof Error ? error.message : "实时 PDF 匹配失败。";
    if (session.matching.lastError !== nextError) {
      console.warn("[realtime-pdf-match]", nextError);
    }
    session.matching.lastError = nextError;
  } finally {
    session.matching.running = false;
    if (session.matching.pending) {
      session.matching.pending = false;
      scheduleRealtimePdfMatching(session);
    }
  }
}

async function connectDeepgramWebSocket(session) {
  let WebSocketCtor;
  try {
    ({ WebSocket: WebSocketCtor } = await import("ws"));
  } catch {
    emitRealtimeEvent(session.workspaceId, {
      type: "error",
      status: "error",
      lastError: "缺少实时转写依赖 ws，当前无法启动流式 STT。",
    });
    return;
  }

  const { apiKey, language: deepgramLanguage } = await getDeepgramSettings();
  if (!apiKey) {
    emitRealtimeEvent(session.workspaceId, {
      type: "error",
      status: "error",
      lastError: "Deepgram API Key 未配置，请在设置中填写。",
    });
    return;
  }

  // ── 首次连接前验证 Key（只在第一次连接时验证，不在重连时重复）──────
  if (session.reconnectAttempts === 0) {
    try {
      const { default: https } = await import("node:https");
      await new Promise((resolve, reject) => {
        const req = https.request(
          "https://api.deepgram.com/v1/auth/token",
          { method: "GET", headers: { Authorization: `Token ${apiKey}` }, timeout: 8000 },
          (res) => {
            if (res.statusCode === 200) {
              resolve(null);
            } else {
              const codes = { 401: "API Key 无效或已过期", 403: "API Key 权限不足", 429: "调用频率超限" };
              reject(new Error(codes[res.statusCode] || `HTTP ${res.statusCode}`));
            }
          },
        );
        req.on("error", reject);
        req.on("timeout", () => reject(new Error("验证请求超时，请检查网络连接")));
        req.end();
      });
    } catch (verifyErr) {
      emitRealtimeEvent(session.workspaceId, {
        type: "error",
        status: "error",
        lastError: `Deepgram Key 验证失败: ${verifyErr.message}`,
      });
      return;
    }
  }

  emitRealtimeEvent(session.workspaceId, {
    type: "status",
    status: "connecting",
    message: "正在连接 Deepgram 流式转写服务...",
  });

  const url = buildDeepgramStreamingUrl(session.sampleRate, session.keywords, session.model, deepgramLanguage);
  console.log("[deepgram] connecting to:", url);
  const ws = new WebSocketCtor(url, {
    headers: { Authorization: `Token ${apiKey}` },
  });

  session.ws = ws;

  ws.on("open", () => {
    session.reconnectAttempts = 0;
    emitRealtimeEvent(session.workspaceId, {
      type: "status",
      status: "listening",
      message: "已连接，等待语音输入...",
    });
    for (const chunk of session.pendingChunks) {
      if (ws.readyState === WEBSOCKET_OPEN) {
        ws.send(chunk);
      }
    }
    session.pendingChunks = [];

    // Send KeepAlive every 8s to prevent Deepgram from closing idle connections
    if (session.keepAliveTimer) {
      clearInterval(session.keepAliveTimer);
    }
    session.keepAliveTimer = setInterval(() => {
      if (ws.readyState === WEBSOCKET_OPEN) {
        ws.send(JSON.stringify({ type: "KeepAlive" }));
      }
    }, 8000);
  });

  ws.on("message", (data) => {
    try {
      const msg = JSON.parse(data.toString());

      if (msg.type === "Results") {
        const transcript = msg.channel?.alternatives?.[0]?.transcript || "";
        const isFinal = msg.is_final === true;

        if (!isFinal) {
          session.partialText = transcript;
          emitRealtimeEvent(session.workspaceId, {
            type: "partial",
            status: "streaming",
            partialText: transcript,
          });
        } else if (transcript.trim()) {
          const segment = {
            id: randomUUID(),
            text: transcript.trim(),
            finalizedAtUtc: new Date().toISOString(),
          };
          session.finalSegments.push(segment);
          appendRealtimeTimedSegment(session, segment, msg);
          session.finalTranscriptText +=
            (session.finalTranscriptText ? "\n" : "") + segment.text;
          session.partialText = "";

          emitRealtimeEvent(session.workspaceId, {
            type: "final",
            status: "streaming",
            finalSegment: segment,
            finalTranscriptText: session.finalTranscriptText,
            partialText: "",
          });

          const cacheRootPath = getWorkspaceCacheRoot();
          getIndexedWorkspace(cacheRootPath, session.workspaceId)
            .then(({ workspace }) => {
              if (workspace) {
                void appendTranscriptToSidebar(workspace, segment);
              }
            })
            .catch(() => {});
          scheduleRealtimePdfMatching(session);
        }
      }

      if (msg.type === "UtteranceEnd") {
        if (session.partialText.trim()) {
          const segment = {
            id: randomUUID(),
            text: session.partialText.trim(),
            finalizedAtUtc: new Date().toISOString(),
          };
          session.finalSegments.push(segment);
          appendRealtimeTimedSegment(session, segment, msg);
          session.finalTranscriptText +=
            (session.finalTranscriptText ? "\n" : "") + segment.text;
          session.partialText = "";
          emitRealtimeEvent(session.workspaceId, {
            type: "final",
            status: "streaming",
            finalSegment: segment,
            finalTranscriptText: session.finalTranscriptText,
            partialText: "",
          });

          const cacheRootPath = getWorkspaceCacheRoot();
          getIndexedWorkspace(cacheRootPath, session.workspaceId)
            .then(({ workspace }) => {
              if (workspace) {
                void appendTranscriptToSidebar(workspace, segment);
              }
            })
            .catch(() => {});
          scheduleRealtimePdfMatching(session);
        }
      }

      // Deepgram API 错误消息（参数不兼容、鉴权失败等）
      if (msg.type === "Error" || msg.error) {
        const errMsg = msg.message || msg.error || JSON.stringify(msg);
        console.error("[deepgram] API error:", errMsg);
        emitRealtimeEvent(session.workspaceId, {
          type: "error",
          status: "error",
          lastError: `Deepgram 错误: ${errMsg}`,
        });
      }
    } catch (_err) {
      // ignore malformed messages
    }
  });

  ws.on("close", (code, reason) => {
    const reasonStr = reason?.toString?.() || "";
    console.error(`[deepgram] ws closed  code=${code}  reason=${reasonStr}`);
    session.ws = null;
    if (session.keepAliveTimer) {
      clearInterval(session.keepAliveTimer);
      session.keepAliveTimer = null;
    }

    if (session.permanentErrorMessage) {
      realtimeSessions.delete(session.workspaceId);
      setWorkspacePipelineStatus(session.workspaceId, {
        runState: "failed",
        processingStatus: "failed",
        message: session.permanentErrorMessage,
        finishedAtUtc: new Date().toISOString(),
      });
      return;
    }

    // Deepgram 4xxx = 永久性错误（鉴权失败、参数不合法等），不再重连
    const isPermanentError = code >= 4000 && code < 5000;
    if (isPermanentError) {
      const detail = reasonStr || `关闭码 ${code}`;
      realtimeSessions.delete(session.workspaceId);
      const errorMessage = `Deepgram 拒绝连接 (${code}): ${detail}`;
      setWorkspacePipelineStatus(session.workspaceId, {
        runState: "failed",
        processingStatus: "failed",
        message: errorMessage,
        finishedAtUtc: new Date().toISOString(),
      });
      emitRealtimeEvent(session.workspaceId, {
        type: "error",
        status: "error",
        lastError: errorMessage,
      });
      return;
    }

    if (
      session.shouldReconnect &&
      session.reconnectAttempts < session.maxReconnectAttempts
    ) {
      session.reconnectAttempts++;
      const delay = Math.min(
        1000 * Math.pow(2, session.reconnectAttempts - 1),
        8000,
      );
      emitRealtimeEvent(session.workspaceId, {
        type: "status",
        status: "connecting",
        message: `连接断开 (${code}${reasonStr ? ": " + reasonStr : ""})，${(delay / 1000).toFixed(0)}s 后重连 (${session.reconnectAttempts}/${session.maxReconnectAttempts})...`,
      });
      setTimeout(() => {
        if (session.shouldReconnect) {
          void connectDeepgramWebSocket(session);
        }
      }, delay);
    } else if (session.shouldReconnect) {
      emitRealtimeEvent(session.workspaceId, {
        type: "error",
        status: "error",
        lastError: `WebSocket 连接失败，已达最大重连次数 (${session.maxReconnectAttempts})。最后关闭码: ${code}`,
      });
    } else {
      emitRealtimeEvent(session.workspaceId, {
        type: "stopped",
        status: "idle",
        message: "实时转写已停止。",
      });
    }
  });

  ws.on("error", (err) => {
    console.error("[deepgram] ws error:", err.message, err.code || "");
    if (String(err.message || "").includes("Unexpected server response: 400")) {
      session.shouldReconnect = false;
      session.permanentErrorMessage =
        "Deepgram 握手请求被拒绝 (HTTP 400)。当前请求参数不合法，已停止自动重连。";
      emitRealtimeEvent(session.workspaceId, {
        type: "error",
        status: "error",
        lastError: session.permanentErrorMessage,
      });
    }
  });
}

const realtimeSummaryLocks = new Map();
const realtimeIdleTimers = new Map();
const realtimeLastSummaries = new Map();

async function generateSummaryFromWindow(
  workspace,
  sidebarState,
  pendingTranscript,
  generatedCount,
  clearAfter,
) {
  const workspaceId = workspace.workspaceId;
  const previousSummary = realtimeLastSummaries.get(workspaceId) || "";

  if (clearAfter) {
    const transcriptWindow = pendingTranscript;
    if (!countVisibleChars(transcriptWindow))
      return { pendingTranscript, generatedCount };

    let summaryResult;
    try {
      summaryResult = await runLiteSynphoniaSummaryWindow(
        transcriptWindow,
        previousSummary,
      );
    } catch (err) {
      console.error("[summary-window] 生成总结失败:", err.message);
      return { pendingTranscript, generatedCount };
    }

    generatedCount += 1;
    const newItem = {
      id: `summary-window-${generatedCount}`,
      summary: summaryResult?.summary || "",
      transcript: transcriptWindow,
      transcriptSegmentIds: [],
      transcriptRange: { startTime: null, endTime: null },
    };

    if (!Array.isArray(sidebarState.items)) sidebarState.items = [];
    sidebarState.items.push(newItem);
    realtimeLastSummaries.set(workspaceId, summaryResult?.summary || "");

    emitRealtimeEvent(workspaceId, {
      type: "summary",
      status: "streaming",
      summaryItem: newItem,
    });

    pendingTranscript = "";
    return { pendingTranscript, generatedCount };
  }

  while (countVisibleChars(pendingTranscript) >= SUMMARY_WINDOW_TRIGGER_CHARS) {
    const transcriptWindow = takeFirstVisibleChars(
      pendingTranscript,
      SUMMARY_WINDOW_TRIGGER_CHARS,
    );

    let summaryResult;
    try {
      summaryResult = await runLiteSynphoniaSummaryWindow(
        transcriptWindow,
        previousSummary,
      );
    } catch (err) {
      console.error("[summary-window] 生成总结失败:", err.message);
      break;
    }

    generatedCount += 1;
    const newItem = {
      id: `summary-window-${generatedCount}`,
      summary: summaryResult?.summary || "",
      transcript: transcriptWindow,
      transcriptSegmentIds: [],
      transcriptRange: { startTime: null, endTime: null },
    };

    if (!Array.isArray(sidebarState.items)) sidebarState.items = [];
    sidebarState.items.push(newItem);
    realtimeLastSummaries.set(workspaceId, summaryResult?.summary || "");

    emitRealtimeEvent(workspaceId, {
      type: "summary",
      status: "streaming",
      summaryItem: newItem,
    });

    const overflowTranscript = sliceAfterVisibleChars(
      pendingTranscript,
      SUMMARY_WINDOW_TRIGGER_CHARS,
    );
    const carryTranscript = takeLastVisibleChars(
      transcriptWindow,
      SUMMARY_WINDOW_OVERLAP_CHARS,
    );
    pendingTranscript = joinTextBlocks(carryTranscript, overflowTranscript);
  }

  return { pendingTranscript, generatedCount };
}

async function flushSummaryWindow(workspace, options = {}) {
  const workspaceId = workspace.workspaceId;
  if (realtimeSummaryLocks.get(workspaceId)) return;
  const minVisibleChars =
    typeof options?.minVisibleChars === "number" &&
    Number.isFinite(options.minVisibleChars)
      ? options.minVisibleChars
      : 0;

  const sidebarPath = workspace.artifacts?.sidebarStatePath;
  if (!sidebarPath) return;

  let sidebarState;
  try {
    sidebarState = JSON.parse(await fs.readFile(sidebarPath, "utf8"));
  } catch {
    return;
  }

  const pendingTranscript = String(
    sidebarState.summaryWindow?.pendingTranscript || "",
  ).trim();
  const visibleChars = countVisibleChars(pendingTranscript);
  if (!visibleChars || visibleChars < minVisibleChars) return;

  realtimeSummaryLocks.set(workspaceId, true);
  try {
    let generatedCount =
      Number(sidebarState.summaryWindow?.generatedCount) ||
      (Array.isArray(sidebarState.items) ? sidebarState.items.length : 0);

    const result = await generateSummaryFromWindow(
      workspace,
      sidebarState,
      pendingTranscript,
      generatedCount,
      true,
    );

    sidebarState.summaryWindow = {
      ...sidebarState.summaryWindow,
      triggerChars: SUMMARY_WINDOW_TRIGGER_CHARS,
      overlapChars: SUMMARY_WINDOW_OVERLAP_CHARS,
      pendingTranscript: result.pendingTranscript,
      pendingChars: countVisibleChars(result.pendingTranscript),
      generatedCount: result.generatedCount,
    };

    await writeJsonFile(sidebarPath, sidebarState);
  } finally {
    realtimeSummaryLocks.delete(workspaceId);
  }
}

async function appendTranscriptToSidebar(workspace, segment) {
  try {
    const sidebarPath = workspace.artifacts?.sidebarStatePath;
    if (!sidebarPath) return;
    const workspaceId = workspace.workspaceId;

    let sidebarState;
    try {
      sidebarState = JSON.parse(await fs.readFile(sidebarPath, "utf8"));
    } catch {
      sidebarState = normalizeSidebarState(null);
    }

    if (!Array.isArray(sidebarState.finalSegments)) {
      sidebarState.finalSegments = [];
    }
    sidebarState.finalSegments.push(segment);
    sidebarState.finalTranscriptText =
      (sidebarState.finalTranscriptText || "") +
      (sidebarState.finalTranscriptText ? "\n" : "") +
      segment.text;
    sidebarState.updatedAtUtc = new Date().toISOString();
    sidebarState.schemaVersion = "2.0";

    let pendingTranscript =
      (sidebarState.summaryWindow?.pendingTranscript || "") + segment.text;
    let generatedCount =
      Number(sidebarState.summaryWindow?.generatedCount) ||
      (Array.isArray(sidebarState.items) ? sidebarState.items.length : 0);

    // Reset idle timer on each new segment
    if (realtimeIdleTimers.has(workspaceId)) {
      clearTimeout(realtimeIdleTimers.get(workspaceId));
    }
    realtimeIdleTimers.set(
      workspaceId,
      setTimeout(() => {
        realtimeIdleTimers.delete(workspaceId);
        void flushSummaryWindow(workspace, {
          minVisibleChars: SUMMARY_IDLE_FLUSH_MIN_CHARS,
        });
      }, SUMMARY_IDLE_TIMEOUT_MS),
    );

    // Sliding window: when pending text >= trigger chars, generate summary
    if (
      countVisibleChars(pendingTranscript) >= SUMMARY_WINDOW_TRIGGER_CHARS &&
      !realtimeSummaryLocks.get(workspaceId)
    ) {
      realtimeSummaryLocks.set(workspaceId, true);

      try {
        const result = await generateSummaryFromWindow(
          workspace,
          sidebarState,
          pendingTranscript,
          generatedCount,
          false,
        );
        pendingTranscript = result.pendingTranscript;
        generatedCount = result.generatedCount;
      } finally {
        realtimeSummaryLocks.delete(workspaceId);
      }
    }

    sidebarState.summaryWindow = {
      ...sidebarState.summaryWindow,
      triggerChars: SUMMARY_WINDOW_TRIGGER_CHARS,
      overlapChars: SUMMARY_WINDOW_OVERLAP_CHARS,
      pendingTranscript,
      pendingChars: countVisibleChars(pendingTranscript),
      generatedCount,
    };

    await writeJsonFile(sidebarPath, sidebarState);
  } catch (_err) {
    // best-effort persistence
  }
}

ipcMain.handle("workspace:start-realtime-transcription", async (_, payload) => {
  const workspaceId = String(payload?.workspaceId || "").trim();
  const chunkDurationMs = [50, 100, 150, 200, 250, 300].includes(
    payload?.chunkDurationMs,
  )
    ? payload.chunkDurationMs
    : 200;
  const sampleRate = Number(payload?.sampleRate) || 16000;
  const payloadKeywords = Array.isArray(payload?.keywords) ? payload.keywords : [];

  if (!workspaceId) {
    return { ok: false, message: "缺少工作区标识。" };
  }

  const existingSession = realtimeSessions.get(workspaceId);
  if (existingSession?.ws) {
    return { ok: false, message: "当前工作区已有正在执行的实时转写。" };
  }

  const cacheRootPath = getWorkspaceCacheRoot();
  const { workspace } = await getIndexedWorkspace(cacheRootPath, workspaceId);
  if (!workspace) {
    return { ok: false, message: "没有找到对应的工作区。" };
  }

  const pdfKeywordHints = await extractPdfKeywordHints(workspace);
  const keywords = normalizeRealtimeKeywordHints(
    mergeKeywordLists(payloadKeywords, pdfKeywordHints),
  );

  const session = createRealtimeSession(
    workspaceId,
    chunkDurationMs,
    sampleRate,
    keywords,
  );

  const transcriptArtifact = await readJsonFile(workspace.artifacts.transcriptPath, null);
  session.timedSegments = extractTimedSegmentsFromTranscriptArtifact(
    transcriptArtifact,
  );
  if (session.timedSegments.length) {
    session.timelineCursorSec =
      Number(session.timedSegments[session.timedSegments.length - 1].endTime) || 0;
  }

  if (workspace.sourceKind === "pdf" && workspace.artifacts?.sourceCopyPath) {
    session.matching.enabled = true;
    session.matching.workspacePath = workspace.workspacePath;
    session.matching.pdfPath = workspace.artifacts.sourceCopyPath;
    session.matching.resultPath = path.join(
      workspace.workspacePath,
      "pdf_match",
      REALTIME_PDF_MATCH_RESULT_FILE_NAME,
    );
    session.matching.transcriptionPayloadPath = path.join(
      workspace.workspacePath,
      REALTIME_PDF_MATCH_TRANSCRIPTION_FILE_NAME,
    );
    session.matching.cacheDir = path.join(
      workspace.workspacePath,
      ".pdf_embed_cache",
    );
    session.matching.providerName = "siliconflow-embed";
    session.matching.modelId =
      APP_MANAGED_PROVIDER_PRESETS["siliconflow-embed"].modelId;
    session.matching.format = "openai";

    if (session.timedSegments.length >= REALTIME_PDF_MATCH_MIN_SEGMENTS) {
      scheduleRealtimePdfMatching(session);
    }
  }

  realtimeSessions.set(workspaceId, session);
  void connectDeepgramWebSocket(session);

  const startedAtUtc = new Date().toISOString();
  const status = setWorkspacePipelineStatus(workspaceId, {
    runState: "running",
    processingStatus: "connecting",
    startedAtUtc,
    message: "正在启动实时流式转写...",
  });

  return { ok: true, status };
});

ipcMain.handle("workspace:push-realtime-audio-chunk", async (_, payload) => {
  const workspaceId = String(payload?.workspaceId || "").trim();
  const session = realtimeSessions.get(workspaceId);
  if (!session) return;

  let audioBuffer;
  if (payload?.audio instanceof Uint8Array) {
    audioBuffer = Buffer.from(payload.audio);
  } else if (payload?.audio?.data) {
    audioBuffer = Buffer.from(payload.audio.data);
  } else {
    return;
  }

  if (session.ws && session.ws.readyState === WEBSOCKET_OPEN) {
    session.ws.send(audioBuffer);
  } else {
    session.pendingChunks.push(audioBuffer);
    if (session.pendingChunks.length > 50) {
      session.pendingChunks.shift();
    }
  }
});

ipcMain.handle(
  "workspace:stop-realtime-transcription",
  async (_, workspaceId) => {
    const nextWorkspaceId = String(workspaceId || "").trim();
    if (!nextWorkspaceId) {
      return { ok: false, message: "缺少工作区标识。" };
    }

    const session = realtimeSessions.get(nextWorkspaceId);
    if (session) {
      session.shouldReconnect = false;
      clearRealtimePdfMatchTimer(session);
      session.matching.pending = false;
      if (session.keepAliveTimer) {
        clearInterval(session.keepAliveTimer);
        session.keepAliveTimer = null;
      }
      if (session.ws) {
        try {
          session.ws.send(JSON.stringify({ type: "CloseStream" }));
        } catch {}
        try {
          session.ws.close();
        } catch {}
      }
      realtimeSessions.delete(nextWorkspaceId);
    }

    // Clear idle timer
    if (realtimeIdleTimers.has(nextWorkspaceId)) {
      clearTimeout(realtimeIdleTimers.get(nextWorkspaceId));
      realtimeIdleTimers.delete(nextWorkspaceId);
    }

    const cacheRootPath = getWorkspaceCacheRoot();
    const { workspace } = await getIndexedWorkspace(
      cacheRootPath,
      nextWorkspaceId,
    );

    // Flush remaining window content before stopping
    if (workspace) {
      try {
        await flushSummaryWindow(workspace);
      } catch (err) {
        console.error(
          "[stop-realtime] flush summary window failed:",
          err.message,
        );
      }
    }

    if (workspace) {
      await persistWorkspaceRecord(
        cacheRootPath,
        createWorkspaceRecord({
          ...workspace,
          status: "paused",
          updatedAtUtc: new Date().toISOString(),
        }),
      );
    }

    // Clean up last summary tracking
    realtimeLastSummaries.delete(nextWorkspaceId);

    const status = setWorkspacePipelineStatus(nextWorkspaceId, {
      runState: "idle",
      processingStatus: "paused",
      message: "实时转写已停止。",
      finishedAtUtc: new Date().toISOString(),
    });

    return { ok: true, status };
  },
);

// ── IPC: PPT/PPTX → PDF 转换 ─────────────────────────────────────
ipcMain.handle("workspace:convert-pptx-to-pdf", async (_, payload) => {
  const workspaceId = String(payload?.workspaceId || "").trim();
  if (!workspaceId) {
    return { ok: false, message: "缺少工作区标识。" };
  }

  const cacheRootPath = getWorkspaceCacheRoot();
  const { workspace } = await getIndexedWorkspace(cacheRootPath, workspaceId);
  if (!workspace) {
    return { ok: false, message: "没有找到对应的工作区。" };
  }

  if (!workspace.artifacts.sourceCopyPath) {
    return { ok: false, message: "工作区没有源文件副本。" };
  }

  if (workspace.sourceKind === "pdf") {
    return { ok: false, message: "当前文件已经是 PDF，无需转换。" };
  }

  try {
    const result = await convertPptxToPdf(
      workspace.artifacts.sourceCopyPath,
      workspace.workspacePath,
    );

    if (!result.ok) {
      return result;
    }

    // 更新工作区元数据：源类型改为 pdf，sourceCopyPath 改为 pdf
    const updatedWorkspace = createWorkspaceRecord({
      ...workspace,
      sourceKind: "pdf",
      artifacts: {
        ...workspace.artifacts,
        sourceCopyPath: result.pdfPath,
      },
      updatedAtUtc: new Date().toISOString(),
    });
    await persistWorkspaceRecord(cacheRootPath, updatedWorkspace);

    return {
      ok: true,
      pdfBytes: result.pdfBytes ? Array.from(result.pdfBytes) : [],
      workspace: updatedWorkspace,
    };
  } catch (error) {
    return {
      ok: false,
      message: error instanceof Error ? error.message : "PPT 转换失败。",
    };
  }
});

// ── IPC: 读取页码匹配结果 ─────────────────────────────────────────
ipcMain.handle("workspace:get-page-match", async (_, workspaceId) => {
  const nextWorkspaceId = String(workspaceId || "").trim();
  if (!nextWorkspaceId) {
    return { ok: false, message: "缺少工作区标识。" };
  }

  const cacheRootPath = getWorkspaceCacheRoot();
  const { workspace } = await getIndexedWorkspace(
    cacheRootPath,
    nextWorkspaceId,
  );
  if (!workspace) {
    return { ok: false, message: "没有找到对应的工作区。" };
  }

  try {
    const pageMatch = await readWorkspacePageMatchTimeline(workspace);
    return { ok: true, pageMatch };
  } catch (error) {
    return {
      ok: false,
      message: error instanceof Error ? error.message : "读取页码匹配失败。",
    };
  }
});

// ── IPC: 知识图谱数据 ─────────────────────────────────────────────
ipcMain.handle("knowledge:get-data", async (_, payload) => {
  try {
    const selectedActivityId = String(payload?.selectedActivityId || "").trim();
    const result = await exportKnowledgeBaseV2Data(selectedActivityId);
    return {
      ok: true,
      data: result.data,
      ingestResults: result.ingestResults,
    };
  } catch (error) {
    return {
      ok: false,
      message:
        error instanceof Error ? error.message : "导出知识库 V2 数据失败。",
    };
  }
});

ipcMain.handle("knowledge:get-graph", async () => {
  try {
    const cacheRootPath = getWorkspaceCacheRoot();
    const graph = await buildActivityKnowledgeGraph(cacheRootPath);
    return { ok: true, graph };
  } catch (error) {
    return {
      ok: false,
      message:
        error instanceof Error ? error.message : "构建知识图谱失败。",
    };
  }
});

app.whenReady().then(async () => {
  await createMainWindow();

  app.on("activate", async () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      await createMainWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
