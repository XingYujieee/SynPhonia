import type {
  ApiConfig,
  AskFileQuestionPayload,
  AskFileQuestionResult,
  CourseFile,
} from "../types";
import { detectCourseFileKind } from "../utils/courseFiles";

const API_CONFIG: ApiConfig = {
  baseUrl: "http://localhost:8000/api",
  useMock: true,
  endpoints: {
    upload: "/files/upload",
    chat: "/qa/ask",
  },
};

const wait = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

function createUrl(path: string): string {
  return `${API_CONFIG.baseUrl}${path}`;
}

async function request<TResponse>(path: string, options: RequestInit = {}): Promise<TResponse> {
  const response = await fetch(createUrl(path), {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers ?? {}),
    },
    ...options,
  });

  if (!response.ok) {
    throw new Error(`请求失败: ${response.status}`);
  }

  return (await response.json()) as TResponse;
}

/**
 * Upload a course file and return the minimal metadata required by the UI.
 * The mock branch mirrors the real contract so the frontend can be integrated
 * with backend services later without rewriting the view layer.
 */
export async function uploadCourseFile(file: File): Promise<CourseFile> {
  if (API_CONFIG.useMock) {
    await wait(300);
    const kind = detectCourseFileKind(file.name, file.type);
    const sourceBytes = new Uint8Array(await file.arrayBuffer());

    return {
      id: crypto.randomUUID(),
      name: file.name,
      type: file.type || "application/octet-stream",
      size: file.size,
      previewUrl: kind === "pdf" ? URL.createObjectURL(file) : "",
      sourceBytes,
      kind,
      extractedText: `Mock 文本索引已生成，可用于右侧问答。文件名: ${file.name}`,
      folderId: "",
    };
  }

  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(createUrl(API_CONFIG.endpoints.upload), {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`文件上传失败: ${response.status}`);
  }

  const uploaded = (await response.json()) as CourseFile;

  return {
    ...uploaded,
    kind: uploaded.kind ?? detectCourseFileKind(uploaded.name, uploaded.type || ""),
  };
}

/**
 * Ask a question against the currently selected PDF.
 * Keeping this service small makes it straightforward to replace the mock
 * branch with the real backend contract later.
 */
export async function askFileQuestion(
  payload: AskFileQuestionPayload,
): Promise<AskFileQuestionResult> {
  if (window.desktopBridge?.askCourseQuestion) {
    return window.desktopBridge.askCourseQuestion(payload);
  }

  if (API_CONFIG.useMock) {
    await wait(650);

    return {
      answer: `根据当前 PDF 内容，你的问题“${payload.question}”大概率与第 ${Math.max(
        1,
        Math.ceil(Math.random() * 15),
      )} 页相关。接真接口后，这里可以返回精确页码、原文片段和解释。`,
      citations: ["当前文件文本索引"],
    };
  }

  return request<AskFileQuestionResult>(API_CONFIG.endpoints.chat, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export const apiConfig = API_CONFIG;
