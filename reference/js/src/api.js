const API_CONFIG = {
  baseUrl: "http://localhost:8000/api",
  useMock: true,
  endpoints: {
    upload: "/files/upload",
    chat: "/qa/ask",
  },
};

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function createUrl(path) {
  return `${API_CONFIG.baseUrl}${path}`;
}

async function request(path, options = {}) {
  const response = await fetch(createUrl(path), {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    throw new Error(`请求失败: ${response.status}`);
  }

  return response.json();
}

export async function uploadCourseFile(file) {
  if (API_CONFIG.useMock) {
    await wait(300);
    const isPdf =
      (file.type && file.type.includes("pdf")) || file.name.toLowerCase().endsWith(".pdf");

    return {
      id: crypto.randomUUID(),
      name: file.name,
      type: file.type || "application/octet-stream",
      size: file.size,
      previewUrl: isPdf ? URL.createObjectURL(file) : "",
      extractedText: `Mock 文本索引已生成，可用于右侧问答。文件名: ${file.name}`,
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

  return response.json();
}

export async function askFileQuestion(payload) {
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

  return request(API_CONFIG.endpoints.chat, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export const apiConfig = API_CONFIG;
