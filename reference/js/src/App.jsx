import { useEffect, useRef, useState } from "react";
import { askFileQuestion, uploadCourseFile } from "./api.js";
import PdfPreview from "./PdfPreview.jsx";

const CURRENT_FOLDER_ID = "folder-current";
const THEME_STORAGE_KEY = "course-pdf-theme";

const INITIAL_FOLDERS = [
  { id: CURRENT_FOLDER_ID, name: "当前课堂资料", expanded: true },
  { id: "folder-ml", name: "机器学习导论", expanded: false },
  { id: "folder-network", name: "计算机网络", expanded: false },
];

const INITIAL_FILES = [
  { id: "demo-1", folderId: "folder-ml", name: "机器学习课件-01.pdf" },
  { id: "demo-2", folderId: "folder-network", name: "TCP协议详解.pdf" },
];

const INITIAL_MESSAGES = [
  {
    role: "assistant",
    text: "上传一份 PDF 之后，就可以在右侧直接针对这份文件提问。",
  },
];

const INITIAL_SUMMARIES = [
  {
    id: "summary-1",
    summary:
      "老师先从课程目标切入，强调这节课要建立整体框架，再逐步拆解关键概念之间的关系。",
    transcript:
      "我们这一节先不要急着记细节，先把整门课的框架搭起来。你要先知道这个问题为什么出现，它解决的核心矛盾是什么，后面每一个公式和结论才有位置。",
  },
  {
    id: "summary-2",
    summary:
      "中段主要在解释核心定义，并反复提醒要把概念和具体场景联系起来理解，而不是机械背诵。",
    transcript:
      "定义本身不难，难的是你能不能在具体情境里认出来。考试或者实际使用的时候，题目不会直接把定义贴给你，所以你必须知道这个概念在真实问题里长什么样。",
  },
  {
    id: "summary-3",
    summary:
      "结尾部分回到应用层面，要求同学把今天的内容和前面章节串联，形成可以复述的知识链路。",
    transcript:
      "回去之后你们要自己复述一遍，从前面讲过的基础概念开始，到今天这个部分怎么接上，再到它能解决什么问题。只要这条链路能讲顺，这一块就算真的学会了。",
  },
];

function PanelToggleIcon({ mirrored = false }) {
  return (
    <svg
      viewBox="0 0 1024 1024"
      width="18"
      height="18"
      fill="currentColor"
      style={{ transform: mirrored ? "scaleX(-1)" : "none" }}
    >
      <path d="M824.888889 170.666667H199.111111a56.888889 56.888889 0 0 0-56.888889 56.888889v568.888888a56.888889 56.888889 0 0 0 56.888889 56.888889h625.777778a56.888889 56.888889 0 0 0 56.888889-56.888889V227.555556a56.888889 56.888889 0 0 0-56.888889-56.888889z m0 597.333333a28.444444 28.444444 0 0 1-28.444445 28.444444H227.555556a28.444444 28.444444 0 0 1-28.444445-28.444444V256a28.444444 28.444444 0 0 1 28.444445-28.444444h568.888888a28.444444 28.444444 0 0 1 28.444445 28.444444z" />
      <path d="M512 256m28.444444 0l227.555556 0q28.444444 0 28.444444 28.444444l0 455.111112q0 28.444444-28.444444 28.444444l-227.555556 0q-28.444444 0-28.444444-28.444444l0-455.111112q0-28.444444 28.444444-28.444444Z" />
    </svg>
  );
}

function App() {
  const [theme, setTheme] = useState(
    () => localStorage.getItem(THEME_STORAGE_KEY) || "light",
  );
  const [viewMode, setViewMode] = useState("normal");
  const [isLeftOpen, setIsLeftOpen] = useState(true);
  const [isRightOpen, setIsRightOpen] = useState(true);
  const [folders, setFolders] = useState(INITIAL_FOLDERS);
  const [files, setFiles] = useState(INITIAL_FILES);
  const [selectedFolderId, setSelectedFolderId] = useState(CURRENT_FOLDER_ID);
  const [currentFileId, setCurrentFileId] = useState("");
  const [expandedSummaryId, setExpandedSummaryId] = useState(INITIAL_SUMMARIES[0].id);
  const [chatMessages, setChatMessages] = useState(INITIAL_MESSAGES);
  const [chatInput, setChatInput] = useState("");
  const [isAsking, setIsAsking] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const chatMessagesRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  useEffect(() => {
    if (chatMessagesRef.current) {
      chatMessagesRef.current.scrollTop = chatMessagesRef.current.scrollHeight;
    }
  }, [chatMessages]);

  const currentFile = files.find((file) => file.id === currentFileId) || null;

  function handleToggleFolder(folderId) {
    setFolders((prev) =>
      prev.map((folder) =>
        folder.id === folderId
          ? { ...folder, expanded: !folder.expanded }
          : folder,
      ),
    );
  }

  function handleSelectFile(fileId, folderId) {
    setCurrentFileId(fileId);
    setSelectedFolderId(folderId);
  }

  async function handleUploadFile(event) {
    const [file] = event.target.files || [];
    if (!file) return;

    try {
      const uploaded = await uploadCourseFile(file);
      const targetFolderId = selectedFolderId || CURRENT_FOLDER_ID;

      setFiles((prev) => [{ ...uploaded, folderId: targetFolderId }, ...prev]);
      setCurrentFileId(uploaded.id);
      setSelectedFolderId(targetFolderId);
      setFolders((prev) =>
        prev.map((folder) =>
          folder.id === targetFolderId ? { ...folder, expanded: true } : folder,
        ),
      );
    } catch (error) {
      console.error("上传失败", error);
    } finally {
      event.target.value = "";
    }
  }

  async function handleSubmitQuestion(event) {
    event.preventDefault();
    if (!chatInput.trim() || isAsking) return;

    const question = chatInput.trim();
    setChatInput("");
    setChatMessages((prev) => [...prev, { role: "user", text: question }]);

    if (!currentFile) {
      setChatMessages((prev) => [
        ...prev,
        { role: "assistant", text: "请先上传或选择一份 PDF。" },
      ]);
      return;
    }

    setIsAsking(true);
    try {
      const response = await askFileQuestion({
        question,
        currentFileName: currentFile.name,
      });
      setChatMessages((prev) => [
        ...prev,
        { role: "assistant", text: response.answer },
      ]);
    } catch (error) {
      setChatMessages((prev) => [
        ...prev,
        { role: "assistant", text: "问答出错，请稍后再试。" },
      ]);
    } finally {
      setIsAsking(false);
    }
  }

  return (
    <div className="app-frame">
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
          <div
            className={`mode-switch mode-${viewMode}`}
            role="tablist"
            aria-label="界面模式"
          >
            <span className="mode-switch-thumb" aria-hidden="true" />
            <button
              className={`mode-switch-button ${
                viewMode === "normal" ? "active" : ""
              }`}
              type="button"
              onClick={() => setViewMode("normal")}
            >
              normal
            </button>
            <button
              className={`mode-switch-button ${
                viewMode === "review" ? "active" : ""
              }`}
              type="button"
              onClick={() => setViewMode("review")}
            >
              review
            </button>
          </div>
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
        }`}
      >
        <aside
          className={`sidebar-panel left-panel ${isLeftOpen ? "open" : "closed"}`}
        >
          <div className="sidebar-panel-inner">
            <div className="sidebar-top-spacer" aria-hidden="true" />
            <div className="file-tree-panel sidebar-mode-shell">
              <div
                className={`sidebar-mode-pane ${
                  viewMode === "normal" ? "active" : "inactive"
                }`}
              >
                <div className="summary-panel">
                  <div className="summary-list">
                    {INITIAL_SUMMARIES.map((item, index) => {
                      const isExpanded = expandedSummaryId === item.id;

                      return (
                        <article
                          className={`summary-item ${isExpanded ? "expanded" : ""}`}
                          key={item.id}
                        >
                          <button
                            className="summary-trigger"
                            type="button"
                            onClick={() =>
                              setExpandedSummaryId((currentId) =>
                                currentId === item.id ? "" : item.id,
                              )
                            }
                          >
                            <div className="summary-trigger-top">
                              <span className="summary-index">
                                {String(index + 1).padStart(2, "0")}
                              </span>
                              <span className="summary-chevron">
                                {isExpanded ? "−" : "+"}
                              </span>
                            </div>
                            <div className="summary-copy">{item.summary}</div>
                          </button>

                          <div
                            className={`summary-transcript-wrap ${
                              isExpanded ? "expanded" : ""
                            }`}
                          >
                            <div className="summary-transcript">
                              <div className="summary-transcript-label">
                                转录内容
                              </div>
                              <p>{item.transcript}</p>
                            </div>
                          </div>
                        </article>
                      );
                    })}
                  </div>
                </div>
              </div>

              <div
                className={`sidebar-mode-pane ${
                  viewMode === "review" ? "active" : "inactive"
                }`}
              >
                <div className="folder-tree">
                  {folders.map((folder) => {
                    const folderFiles = files.filter(
                      (file) => file.folderId === folder.id,
                    );

                    return (
                      <article className="folder-item" key={folder.id}>
                        <button
                          className={`folder-row ${
                            selectedFolderId === folder.id ? "active" : ""
                          }`}
                          type="button"
                          onClick={() => handleToggleFolder(folder.id)}
                        >
                          <span className="folder-row-main">
                            <span className="folder-chevron">
                              {folder.expanded ? "▾" : "▸"}
                            </span>
                            <span>{folder.name}</span>
                          </span>
                          <span className="folder-count">{folderFiles.length}</span>
                        </button>

                        {folder.expanded ? (
                          <div className="folder-children">
                            {folderFiles.length ? (
                              folderFiles.map((file) => (
                                <button
                                  className={`tree-file ${
                                    file.id === currentFileId ? "active" : ""
                                  }`}
                                  key={file.id}
                                  type="button"
                                  onClick={() => handleSelectFile(file.id, folder.id)}
                                >
                                  {file.name}
                                </button>
                              ))
                            ) : (
                              <div className="tree-empty">
                                这个文件夹里还没有文件
                              </div>
                            )}
                          </div>
                        ) : null}
                      </article>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        </aside>

        <main className="main-layer">
          <section className="preview-shell">
            <div className="preview-toolbar">
              <div className="preview-toolbar-actions">
                <span className="panel-badge">
                  {currentFile ? currentFile.name : "未选择文件"}
                </span>
                <button
                  className="icon-button"
                  type="button"
                  onClick={() => setIsSettingsOpen(true)}
                  title="打开设置"
                >
                  ⚙
                </button>
                <input
                  ref={fileInputRef}
                  id="file-input"
                  type="file"
                  accept=".pdf"
                  hidden
                  onChange={handleUploadFile}
                />
              </div>
            </div>

            <div
              className={`preview-stage ${currentFile ? "" : "preview-stage-empty"}`.trim()}
              onClick={() => {
                if (!currentFile) {
                  fileInputRef.current?.click();
                }
              }}
            >
              {currentFile ? (
                <PdfPreview file={currentFile} />
              ) : (
                <div className="empty-state">
                  <div className="upload-card">
                    <div className="upload-icon">PDF</div>
                    <div className="upload-title">点击上传 PDF 开始预览</div>
                    <div className="upload-copy">
                      上传后的文件会进入当前选中的文件夹。
                    </div>
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
            <div className="chat-header">
              <h2 className="section-title">课件问答</h2>
            </div>

            <div className="chat-shell">
              <div className="chat-messages" ref={chatMessagesRef}>
                {chatMessages.map((message, index) => (
                  <div
                    className={`message ${message.role}`}
                    key={`${message.role}-${index}`}
                  >
                    {message.text}
                  </div>
                ))}
                {isAsking ? (
                  <div className="message assistant">AI 正在思考中...</div>
                ) : null}
              </div>

              <form className="chat-form" onSubmit={handleSubmitQuestion}>
                <textarea
                  value={chatInput}
                  onChange={(event) => setChatInput(event.target.value)}
                  placeholder="针对这份 PDF 提问..."
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      handleSubmitQuestion(event);
                    }
                  }}
                />
              </form>
            </div>
          </div>
        </aside>
      </div>

      {isSettingsOpen ? (
        <div className="modal-overlay" onClick={() => setIsSettingsOpen(false)}>
          <div className="modal-content" onClick={(event) => event.stopPropagation()}>
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
                  onChange={(event) => setTheme(event.target.value)}
                >
                  <option value="light">浅色</option>
                  <option value="dark">深色</option>
                </select>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default App;
