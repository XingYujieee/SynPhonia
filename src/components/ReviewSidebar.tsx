import { useEffect, useState, type MouseEvent as ReactMouseEvent } from "react";
import type { CourseFile, WorkspaceFolder } from "../types";

interface ReviewSidebarProps {
  folders: ReadonlyArray<WorkspaceFolder>;
  files: ReadonlyArray<CourseFile>;
  selectedFolderId: string;
  currentFileId: string;
  onToggleFolder: (folderId: string) => void;
  onSelectFile: (fileId: string, folderId: string) => void;
  onRenameWorkspace: (workspaceId: string) => void;
  onDeleteWorkspace: (workspaceId: string) => void;
  onOpenKnowledgeBase: () => void;
}

function KnowledgeBaseIcon() {
  return (
    <svg
      viewBox="0 0 64 64"
      className="review-kb-icon"
      aria-hidden="true"
      focusable="false"
    >
      <g fill="none" stroke="currentColor" strokeWidth="5" strokeLinecap="round">
        <path d="M32 32 L17 16" />
        <path d="M32 32 L10 31" />
        <path d="M32 32 L47 18" />
        <path d="M32 32 L52 32" />
        <path d="M32 32 L28 51" />
      </g>
      <g fill="currentColor">
        <circle cx="32" cy="32" r="12" />
        <circle cx="15" cy="14" r="7" />
        <circle cx="8" cy="31" r="6" />
        <circle cx="50" cy="18" r="6" />
        <circle cx="54" cy="33" r="6" />
        <circle cx="27" cy="54" r="9" />
      </g>
    </svg>
  );
}

function formatCreatedAt(createdAtUtc?: string): string {
  if (!createdAtUtc) {
    return "创建日期未知";
  }

  const value = new Date(createdAtUtc);
  if (Number.isNaN(value.valueOf())) {
    return "创建日期未知";
  }

  return `创建于 ${value.toLocaleDateString()}`;
}

export default function ReviewSidebar({
  folders,
  files,
  selectedFolderId,
  currentFileId,
  onToggleFolder,
  onSelectFile,
  onRenameWorkspace,
  onDeleteWorkspace,
  onOpenKnowledgeBase,
}: ReviewSidebarProps) {
  const [openMenuFolderId, setOpenMenuFolderId] = useState("");

  useEffect(() => {
    function handlePointerDown(event: PointerEvent): void {
      const target = event.target;
      if (target instanceof Element && target.closest(".folder-menu-shell")) {
        return;
      }

      setOpenMenuFolderId("");
    }

    document.addEventListener("pointerdown", handlePointerDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
    };
  }, []);

  function handleToggleMenu(
    event: ReactMouseEvent<HTMLButtonElement>,
    folderId: string,
  ): void {
    event.stopPropagation();
    setOpenMenuFolderId((currentId) => (currentId === folderId ? "" : folderId));
  }

  function handleRename(
    event: ReactMouseEvent<HTMLButtonElement>,
    folderId: string,
  ): void {
    event.stopPropagation();
    setOpenMenuFolderId("");
    onRenameWorkspace(folderId);
  }

  function handleDelete(
    event: ReactMouseEvent<HTMLButtonElement>,
    folderId: string,
  ): void {
    event.stopPropagation();
    setOpenMenuFolderId("");
    onDeleteWorkspace(folderId);
  }

  return (
    <div className="review-panel">
      <div className="review-header">
        <h2 className="section-title">历史记录</h2>
        <button
          className="icon-button review-kb-icon-button"
          type="button"
          aria-label="打开知识库"
          title="打开知识库"
          onClick={onOpenKnowledgeBase}
        >
          <KnowledgeBaseIcon />
        </button>
      </div>

      {!folders.length ? (
        <div className="tree-empty">还没有可用的工作区</div>
      ) : (
        <div className="folder-tree">
          {folders.map((folder) => {
            const folderFiles = files.filter((file) => file.folderId === folder.id);

            return (
              <article className="folder-item" key={folder.id}>
                <div className={`folder-row ${selectedFolderId === folder.id ? "active" : ""}`}>
                  <button
                    className="folder-row-button"
                    type="button"
                    onClick={() => onToggleFolder(folder.id)}
                  >
                    <span className="folder-row-main">
                      <span className="folder-chevron">
                        {folder.expanded ? "▾" : "▸"}
                      </span>
                      <span className="folder-row-copy">
                        <span className="folder-title">{folder.name}</span>
                        <span className="folder-date">
                          {formatCreatedAt(folder.createdAtUtc)}
                        </span>
                      </span>
                    </span>
                  </button>

                  <div className="folder-row-side">
                    <span className="folder-count">{folderFiles.length}</span>
                    <div className="folder-menu-shell">
                      <button
                        className={`folder-menu-trigger ${
                          openMenuFolderId === folder.id ? "active" : ""
                        }`}
                        type="button"
                        aria-label={`${folder.name} 工作区操作`}
                        aria-expanded={openMenuFolderId === folder.id}
                        onClick={(event) => handleToggleMenu(event, folder.id)}
                      >
                        ⋯
                      </button>

                      {openMenuFolderId === folder.id ? (
                        <div className="folder-menu-popover">
                          <button
                            className="folder-menu-item"
                            type="button"
                            onClick={(event) => handleRename(event, folder.id)}
                          >
                            重命名
                          </button>
                          <button
                            className="folder-menu-item danger"
                            type="button"
                            onClick={(event) => handleDelete(event, folder.id)}
                          >
                            删除
                          </button>
                        </div>
                      ) : null}
                    </div>
                  </div>
                </div>

                {folder.expanded ? (
                  <div className="folder-children">
                    {folderFiles.length ? (
                      folderFiles.map((file) => (
                        <button
                          className={`tree-file ${file.id === currentFileId ? "active" : ""}`}
                          key={file.id}
                          type="button"
                          onClick={() => onSelectFile(file.id, folder.id)}
                        >
                          {file.name}
                        </button>
                      ))
                    ) : (
                      <div className="tree-empty">这个文件夹里还没有文件</div>
                    )}
                  </div>
                ) : null}
              </article>
            );
          })}
        </div>
      )}
    </div>
   );
}
