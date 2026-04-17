import type { ChatMessage, CourseFile, SummaryCard, WorkspaceFolder } from "./types";
export const THEME_STORAGE_KEY = "course-pdf-theme";

export const INITIAL_FOLDERS: ReadonlyArray<WorkspaceFolder> = [];

export const INITIAL_FILES: ReadonlyArray<CourseFile> = [];

export const INITIAL_MESSAGES: ReadonlyArray<ChatMessage> = [];

export const INITIAL_SUMMARIES: ReadonlyArray<SummaryCard> = [];

export function cloneInitialFolders(): WorkspaceFolder[] {
  return INITIAL_FOLDERS.map((folder) => ({ ...folder }));
}

export function cloneInitialFiles(): CourseFile[] {
  return INITIAL_FILES.map((file) => ({ ...file }));
}

export function cloneInitialMessages(): ChatMessage[] {
  return INITIAL_MESSAGES.map((message) => ({ ...message }));
}

export function cloneInitialSummaries(): SummaryCard[] {
  return INITIAL_SUMMARIES.map((item) => ({ ...item }));
}
