import type { CourseFile, CourseFileKind } from "../types";

export function detectCourseFileKind(
  fileName: string,
  mimeType = "",
): CourseFileKind {
  const normalizedType = mimeType.toLowerCase();
  const lowerName = fileName.toLowerCase();

  if (normalizedType.includes("pdf") || lowerName.endsWith(".pdf")) {
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

export function isPdfCourseFile(
  file: Pick<CourseFile, "name" | "type" | "kind">,
): boolean {
  if (file.kind) {
    return file.kind === "pdf";
  }

  return detectCourseFileKind(file.name, file.type ?? "") === "pdf";
}
