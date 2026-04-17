import { useEffect, useRef, useState } from "react";
import { GlobalWorkerOptions, TextLayer, getDocument } from "pdfjs-dist";
import type {
  PDFDocumentProxy,
  RenderTask,
} from "pdfjs-dist/types/src/display/api";
import workerSource from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import type { CourseFile } from "../types";

const DEFAULT_SCALE = 1.1;
const MIN_SCALE = 0.8;
const MAX_SCALE = 2;
const SCALE_STEP = 0.15;

GlobalWorkerOptions.workerSrc = workerSource;

interface PdfPreviewProps {
  file: CourseFile;
  /** 外部指定跳转到某页（1-indexed）。页码匹配模块推进页时由父组件传入。 */
  targetPage?: number;
  onPageChange?: (page: number) => void;
}

function formatScale(scale: number): string {
  return `${Math.round(scale * 100)}%`;
}

function hasErrorName(error: unknown, name: string): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "name" in error &&
    (error as { name?: string }).name === name
  );
}

function getErrorMessage(error: unknown): string {
  if (hasErrorName(error, "InvalidPDFException")) {
    return "这个文件不是有效的 PDF。";
  }

  if (hasErrorName(error, "MissingPDFException")) {
    return "没有找到可加载的 PDF 文件。";
  }

  return "PDF 加载失败，请检查文件内容或预览地址。";
}

function PreviousPageIcon() {
  return (
    <svg viewBox="0 0 1024 1024" className="viewer-icon" aria-hidden="true">
      <path
        d="M753.152 138.752l-72.704-72.704L235.008 512l445.44 445.952 72.704-72.704L379.392 512l373.76-373.248z"
        fill="currentColor"
      />
    </svg>
  );
}

function NextPageIcon() {
  return (
    <svg viewBox="0 0 1024 1024" className="viewer-icon" aria-hidden="true">
      <path
        d="M343.552 66.048L270.848 138.752l373.76 373.248-373.76 373.248 72.704 72.704 445.44-445.952-445.44-445.952z"
        fill="currentColor"
      />
    </svg>
  );
}

export default function PdfPreview({ file, targetPage, onPageChange }: PdfPreviewProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const pageLayerRef = useRef<HTMLDivElement | null>(null);
  const textLayerContainerRef = useRef<HTMLDivElement | null>(null);
  const renderTaskRef = useRef<RenderTask | null>(null);
  const textLayerRef = useRef<TextLayer | null>(null);

  const [pdfDocument, setPdfDocument] = useState<PDFDocumentProxy | null>(null);
  const [pageCount, setPageCount] = useState(0);
  const [pageNumber, setPageNumber] = useState(1);
  const [scale, setScale] = useState(DEFAULT_SCALE);
  const [isAutoPagingEnabled, setIsAutoPagingEnabled] = useState(false);
  const [isLoadingDocument, setIsLoadingDocument] = useState(false);
  const [isRenderingPage, setIsRenderingPage] = useState(false);
  const [pageMotionDirection, setPageMotionDirection] = useState<
    "forward" | "backward" | "idle"
  >("idle");
  const [errorMessage, setErrorMessage] = useState("");
  // 标记是否来自外部跳页（用于区分"自动"和"手动"）
  const lastTargetPageRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    setPdfDocument(null);
    setPageCount(0);
    setPageNumber(1);
    setScale(DEFAULT_SCALE);
    setIsAutoPagingEnabled(false);
    setPageMotionDirection("idle");
    setErrorMessage("");
    renderTaskRef.current?.cancel();
    textLayerRef.current?.cancel();
    lastTargetPageRef.current = undefined;

    if (!file.previewUrl) {
      setIsLoadingDocument(false);
      return;
    }

    let isCancelled = false;
    const loadingTask = getDocument({
      url: file.previewUrl,
      isEvalSupported: false,
    });

    setIsLoadingDocument(true);

    loadingTask.promise
      .then((nextDocument) => {
        if (isCancelled) {
          void nextDocument.destroy();
          return;
        }

        setPdfDocument(nextDocument);
        setPageCount(nextDocument.numPages);
        setPageNumber(1);
      })
      .catch((error: unknown) => {
        if (!isCancelled) {
          setErrorMessage(getErrorMessage(error));
        }
      })
      .finally(() => {
        if (!isCancelled) {
          setIsLoadingDocument(false);
        }
      });

    return () => {
      isCancelled = true;
      renderTaskRef.current?.cancel();
      textLayerRef.current?.cancel();
      void loadingTask.destroy();
    };
  }, [file.id, file.previewUrl]);

  useEffect(() => {
    if (!pdfDocument) {
      return;
    }

    return () => {
      void pdfDocument.destroy();
    };
  }, [pdfDocument]);

  // 响应外部 targetPage 变化，自动跳到指定页（页码匹配联动）
  useEffect(() => {
    if (
      !isAutoPagingEnabled ||
      targetPage == null ||
      !pdfDocument ||
      pageCount === 0 ||
      targetPage === lastTargetPageRef.current
    ) {
      return;
    }

    const clamped = Math.min(pageCount, Math.max(1, targetPage));
    if (clamped !== pageNumber) {
      lastTargetPageRef.current = targetPage;
      setPageMotionDirection(clamped > pageNumber ? "forward" : "backward");
      setPageNumber(clamped);
      onPageChange?.(clamped);
    }
  }, [isAutoPagingEnabled, targetPage, pdfDocument, pageCount, pageNumber, onPageChange]);

  useEffect(() => {
    if (!pdfDocument || !canvasRef.current || !textLayerContainerRef.current || !pageLayerRef.current) {
      return;
    }

    let isCancelled = false;

    async function renderPage(): Promise<void> {
      setIsRenderingPage(true);
      setErrorMessage("");

      try {
        const activeDocument = pdfDocument;
        if (!activeDocument) {
          return;
        }

        const page = await activeDocument.getPage(pageNumber);
        const viewport = page.getViewport({ scale });
        const canvas = canvasRef.current;
        const pageLayer = pageLayerRef.current;
        const textLayerContainer = textLayerContainerRef.current;

        if (!canvas || !pageLayer || !textLayerContainer) {
          return;
        }

        const context = canvas.getContext("2d", { alpha: false });

        if (!context) {
          setErrorMessage("当前浏览器无法初始化 PDF 画布。");
          return;
        }

        const outputScale = window.devicePixelRatio || 1;
        canvas.width = Math.floor(viewport.width * outputScale);
        canvas.height = Math.floor(viewport.height * outputScale);
        canvas.style.width = `${Math.floor(viewport.width)}px`;
        canvas.style.height = `${Math.floor(viewport.height)}px`;
        pageLayer.style.width = `${Math.floor(viewport.width)}px`;
        pageLayer.style.height = `${Math.floor(viewport.height)}px`;
        context.clearRect(0, 0, canvas.width, canvas.height);
        textLayerContainer.replaceChildren();

        renderTaskRef.current?.cancel();
        textLayerRef.current?.cancel();

        const renderTask = page.render({
          canvasContext: context,
          viewport,
          transform:
            outputScale === 1
              ? undefined
              : [outputScale, 0, 0, outputScale, 0, 0],
        });

        renderTaskRef.current = renderTask;
        await renderTask.promise;

        const textContent = await page.getTextContent();
        const textLayer = new TextLayer({
          textContentSource: textContent,
          container: textLayerContainer,
          viewport,
        });
        textLayerRef.current = textLayer;
        await textLayer.render();

        if (!isCancelled) {
          setIsRenderingPage(false);
          window.requestAnimationFrame(() => {
            setPageMotionDirection("idle");
          });
        }
      } catch (error: unknown) {
        if (isCancelled || hasErrorName(error, "RenderingCancelledException")) {
          return;
        }

        setIsRenderingPage(false);
        setErrorMessage("PDF 页面渲染失败。");
      }
    }

    void renderPage();

    return () => {
      isCancelled = true;
      renderTaskRef.current?.cancel();
      textLayerRef.current?.cancel();
    };
  }, [pageNumber, pdfDocument, scale]);

  function handlePreviousPage(): void {
    setPageMotionDirection("backward");
    setPageNumber((currentPage) => Math.max(1, currentPage - 1));
  }

  function handleNextPage(): void {
    setPageMotionDirection("forward");
    setPageNumber((currentPage) => Math.min(pageCount, currentPage + 1));
  }

  function handleZoomOut(): void {
    setScale((currentScale) =>
      Math.max(MIN_SCALE, Number((currentScale - SCALE_STEP).toFixed(2))),
    );
  }

  function handleZoomIn(): void {
    setScale((currentScale) =>
      Math.min(MAX_SCALE, Number((currentScale + SCALE_STEP).toFixed(2))),
    );
  }

  return (
    <div className="pdf-viewer">
      <div className="pdf-viewer-toolbar">
        <div className="pdf-viewer-status">
          {pageCount ? `第 ${pageNumber} / ${pageCount} 页` : "正在准备页面"}
          {targetPage != null && targetPage > 0 && (
            <span className="pdf-page-match-badge" title="由页码匹配自动定位">
              ⚡ 第 {targetPage} 页
            </span>
          )}
        </div>
        <div className="pdf-viewer-controls">
          <button
            className={`viewer-button viewer-auto-button ${
              isAutoPagingEnabled ? "active" : ""
            }`}
            type="button"
            onClick={() => setIsAutoPagingEnabled((current) => !current)}
            aria-pressed={isAutoPagingEnabled}
            title={isAutoPagingEnabled ? "关闭自动翻页" : "开启自动翻页"}
          >
            自动翻页
          </button>
          <button
            className="viewer-button"
            type="button"
            onClick={handlePreviousPage}
            disabled={
              isAutoPagingEnabled || !pdfDocument || pageNumber <= 1
            }
            aria-label="上一页"
            title="上一页"
          >
            <PreviousPageIcon />
          </button>
          <button
            className="viewer-button"
            type="button"
            onClick={handleNextPage}
            disabled={
              isAutoPagingEnabled || !pdfDocument || pageNumber >= pageCount
            }
            aria-label="下一页"
            title="下一页"
          >
            <NextPageIcon />
          </button>
          <button
            className="viewer-button"
            type="button"
            onClick={handleZoomOut}
            disabled={isAutoPagingEnabled || scale <= MIN_SCALE}
            aria-label="缩小"
            title="缩小"
          >
            −
          </button>
          <span
            className={`viewer-scale ${
              isAutoPagingEnabled ? "is-disabled" : ""
            }`}
          >
            {formatScale(scale)}
          </span>
          <button
            className="viewer-button"
            type="button"
            onClick={handleZoomIn}
            disabled={isAutoPagingEnabled || scale >= MAX_SCALE}
            aria-label="放大"
            title="放大"
          >
            +
          </button>
        </div>
      </div>

      <div className="pdf-canvas-wrap">
        {errorMessage ? (
          <div className="pdf-state">
            <strong>预览不可用</strong>
            <p>{errorMessage}</p>
          </div>
        ) : isLoadingDocument ? (
          <div className="pdf-state">
            <strong>正在加载 PDF</strong>
            <p>文件已经导入，正在用 PDF.js 生成预览。</p>
          </div>
        ) : (
          <>
            {isRenderingPage ? (
              <div className="pdf-rendering-badge">渲染中</div>
            ) : null}
            <div
              ref={pageLayerRef}
              className={`pdf-page-layer ${
                pageMotionDirection !== "idle" ? `is-${pageMotionDirection}` : ""
              } ${isRenderingPage ? "is-transitioning" : ""}`}
            >
              <canvas ref={canvasRef} className="pdf-canvas" />
              <div ref={textLayerContainerRef} className="pdf-text-layer" />
            </div>
          </>
        )}
      </div>
    </div>
  );
}
