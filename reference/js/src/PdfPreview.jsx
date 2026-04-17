import { useEffect, useRef, useState } from "react";
import { GlobalWorkerOptions, getDocument } from "pdfjs-dist/build/pdf.mjs";
import workerSource from "pdfjs-dist/build/pdf.worker.min.mjs?url";

const DEFAULT_SCALE = 1.1;
const MIN_SCALE = 0.8;
const MAX_SCALE = 2;
const SCALE_STEP = 0.15;

GlobalWorkerOptions.workerSrc = workerSource;

function formatScale(scale) {
  return `${Math.round(scale * 100)}%`;
}

function getErrorMessage(error) {
  if (error?.name === "InvalidPDFException") {
    return "这个文件不是有效的 PDF。";
  }

  if (error?.name === "MissingPDFException") {
    return "没有找到可加载的 PDF 文件。";
  }

  return "PDF 加载失败，请检查文件内容或预览地址。";
}

export default function PdfPreview({ file }) {
  const canvasRef = useRef(null);
  const renderTaskRef = useRef(null);
  const [pdfDocument, setPdfDocument] = useState(null);
  const [pageCount, setPageCount] = useState(0);
  const [pageNumber, setPageNumber] = useState(1);
  const [scale, setScale] = useState(DEFAULT_SCALE);
  const [isLoadingDocument, setIsLoadingDocument] = useState(false);
  const [isRenderingPage, setIsRenderingPage] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    setPdfDocument(null);
    setPageCount(0);
    setPageNumber(1);
    setScale(DEFAULT_SCALE);
    setErrorMessage("");
    renderTaskRef.current?.cancel();

    if (!file?.previewUrl) {
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
          nextDocument.destroy().catch(() => {});
          return;
        }

        setPdfDocument(nextDocument);
        setPageCount(nextDocument.numPages);
        setPageNumber(1);
      })
      .catch((error) => {
        if (isCancelled) {
          return;
        }

        setErrorMessage(getErrorMessage(error));
      })
      .finally(() => {
        if (!isCancelled) {
          setIsLoadingDocument(false);
        }
      });

    return () => {
      isCancelled = true;
      renderTaskRef.current?.cancel();
      loadingTask.destroy();
    };
  }, [file?.id, file?.previewUrl]);

  useEffect(() => {
    if (!pdfDocument) {
      return;
    }

    return () => {
      pdfDocument.destroy().catch(() => {});
    };
  }, [pdfDocument]);

  useEffect(() => {
    if (!pdfDocument || !canvasRef.current) {
      return;
    }

    let isCancelled = false;

    async function renderPage() {
      setIsRenderingPage(true);
      setErrorMessage("");

      try {
        const page = await pdfDocument.getPage(pageNumber);
        const viewport = page.getViewport({ scale });
        const canvas = canvasRef.current;

        if (!canvas) {
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
        context.clearRect(0, 0, canvas.width, canvas.height);

        renderTaskRef.current?.cancel();
        const renderTask = page.render({
          canvasContext: context,
          viewport,
          transform: outputScale === 1 ? null : [outputScale, 0, 0, outputScale, 0, 0],
        });

        renderTaskRef.current = renderTask;
        await renderTask.promise;

        if (!isCancelled) {
          setIsRenderingPage(false);
        }
      } catch (error) {
        if (isCancelled || error?.name === "RenderingCancelledException") {
          return;
        }

        setIsRenderingPage(false);
        setErrorMessage("PDF 页面渲染失败。");
      }
    }

    renderPage();

    return () => {
      isCancelled = true;
      renderTaskRef.current?.cancel();
    };
  }, [pageNumber, pdfDocument, scale]);

  function handlePreviousPage() {
    setPageNumber((currentPage) => Math.max(1, currentPage - 1));
  }

  function handleNextPage() {
    setPageNumber((currentPage) => Math.min(pageCount, currentPage + 1));
  }

  function handleZoomOut() {
    setScale((currentScale) =>
      Math.max(MIN_SCALE, Number((currentScale - SCALE_STEP).toFixed(2))),
    );
  }

  function handleZoomIn() {
    setScale((currentScale) =>
      Math.min(MAX_SCALE, Number((currentScale + SCALE_STEP).toFixed(2))),
    );
  }

  return (
    <div className="pdf-viewer">
      <div className="pdf-viewer-toolbar">
        <div className="pdf-viewer-status">
          {pageCount ? `第 ${pageNumber} / ${pageCount} 页` : "正在准备页面"}
        </div>
        <div className="pdf-viewer-controls">
          <button
            className="viewer-button"
            type="button"
            onClick={handlePreviousPage}
            disabled={!pdfDocument || pageNumber <= 1}
          >
            上一页
          </button>
          <button
            className="viewer-button"
            type="button"
            onClick={handleNextPage}
            disabled={!pdfDocument || pageNumber >= pageCount}
          >
            下一页
          </button>
          <button
            className="viewer-button"
            type="button"
            onClick={handleZoomOut}
            disabled={scale <= MIN_SCALE}
          >
            缩小
          </button>
          <span className="viewer-scale">{formatScale(scale)}</span>
          <button
            className="viewer-button"
            type="button"
            onClick={handleZoomIn}
            disabled={scale >= MAX_SCALE}
          >
            放大
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
            {isRenderingPage ? <div className="pdf-rendering-badge">渲染中</div> : null}
            <canvas ref={canvasRef} className="pdf-canvas" />
          </>
        )}
      </div>
    </div>
  );
}
