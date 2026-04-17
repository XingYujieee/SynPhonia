import { useState } from "react";
import type {
  SummaryCard,
  SummaryEmptyState,
} from "../types";

type SidebarTab = "summary" | "transcript";

interface SummarySidebarProps {
  summaries: ReadonlyArray<SummaryCard>;
  expandedSummaryId: string;
  emptyState?: SummaryEmptyState;
  onToggleSummary: (summaryId: string) => void;
  realtimePartialText?: string;
  fullTranscriptText?: string;
}

export default function SummarySidebar({
  summaries,
  expandedSummaryId,
  emptyState,
  onToggleSummary,
  realtimePartialText,
  fullTranscriptText,
}: SummarySidebarProps) {
  const [activeTab, setActiveTab] = useState<SidebarTab>("summary");

  const hasTranscript =
    !!fullTranscriptText?.trim() || !!realtimePartialText?.trim();

  return (
    <div className="summary-panel">
      {/* ── 顶部 Tab 栏 ── */}
      <div className="summary-tabs">
        <button
          className={`summary-tab ${activeTab === "summary" ? "active" : ""}`}
          type="button"
          onClick={() => setActiveTab("summary")}
        >
          总结
          {summaries.length > 0 && (
            <span className="summary-tab-badge">{summaries.length}</span>
          )}
        </button>
        <button
          className={`summary-tab ${activeTab === "transcript" ? "active" : ""}`}
          type="button"
          onClick={() => setActiveTab("transcript")}
        >
          转录全文
          {realtimePartialText && <span className="realtime-tab-dot" />}
        </button>
      </div>

      {/* ── 总结 Tab ── */}
      {activeTab === "summary" && (
        <>
          {realtimePartialText ? (
            <div className="realtime-transcript-indicator">
              <span className="realtime-indicator-dot" />
              <span className="realtime-indicator-text">{realtimePartialText}</span>
            </div>
          ) : null}

          {!summaries.length ? (
            <div className="summary-empty">
              <div className="summary-empty-title">
                {fullTranscriptText
                  ? "转录中，等待生成摘要…"
                  : emptyState?.title || "等待转录与总结结果"}
              </div>
              {fullTranscriptText ? (
                <div className="summary-empty-transcript">
                  <div className="summary-empty-transcript-label">
                    已转录内容（累积满 200 字后自动生成摘要）
                  </div>
                  <p className="summary-realtime-accumulate">{fullTranscriptText}</p>
                </div>
              ) : (
                <div className="summary-empty-copy">
                  {emptyState?.copy ||
                    "上传文件后，后端生成的分段总结会显示在这里。"}
                </div>
              )}
              {emptyState?.transcriptPreview && !fullTranscriptText ? (
                <div className="summary-empty-transcript">
                  <div className="summary-empty-transcript-label">
                    本次转录结果
                  </div>
                  <p>{emptyState.transcriptPreview}</p>
                </div>
              ) : null}
            </div>
          ) : (
            <div className="summary-list">
              {summaries.map((item, index) => {
                const isExpanded = expandedSummaryId === item.id;

                return (
                  <article
                    className={`summary-item ${isExpanded ? "expanded" : ""}`}
                    key={item.id}
                  >
                    <button
                      className="summary-trigger"
                      type="button"
                      onClick={() => onToggleSummary(item.id)}
                    >
                      <div className="summary-trigger-top">
                        <span className="summary-index">
                          {String(index + 1).padStart(2, "0")}
                        </span>
                        <span className="summary-chevron">
                          {isExpanded ? "−" : "+"}
                        </span>
                      </div>
                    </button>
                    <div className="summary-copy">{item.summary}</div>

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
          )}
        </>
      )}

      {/* ── 转录全文 Tab ── */}
      {activeTab === "transcript" && (
        <div className="transcript-full-panel">
          {realtimePartialText && (
            <div className="realtime-transcript-indicator">
              <span className="realtime-indicator-dot" />
              <span className="realtime-indicator-text">
                {realtimePartialText}
              </span>
            </div>
          )}

          {hasTranscript ? (
            <div className="transcript-full-body">
              <pre className="transcript-full-text">
                {fullTranscriptText || ""}
              </pre>
            </div>
          ) : (
            <div className="summary-empty">
              <div className="summary-empty-title">暂无转录内容</div>
              <div className="summary-empty-copy">
                开始监听后，完整转录文本会显示在这里，支持全文选中复制。
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
