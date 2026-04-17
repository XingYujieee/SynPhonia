import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import ForceGraph2D from "react-force-graph-2d";
import type {
  KnowledgeBaseActivity,
  KnowledgeBaseContentLine,
  KnowledgeBaseData,
  KnowledgeBaseGraphNode,
} from "../types";

interface KnowledgeGraphPanelProps {
  data: KnowledgeBaseData | null;
  isLoading: boolean;
  onClose: () => void;
  onActivityOpen?: (activityId: string) => void;
}

type KnowledgeTab = "overview" | "graph" | "files";
type FileTab = "transcript" | "summary" | "slides";

interface RenderGraphNode extends KnowledgeBaseGraphNode {
  id: string;
  label: string;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number;
  fy?: number;
}

const GRAPH_WIDTH = 860;
const GRAPH_HEIGHT = 520;
const NODE_RADIUS = 4.5;

function formatDateTime(value?: string | null): string {
  if (!value) {
    return "时间未知";
  }

  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) {
    return "时间未知";
  }

  return date.toLocaleString();
}

function formatDuration(minutes?: number): string {
  if (!minutes || minutes <= 0) {
    return "时长未知";
  }

  if (minutes < 60) {
    return `${minutes} 分钟`;
  }

  const hours = Math.floor(minutes / 60);
  const remain = minutes % 60;
  return remain ? `${hours} 小时 ${remain} 分钟` : `${hours} 小时`;
}

function buildTopicGroups(
  activities: KnowledgeBaseActivity[],
  contentLines: KnowledgeBaseContentLine[],
): Array<{ id: string; title: string; activities: KnowledgeBaseActivity[] }> {
  if (contentLines.length) {
    return contentLines.map((line) => ({
      id: line.content_line_id,
      title: line.title,
      activities: line.activities
        .map((item) =>
          activities.find((activity) => activity.activity_id === item.activity_id),
        )
        .filter((activity): activity is KnowledgeBaseActivity => Boolean(activity)),
    }));
  }

  const groups = new Map<string, KnowledgeBaseActivity[]>();
  for (const activity of activities) {
    const key = activity.keywords_of_keywords[0] || activity.scene_type || "未分类";
    const current = groups.get(key) || [];
    current.push(activity);
    groups.set(key, current);
  }

  return [...groups.entries()].map(([title, groupedActivities], index) => ({
    id: `topic-${index}`,
    title,
    activities: groupedActivities,
  }));
}

function drawNodeLabel(
  node: RenderGraphNode,
  ctx: CanvasRenderingContext2D,
  globalScale: number,
) {
  const label = node.label;
  const fontSize = Math.max(10, 12 / globalScale);
  ctx.font = `500 ${fontSize}px var(--font-ui, sans-serif)`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.lineWidth = 4 / globalScale;
  ctx.strokeStyle = "rgba(255,255,255,0.96)";
  ctx.fillStyle = "#555";
  ctx.strokeText(label, node.x ?? 0, (node.y ?? 0) + NODE_RADIUS + 14 / globalScale);
  ctx.fillText(label, node.x ?? 0, (node.y ?? 0) + NODE_RADIUS + 14 / globalScale);
}

export default function KnowledgeGraphPanel({
  data,
  isLoading,
  onClose,
  onActivityOpen,
}: KnowledgeGraphPanelProps) {
  const [activeTab, setActiveTab] = useState<KnowledgeTab>("overview");
  const [selectedActivityId, setSelectedActivityId] = useState("");
  const [activeFileTab, setActiveFileTab] = useState<FileTab>("transcript");
  const [manualNodePositions, setManualNodePositions] = useState<
    Record<string, { x: number; y: number }>
  >({});
  const graphRef = useRef<any>(undefined);

  const activities = data?.core_data.activities ?? [];
  const graphNodes = data?.graph_view.nodes ?? [];
  const graphEdges = data?.graph_view.edges ?? [];

  useEffect(() => {
    const preferred =
      data?.core_data.selected_activity?.activity_id || activities[0]?.activity_id || "";
    setSelectedActivityId(preferred);
    setActiveFileTab("transcript");
  }, [data, activities]);

  useEffect(() => {
    const nodeIds = new Set(graphNodes.map((node) => node.node_id));
    setManualNodePositions((current) =>
      Object.fromEntries(
        Object.entries(current).filter(([nodeId]) => nodeIds.has(nodeId)),
      ),
    );
  }, [graphNodes]);

  const recentActivities = useMemo(
    () =>
      [...activities].sort(
        (left, right) =>
          Date.parse(right.start_time || "") - Date.parse(left.start_time || ""),
      ),
    [activities],
  );

  const selectedActivity = useMemo(
    () =>
      activities.find((activity) => activity.activity_id === selectedActivityId) ??
      recentActivities[0] ??
      null,
    [activities, recentActivities, selectedActivityId],
  );

  const topicGroups = useMemo(
    () => buildTopicGroups(activities, data?.core_data.content_lines ?? []),
    [activities, data?.core_data.content_lines],
  );

  const graphData = useMemo(
    () => ({
      nodes: graphNodes.map((node) => {
        const manual = manualNodePositions[node.node_id];
        return {
          ...node,
          id: node.node_id,
          label: node.activity_name || node.title,
          x: manual?.x,
          y: manual?.y,
          fx: manual?.x,
          fy: manual?.y,
        } satisfies RenderGraphNode;
      }),
      links: graphEdges.map((edge, index) => ({
        ...edge,
        curvature: ((index % 5) - 2) * 0.08,
      })),
    }),
    [graphNodes, graphEdges, manualNodePositions],
  );

  const selectedNodeEdges = useMemo(
    () =>
      graphEdges.filter(
        (edge) =>
          edge.source_activity_id === selectedActivity?.activity_id ||
          edge.target_activity_id === selectedActivity?.activity_id,
      ),
    [graphEdges, selectedActivity?.activity_id],
  );

  useEffect(() => {
    if (activeTab !== "graph" || !graphData.nodes.length) {
      return;
    }

    const frameId = window.requestAnimationFrame(() => {
      const fg = graphRef.current;
      if (!fg) {
        return;
      }

      const nodeCount = graphData.nodes.length;
      const charge = fg.d3Force("charge");
      if (charge && typeof charge.strength === "function") {
        charge.strength(nodeCount <= 2 ? -1500 : nodeCount <= 4 ? -820 : -420);
      }

      const link = fg.d3Force("link");
      if (link && typeof link.distance === "function") {
        link.distance(nodeCount <= 2 ? 420 : nodeCount <= 4 ? 280 : 180);
      }
      if (link && typeof link.strength === "function") {
        link.strength(0.32);
      }

      fg.d3ReheatSimulation();
      window.setTimeout(() => {
        fg.zoomToFit(250, 80);
      }, 40);
    });

    return () => {
      window.cancelAnimationFrame(frameId);
    };
  }, [activeTab, graphData]);

  function handleSelectActivity(activityId: string): void {
    setSelectedActivityId(activityId);
  }

  return (
    <div className="kb-overlay" onClick={onClose}>
      <div className="kb-panel" onClick={(event) => event.stopPropagation()}>
        <div className="kb-header">
          <div className="kb-header-main">
            <div className="kb-title">知识库</div>
            <div className="kb-header-meta">
              <span className="kb-badge">
                {data?.core_data.counts.activity_count ?? 0} 条活动
              </span>
              <span className="kb-badge">
                {data?.core_data.counts.content_line_count ?? 0} 条内容主线
              </span>
            </div>
          </div>
          <div className="kb-tabs">
            {([
              ["overview", "总览"],
              ["graph", "知识图谱"],
              ["files", "文件预览"],
            ] as const).map(([tab, label]) => (
              <button
                key={tab}
                className={`kb-tab ${activeTab === tab ? "active" : ""}`}
                type="button"
                onClick={() => setActiveTab(tab)}
              >
                {label}
              </button>
            ))}
          </div>
          <button className="icon-button" type="button" onClick={onClose}>
            ✕
          </button>
        </div>

        <div className="kb-body">
          {isLoading ? (
            <div className="kb-state">正在整理知识库数据…</div>
          ) : !activities.length ? (
            <div className="kb-state">
              <strong>暂无可展示的知识库记录</strong>
              <p>先完成至少一次活动转录和总结，这里才会形成总览、图谱和文件预览。</p>
            </div>
          ) : activeTab === "overview" ? (
            <div className="kb-overview">
              <section className="kb-section">
                <div className="kb-section-title">最近记录</div>
                <div className="kb-activity-list">
                  {recentActivities.map((activity) => (
                    <button
                      key={activity.activity_id}
                      className={`kb-activity-card ${
                        selectedActivity?.activity_id === activity.activity_id
                          ? "active"
                          : ""
                      }`}
                      type="button"
                      onClick={() => handleSelectActivity(activity.activity_id)}
                    >
                      <span className="kb-activity-card-title">
                        {activity.activity_name || activity.title}
                      </span>
                      <span className="kb-activity-card-summary">
                        {activity.summary_of_summary}
                      </span>
                      <span className="kb-activity-card-meta">
                        {formatDateTime(activity.start_time)}
                      </span>
                    </button>
                  ))}
                </div>
              </section>

              <section className="kb-section">
                <div className="kb-section-title">主题分组</div>
                <div className="kb-topic-list">
                  {topicGroups.map((group) => (
                    <article className="kb-topic-group" key={group.id}>
                      <div className="kb-topic-title">{group.title}</div>
                      <div className="kb-topic-items">
                        {group.activities.map((activity) => (
                          <button
                            key={activity.activity_id}
                            className="kb-topic-item"
                            type="button"
                            onClick={() => handleSelectActivity(activity.activity_id)}
                          >
                            <span>{activity.activity_name || activity.title}</span>
                            <span>{activity.summary_of_summary}</span>
                          </button>
                        ))}
                      </div>
                    </article>
                  ))}
                </div>
              </section>

              <aside className="kb-detail-card">
                {selectedActivity ? (
                  <>
                    <div className="kb-detail-top">
                      <div>
                        <div className="kb-detail-title">
                          {selectedActivity.activity_name || selectedActivity.title}
                        </div>
                        <div className="kb-detail-meta">
                          {formatDateTime(selectedActivity.start_time)} ·{" "}
                          {formatDuration(selectedActivity.duration_minutes)}
                        </div>
                      </div>
                      {onActivityOpen ? (
                        <button
                          className="secondary-button"
                          type="button"
                          onClick={() => onActivityOpen(selectedActivity.activity_id)}
                        >
                          打开工作区
                        </button>
                      ) : null}
                    </div>

                    <p className="kb-detail-summary">
                      {selectedActivity.activity_intro ||
                        selectedActivity.summary_of_summary}
                    </p>

                    <div className="kb-chip-row">
                      {selectedActivity.keywords_of_keywords.map((keyword) => (
                        <span className="kb-chip primary" key={keyword}>
                          {keyword}
                        </span>
                      ))}
                      {selectedActivity.keywords.map((keyword) => (
                        <span className="kb-chip" key={keyword}>
                          {keyword}
                        </span>
                      ))}
                    </div>

                    <div className="kb-detail-block">
                      <div className="kb-detail-block-title">完整总结</div>
                      <p>{selectedActivity.summary_text}</p>
                    </div>
                  </>
                ) : null}
              </aside>
            </div>
          ) : activeTab === "graph" ? (
            <div className="kb-graph">
              <div className="kb-graph-canvas">
                <ForceGraph2D
                  ref={graphRef}
                  width={GRAPH_WIDTH}
                  height={GRAPH_HEIGHT}
                  graphData={graphData}
                  linkSource="source_activity_id"
                  linkTarget="target_activity_id"
                  backgroundColor="#ffffff"
                  nodeRelSize={selectedActivity ? 8 : 7}
                  nodeVal={(node) =>
                    node.id === selectedActivity?.activity_id ? 10 : 8
                  }
                  nodeColor={(node) =>
                    node.id === selectedActivity?.activity_id ? "#5fa0ff" : "#8ebcff"
                  }
                  linkColor={(link) =>
                    link.source_activity_id === selectedActivity?.activity_id ||
                    link.target_activity_id === selectedActivity?.activity_id
                      ? "#2f6fed"
                      : "#8d8d8d"
                  }
                  linkWidth={(link) =>
                    link.source_activity_id === selectedActivity?.activity_id ||
                    link.target_activity_id === selectedActivity?.activity_id
                      ? 2.2
                      : 1.5
                  }
                  linkCurvature="curvature"
                  enableNodeDrag
                  enableZoomInteraction={false}
                  enablePanInteraction={false}
                  autoPauseRedraw={false}
                  cooldownTicks={120}
                  d3AlphaDecay={0.03}
                  d3VelocityDecay={0.35}
                  nodeLabel={(node) => node.label}
                  onNodeClick={(node) => {
                    handleSelectActivity(String(node.id));
                  }}
                  onNodeDragEnd={(node) => {
                    const nextX = Number(node.x || 0);
                    const nextY = Number(node.y || 0);
                    setManualNodePositions((current) => ({
                      ...current,
                      [String(node.id)]: { x: nextX, y: nextY },
                    }));
                    node.fx = nextX;
                    node.fy = nextY;
                  }}
                  nodeCanvasObjectMode={() => "after"}
                  nodeCanvasObject={(node, ctx, globalScale) => {
                    drawNodeLabel(node as RenderGraphNode, ctx, globalScale);
                  }}
                />
              </div>

              <aside className="kb-graph-detail">
                {selectedActivity ? (
                  <>
                    <div className="kb-detail-title">
                      {selectedActivity.activity_name || selectedActivity.title}
                    </div>
                    <div className="kb-detail-meta">
                      {selectedActivity.scene_type || "activity"} ·{" "}
                      {formatDateTime(selectedActivity.start_time)}
                    </div>
                    <p className="kb-detail-summary">
                      {selectedActivity.summary_of_summary}
                    </p>
                    <div className="kb-chip-row">
                      {selectedActivity.keywords.map((keyword) => (
                        <span className="kb-chip" key={keyword}>
                          {keyword}
                        </span>
                      ))}
                    </div>
                    <div className="kb-relations">
                      <div className="kb-detail-block-title">关联关系</div>
                      {selectedNodeEdges.length ? (
                        selectedNodeEdges.map((edge) => {
                          const otherId =
                            edge.source_activity_id === selectedActivity.activity_id
                              ? edge.target_activity_id
                              : edge.source_activity_id;
                          const other = activities.find(
                            (activity) => activity.activity_id === otherId,
                          );
                          return (
                            <div className="kb-relation-item" key={edge.relation_id}>
                              <span className="kb-relation-name">
                                {other?.activity_name || other?.title || otherId}
                              </span>
                              <span className="kb-relation-strength">
                                {edge.strength}
                              </span>
                            </div>
                          );
                        })
                      ) : (
                        <div className="kb-empty-inline">当前活动还没有可展示的关系边。</div>
                      )}
                    </div>
                  </>
                ) : null}
              </aside>
            </div>
          ) : (
            <div className="kb-files">
              <aside className="kb-files-sidebar">
                <div className="kb-section-title">活动列表</div>
                <div className="kb-activity-list compact">
                  {recentActivities.map((activity) => (
                    <button
                      key={activity.activity_id}
                      className={`kb-activity-card ${
                        selectedActivity?.activity_id === activity.activity_id
                          ? "active"
                          : ""
                      }`}
                      type="button"
                      onClick={() => handleSelectActivity(activity.activity_id)}
                    >
                      <span className="kb-activity-card-title">
                        {activity.activity_name || activity.title}
                      </span>
                      <span className="kb-activity-card-meta">
                        {formatDateTime(activity.start_time)}
                      </span>
                    </button>
                  ))}
                </div>
              </aside>

              <section className="kb-files-preview">
                {selectedActivity ? (
                  <>
                    <div className="kb-files-header">
                      <div>
                        <div className="kb-detail-title">
                          {selectedActivity.activity_name || selectedActivity.title}
                        </div>
                        <div className="kb-detail-meta">
                          {formatDateTime(selectedActivity.start_time)}
                        </div>
                      </div>
                      {onActivityOpen ? (
                        <button
                          className="secondary-button"
                          type="button"
                          onClick={() => onActivityOpen(selectedActivity.activity_id)}
                        >
                          打开工作区
                        </button>
                      ) : null}
                    </div>

                    <div className="kb-file-tabs">
                      {([
                        ["transcript", "转录文稿"],
                        ["summary", "总结文稿"],
                        ["slides", "PPT / 地址"],
                      ] as const).map(([tab, label]) => (
                        <button
                          key={tab}
                          className={`kb-file-tab ${
                            activeFileTab === tab ? "active" : ""
                          }`}
                          type="button"
                          onClick={() => setActiveFileTab(tab)}
                        >
                          {label}
                        </button>
                      ))}
                    </div>

                    <div className="kb-file-content">
                      {activeFileTab === "transcript" ? (
                        <>
                          <div className="kb-file-path">
                            {selectedActivity.transcript_file_path || "未记录转录文件路径"}
                          </div>
                          <pre>{selectedActivity.transcript_text || "暂无转录文本"}</pre>
                        </>
                      ) : activeFileTab === "summary" ? (
                        <>
                          <div className="kb-file-path">
                            {selectedActivity.summary_file_path || "未记录总结文件路径"}
                          </div>
                          <pre>{selectedActivity.summary_text || "暂无总结文本"}</pre>
                        </>
                      ) : (
                        <div className="kb-slide-preview">
                          <div className="kb-file-path">
                            {selectedActivity.ppt_file_path || "未记录课件路径"}
                          </div>
                          <p>
                            {selectedActivity.ppt_text_excerpt ||
                              "当前只展示课件文件地址；后续可以在这里扩展真正的文件预览。"}
                          </p>
                        </div>
                      )}
                    </div>
                  </>
                ) : null}
              </section>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
