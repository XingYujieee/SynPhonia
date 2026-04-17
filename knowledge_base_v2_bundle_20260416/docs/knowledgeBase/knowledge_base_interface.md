# 知识库模块当前接口说明

## 1. 文档目的

本文档描述的是当前已经落地到代码中的知识库接口，而不是仅停留在讨论阶段的目标草案。

如果需要看“为什么这样设计”和“下一阶段的原则边界”，请同时参考：

- `knowledge_base_v2_contract.md`
- `knowledge_base_boundary.md`

## 2. 当前模块定位

当前知识库模块已经是一个独立板块，职责是：

- 接收一次完整活动的结构化结果
- 校验输入字段
- 保存活动级元数据
- 暴露活动数据、图谱数据、搜索能力和兼容视图输出

它不负责：

- 生成转录文本
- 生成总结文本
- 管理前端页面布局
- 直接从旧模块内部拉取运行中状态

## 3. 当前 ingestion 输入契约

### 3.1 当前必填字段

当前代码要求以下字段为必填：

- `activity_id`
- `start_time`
- `end_time`
- `transcript_text`
- `summary_text`
- `summary_of_summary`
- `keywords`
- `keywords_of_keywords`
- `ppt_present`
- `activity_intro`
- `activity_name`

补充规则：

- 当 `ppt_present=true` 时，必须提供 `ppt_file_path` 或 `ppt_id`
- `keywords` 和 `keywords_of_keywords` 不能为空集合
- 空字符串、空白字符串、空列表，都会按缺失处理

### 3.2 当前可选字段

当前代码支持但不强制要求以下字段：

- `activity_dir`
- `transcript_file_path`
- `summary_file_path`
- `transcript_meta`
- `summary_meta`
- `matched_slides`
- `ppt_text_excerpt`
- `scene_type`

### 3.3 当前异常返回方式

当输入缺失关键字段或字段格式非法时，`ingest_completed_activity(...)` 当前不会静默成功，也不会直接把整个服务打崩。

它会返回形如：

```json
{
  "activity_id": "some-id",
  "status": "invalid_input",
  "missing_fields": ["summary_of_summary", "activity_name"],
  "invalid_fields": {
    "end_time": "Field 'end_time' must not be earlier than 'start_time'."
  }
}
```

也就是说：

- `status=invalid_input` 表示本次入库被拒绝
- `missing_fields` 表示缺失的必填字段
- `invalid_fields` 表示格式或逻辑错误字段

## 4. 当前存储行为

当前知识库会在自己的 workspace 下维护：

- 每个活动一份 `record.json`
- 一份 `relation_overrides.json`

当前实现还会保留完整文本内容：

- `transcript_text`
- `summary_text`
- `summary_of_summary`

同时也会保留文件引用路径：

- `activity_dir`
- `transcript_file_path`
- `summary_file_path`
- `ppt_file_path`

当前策略是：

- 如果上游已经提供 `transcript_file_path` / `summary_file_path`，知识库直接保存这些路径引用
- 如果没有提供，知识库会回退到在自身 workspace 中生成 `transcript.txt` / `summary.txt`

这意味着当前实现既支持更贴近真实应用的“外部活动目录”模式，也兼容旧的“知识库内部回退落盘”模式。

## 5. 当前输出接口

### 5.1 核心数据导出

当前主接口之一是：

- `export_core_data(selected_activity_id=None)`

它返回：

- `activities`
- `selected_activity`
- `content_lines`
- `counts`

其中每个活动项当前会包含：

- 基础标识和时间信息
- `activity_name`
- `activity_intro`
- `transcript_text`
- `summary_text`
- `summary_of_summary`
- `keywords`
- `keywords_of_keywords`
- 文件引用路径
- 关联信息
- 所属内容主线信息

### 5.2 图谱数据导出

当前图谱接口是：

- `export_graph_view()`

它返回：

- `nodes`
- `edges`

节点中会包含活动名称、简介、简要总结、关键词、场景类型等信息。

边中会包含：

- `relation_id`
- `source_activity_id`
- `target_activity_id`
- `strength`
- `state`
- `reasons`
- `source_type`

### 5.3 兼容视图导出

为了兼容旧的无前端验证方式，当前仍保留：

- `export_view_bundle(selected_activity_id=None)`

它输出的仍是偏页面组织的数据块：

- `navigation`
- `history`
- `relation_map`
- `timeline_calendar`
- `timeline_line_view`
- `file_lookup`
- `detail_panel`

当前它属于兼容层，而不是未来唯一主接口。

### 5.4 组合导出

当前还提供：

- `export_all_views(selected_activity_id=None)`

它会一次性返回：

- `core_data`
- `graph_view`
- `legacy_view_bundle`

这个接口当前主要用于 CLI demo 和无前端联调检查。

## 6. 当前搜索接口

当前仍保留搜索能力，但搜索结果不属于默认导出的基础数据。

当前搜索接口：

- `search(query)`
- `search_current_page(text, query)`

当前搜索范围包括：

- `activity_id`
- `activity_name`
- `activity_intro`
- `summary_of_summary`
- `summary_text`
- `transcript_text`
- `keywords`
- `keywords_of_keywords`
- `activity_dir`
- `transcript_file_path`
- `summary_file_path`
- `ppt_file_path`
- 时间字段

也就是说：

- 前端负责“什么时候搜索、怎么展示搜索结果”
- 知识库负责“提供搜索能力和搜索所需底层数据”

## 7. 当前 CLI 入口

当前主要 CLI 命令包括：

- `python -m knowledge_base.cli ingest-file`
- `python -m knowledge_base.cli export-data`
- `python -m knowledge_base.cli export-views`
- `python -m knowledge_base.cli search`
- `python -m knowledge_base.cli set-relation`
- `python -m knowledge_base.cli demo`

说明：

- `export-data` 导出当前主接口数据，即 `core_data + graph_view`
- `export-views` 导出旧的兼容视图 bundle
- `demo` 会先 ingest，再导出 `core_data + graph_view + legacy_view_bundle`

## 8. 当前消费者

当前知识库接口面向的消费者包括：

- 后续应用前端
- 上层活动协调器
- CLI 验证工具
- 自动化测试

推荐原则：

- 消费者优先读取 service 暴露的接口输出
- 不直接依赖 workspace 内部文件结构
- 若只是为了文件预览，再使用记录中的文件路径字段
