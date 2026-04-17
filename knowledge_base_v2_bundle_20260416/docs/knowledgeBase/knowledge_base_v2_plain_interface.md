# 知识库 V2 通俗接口说明

## 1. 这份文档是给谁看的

这份文档面向后续要接手本知识库模块的开发者。

它不强调内部实现细节，而是用更通俗的话说明两件事：

- 知识库 V2 需要接收什么数据
- 知识库 V2 会对外提供什么数据

如果你只关心“我应该给知识库什么”和“我能从知识库拿到什么”，优先看这份文档即可。

如果你还想看更正式的边界和细节，可以再参考：

- `docs/knowledgeBase/knowledge_base_interface.md`
- `docs/knowledgeBase/knowledge_base_boundary.md`
- `docs/knowledgeBase/knowledge_base_v2_contract.md`

## 2. 先说清楚：知识库 V2 是做什么的

知识库 V2 的职责很简单：

- 接收“一次完整活动”的结果
- 校验这些结果是否齐全、格式是否正确
- 把活动保存下来
- 对外提供后续可用的数据

它不是：

- 转录模块
- 总结模块
- 前端页面
- 前端状态管理器

也就是说，知识库不负责生成转录和总结，它只负责接收、保存、整理、提供数据。

## 3. 什么时候可以把一条活动数据交给知识库

只有当一次活动已经结束，并且上层模块确认“这是一条完整活动记录”时，才应该调用知识库入库。

换句话说，知识库 V2 不处理“进行中的活动片段”，而是处理“完整活动结果”。

典型场景是：

1. 用户开始一次活动
2. 上游模块完成转录
3. 上游模块完成总结
4. 用户确认本次活动结束
5. 上层协调器把完整活动记录交给知识库

## 4. 知识库 V2 需要接收的输入数据

知识库 V2 的输入单位是“一条活动记录”。

可以理解为：每完成一次活动，就向知识库提交一组字段。

### 4.1 必填字段

以下字段是当前 V2 默认要求上游提供的正式字段：

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

下面用通俗的话解释每个字段：

- `activity_id`
  - 这条活动记录的唯一标识。
  - 例如：`activity-0001`

- `start_time`
  - 本次活动开始时间。
  - 要用 ISO 时间字符串。
  - 例如：`2026-04-16T09:30:00`

- `end_time`
  - 本次活动结束时间。
  - 也要用 ISO 时间字符串。

- `transcript_text`
  - 本次活动的完整转录文本。
  - 这是给知识库做预览、搜索、记录使用的全文本。

- `summary_text`
  - 本次活动的完整总结文本。
  - 这是完整版本，不是短摘要。

- `summary_of_summary`
  - 对完整总结再做一次更短的概括。
  - 可以理解为“活动的简短摘要”。

- `keywords`
  - 本次活动的关键词列表。
  - 例如：`["牛顿第二定律", "受力分析", "加速度"]`

- `keywords_of_keywords`
  - 比 `keywords` 更上层、更抽象的一组关键词。
  - 可以理解为“主题级关键词”。
  - 例如：`["经典力学", "课堂讲解"]`

- `ppt_present`
  - 是否存在 PPT 或课件文件。
  - 类型是布尔值：`true` 或 `false`

- `activity_intro`
  - 活动简介。
  - 当前约定里，默认可以直接取 `summary_of_summary` 的值。
  - 后续允许用户编辑。

- `activity_name`
  - 活动名称。
  - 当前约定里，默认可以按活动顺序命名，比如“活动1”“活动2”。
  - 后续允许用户编辑。

### 4.2 可选字段

以下字段不是强制要求，但知识库 V2 支持接收：

- `activity_dir`
- `transcript_file_path`
- `summary_file_path`
- `transcript_meta`
- `summary_meta`
- `matched_slides`
- `ppt_text_excerpt`
- `scene_type`
- `ppt_file_path`
- `ppt_id`

这些字段的通俗解释如下：

- `activity_dir`
  - 这次活动在本机上的活动目录路径。
  - 如果上层模块已经有统一的活动文件夹，可以把路径传给知识库。

- `transcript_file_path`
  - 转录文档在本机上的保存路径。

- `summary_file_path`
  - 总结文档在本机上的保存路径。

- `transcript_meta`
  - 转录过程的一些附加信息。
  - 例如片段数、平均置信度、语言、录音时长。
  - 这类信息更偏诊断和调试，不是核心业务字段。

- `summary_meta`
  - 总结过程的一些附加信息。
  - 例如调用了哪个模型、摘要轮数、提供方是谁。
  - 同样更偏诊断和调试。

- `matched_slides`
  - 如果做过 PPT 匹配，这里可以放匹配到的页码和时间段。

- `ppt_text_excerpt`
  - PPT 中抽取出来的预览文本。

- `scene_type`
  - 场景类型。
  - 例如“课堂”“会议”“讨论”等。

- `ppt_file_path`
  - PPT 文件在本机上的路径。

- `ppt_id`
  - PPT 的资源 ID。
  - 这是为了以后可能的资源管理系统预留的字段。

### 4.3 关于默认值的当前约定

当前已经确认的默认策略是：

- `activity_name`
  - 默认和活动顺序相关，例如“活动1”“活动2”
  - 如果用户后续改名，可以覆盖这个默认值

- `activity_intro`
  - 默认取 `summary_of_summary`
  - 如果用户后续修改简介，可以覆盖这个默认值

需要注意：

- 这是“上层协调器或入库适配层”更适合做的默认值策略
- 知识库可以接收这些值，但不建议把正式业务默认规则藏在知识库内部偷偷生成

## 5. 知识库 V2 的输入校验规则

知识库不会无条件接受任何输入。

当前最重要的校验规则有：

- 必填字段不能为空
- `keywords` 不能为空列表
- `keywords_of_keywords` 不能为空列表
- `start_time` 和 `end_time` 必须是合法时间字符串
- `end_time` 不能早于 `start_time`
- 如果 `ppt_present=true`，就必须提供 `ppt_file_path` 或 `ppt_id`

如果输入不合法，知识库不会假装成功入库。

它会返回一个明确的失败结果，例如：

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

这表示：

- `status=invalid_input`
  - 这次提交没有成功入库

- `missing_fields`
  - 缺了哪些必填字段

- `invalid_fields`
  - 哪些字段格式错了，或者逻辑不对

## 6. 知识库 V2 入库后会保存什么

对于每次成功入库的活动，知识库会保存两类信息：

- 结构化字段
- 文本与文件引用

### 6.1 结构化字段

知识库会保存这次活动的核心字段，例如：

- 活动 ID
- 开始和结束时间
- 活动名称
- 活动简介
- 全量转录
- 全量总结
- 简要总结
- 关键词
- 上层关键词
- PPT 是否存在
- 关系计算所需的其他信息

### 6.2 文本与文件路径

当前 V2 采用的是“双保存思路”：

- 一方面保存全文本
- 一方面保存文件路径

原因是：

- 全文本便于搜索、预览、关系判断
- 文件路径便于前端以后打开本机文件或显示文件来源

这也是为什么当前 V2 同时支持：

- `transcript_text` / `summary_text`
- `transcript_file_path` / `summary_file_path`

## 7. 知识库 V2 对外提供哪些输出数据

知识库 V2 的主输出现在分成两类：

- 核心数据输出
- 图谱数据输出

旧的页面化输出仍然保留，但只是兼容层。

### 7.1 主输出一：核心数据

接口：

- `export_core_data(selected_activity_id=None)`

这类输出适合前端或其他模块直接使用，因为它更像“原始业务数据”，而不是已经替你排好页面布局的数据。

它主要返回：

- `activities`
- `selected_activity`
- `content_lines`
- `counts`

通俗理解：

- `activities`
  - 所有活动的基础数据列表

- `selected_activity`
  - 如果你指定了某个活动 ID，这里会给你那一条活动的完整数据

- `content_lines`
  - 知识库根据活动关系整理出的“内容主线”

- `counts`
  - 一些统计信息，例如活动数量、主线数量、附件数量

其中单条活动通常会包含：

- `activity_id`
- `activity_name`
- `activity_intro`
- `start_time`
- `end_time`
- `duration_minutes`
- `transcript_text`
- `summary_text`
- `summary_of_summary`
- `keywords`
- `keywords_of_keywords`
- `ppt_present`
- `activity_dir`
- `transcript_file_path`
- `summary_file_path`
- `ppt_file_path`
- `ppt_id`
- `ppt_text_excerpt`
- `matched_slides`
- `transcript_meta`
- `summary_meta`
- `relations`
- `content_line`
- `files`

可以把它理解为：

- 一份足够完整的“活动数据包”
- 前端可以自己决定怎么把这些数据做成列表、详情页、筛选器、搜索结果或预览页

### 7.2 主输出二：图谱数据

接口：

- `export_graph_view()`

它主要返回：

- `nodes`
- `edges`

通俗理解：

- `nodes`
  - 每个活动在知识图谱里对应一个节点

- `edges`
  - 两个活动之间的关系边

节点通常包含：

- 活动 ID
- 活动名称
- 活动简介
- 简短摘要
- 关键词
- 上层关键词
- 开始时间
- 是否有 PPT

边通常包含：

- `relation_id`
- `source_activity_id`
- `target_activity_id`
- `strength`
- `state`
- `reasons`
- `source_type`

适用场景：

- 前端做知识图谱可视化
- 上层模块做活动关联分析
- 后续做关系审核或人工修正

### 7.3 兼容输出：旧式 view bundle

接口：

- `export_view_bundle(selected_activity_id=None)`

这组输出不是 V2 未来主方向，但当前还保留，主要为了：

- 兼容旧验证方式
- CLI 演示
- 调试时快速查看结果

它返回的是已经带有“页面区域含义”的数据块，例如：

- `navigation`
- `history`
- `relation_map`
- `timeline_calendar`
- `timeline_line_view`
- `file_lookup`
- `detail_panel`

如果你是在开发新的前端，不建议优先依赖这组输出。
更推荐优先使用：

- `export_core_data()`
- `export_graph_view()`

### 7.4 组合输出

接口：

- `export_all_views(selected_activity_id=None)`

它一次性返回：

- `core_data`
- `graph_view`
- `legacy_view_bundle`

这个接口更适合：

- CLI demo
- 联调检查
- 快速查看当前知识库状态

## 8. 知识库 V2 的搜索能力是什么

知识库当前保留搜索能力，但搜索不是默认主输出的一部分。

接口包括：

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
- 文件路径字段
- 时间字段

要注意：

- 知识库负责提供搜索能力
- 前端负责决定什么时候搜、怎么展示结果

也就是说，V2 的思路仍然是“知识库给数据和能力，前端负责功能分区和交互”。

## 9. 其他开发者接这个模块时，最重要的几条原则

- 不要把前端页面结构写回知识库核心接口里
- 不要让知识库依赖前端组件的显示需求来改字段语义
- 不要把知识库当成转录模块或总结模块
- 主接口优先看 `export_core_data()` 和 `export_graph_view()`
- `export_view_bundle()` 只当兼容层，不当未来唯一主接口
- `transcript_meta` 和 `summary_meta` 是可选诊断信息，不是主业务字段
- `activity_name` 和 `activity_intro` 支持默认值，也支持后续用户编辑

## 10. 当前推荐的交接理解方式

如果把整个知识库 V2 用一句话说明给其他开发者，可以这样说：

“知识库 V2 接收一条完整活动记录，保存活动全文、摘要、关键词、路径和关系信息，再以数据优先的方式对外输出活动数据和图谱数据；旧的页面化 bundle 仍保留，但只是兼容层，不是未来主接口。”
