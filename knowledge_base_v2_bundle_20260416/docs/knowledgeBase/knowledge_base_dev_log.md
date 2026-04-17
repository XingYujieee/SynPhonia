# Knowledge Base Development Log

## Entry 001

Date: 2026-04-14
Task: Establish the standalone knowledge base module and its validation scaffolding

- What changed:
  - Defined the independent implementation area for the knowledge base under `knowledge_base/`.
  - Added interface and boundary documentation so future integration can target explicit inputs and outputs rather than historical module internals.
  - Added a frontend-free validation approach based on sample activities, exported view bundles, CLI inspection, and automated tests.
- Related modules:
  - `knowledge_base/`
  - `knowledge_base_samples/`
  - `tests/knowledge_base/`
  - `docs/knowledgeBase/`
- Unfinished tail:
  - Real integration with a future activity coordinator remains outside the current task scope.
  - No production frontend consumes the new outputs yet.
- Future optimization and technical debt:
  - If the knowledge base later grows large, dynamic scanning of per-activity files may need to be replaced by a lightweight index cache.
  - Search is intentionally predictable and lightweight in v1; semantic retrieval is reserved for a later phase.

## Entry 002

Date: 2026-04-14
Task: Implement the first runnable knowledge base core and the frontend-free validation flow

- What changed:
  - Implemented the standalone knowledge base package with unified activity input validation, per-activity storage, rule-based relation generation, content-line construction, search helpers, and view-bundle export logic.
  - Added a CLI entrypoint that supports JSON ingestion, view export, wide-scope search, and user relation overrides without requiring any frontend code.
  - Added sample activity data and placeholder local PPT assets so the module can be validated through stable, repeatable example inputs.
  - Added automated tests covering ingestion, optional-field tolerance, relation and content-line behavior, wide search, and manual relation override persistence.
  - Refined the relation rules to reduce cross-scene false positives and keep lighter continuation links in the pending-confirmation path.
- Related modules:
  - `knowledge_base/`
  - `knowledge_base_samples/`
  - `tests/knowledge_base/test_service.py`
  - `docs/knowledgeBase/README.md`
  - `docs/knowledgeBase/knowledge_base_interface.md`
  - `docs/knowledgeBase/knowledge_base_boundary.md`
  - `docs/knowledgeBase/knowledge_base_dev_log.md`
- Unfinished tail:
  - The future upper activity coordinator is still outside the current implementation scope, so real ingestion currently relies on direct service calls or CLI input files.
  - No production frontend consumes the exported view bundle yet.
- Future optimization and technical debt:
  - The current relation rules are intentionally lightweight and interpretable, but further domain-specific tuning may be needed once real activity data replaces the sample dataset.
  - PPT handling in v1 is limited to local-path inspection plus external-open compatibility; in-app PPT rendering remains intentionally out of scope.

## Entry 003

Date: 2026-04-15
Task: 固化知识库下一阶段的中文契约文档

- What changed:
  - 新增了中文的 `knowledge_base_v2_contract.md`，把知识库下一轮开发的输入契约、输出契约、边界调整和开发顺序正式写明。
  - 明确确认 `summary_of_summary`、`keywords_of_keywords`、`activity_name`、`activity_intro` 为上游必须提供的正式字段。
  - 在文档中明确了缺失关键字段时的处理要求：知识库必须返回清晰异常，不能静默吞掉或偷偷兜底为“成功入库”。
  - 对当前 v1 中的 `view bundle` 做了术语澄清，并将其定位为兼容层或调试层，而不是后续主接口。
  - 更新了知识库文档首页，使其区分“当前 v1 实现文档”和“后续 v2 目标契约文档”。
- Related modules:
  - `docs/knowledgeBase/knowledge_base_v2_contract.md`
  - `docs/knowledgeBase/README.md`
  - `docs/knowledgeBase/knowledge_base_dev_log.md`
  - `docs/project-log.md`
- Unfinished tail:
  - 当前代码中的 schema、storage、service 和 CLI 仍未正式实现 v2 契约。
  - `knowledge_base_interface.md` 记录的仍主要是当前 v1 实现，而不是本次确认后的目标状态。
- Future optimization and technical debt:
  - 当 v2 代码实现开始后，需要决定 `view bundle` 是彻底退役，还是保留为兼容导出接口。
  - 如果未来既要支持“文本内容直接传入”又要支持“仅根据文件路径补载”，需要更严格地界定最小可接受输入集和错误恢复策略。

## Entry 004

Date: 2026-04-16
Task: 实现知识库 V2 的首轮代码落地

- What changed:
  - 将活动 ingestion schema 升级为 V2 形态，正式接入 `summary_of_summary`、`keywords_of_keywords`、`activity_intro`、`activity_name`，并支持 `activity_dir`、`transcript_file_path`、`summary_file_path`。
  - 增加了显式输入校验失败返回；当关键字段缺失或格式非法时，service 现在返回 `status=invalid_input`、`missing_fields` 和 `invalid_fields`。
  - 调整了存储策略：知识库现在会把完整文本直接存入 metadata，同时保留文件路径引用；只有在缺少文本文件路径时才回退到 workspace 内部生成文本文件。
  - 新增了 `export_core_data()` 和 `export_graph_view()` 两类主输出，同时保留 `export_view_bundle()` 作为兼容层，并增加 `export_all_views()` 作为 CLI demo 的组合导出。
  - 更新了搜索字段范围，使搜索能力覆盖活动名称、活动简介、summary_of_summary、keywords_of_keywords 以及新的文件路径字段。
  - 更新了样例数据和自动化测试，使其符合新的输入契约，并覆盖缺字段异常、外部文件引用、回退文件生成、图谱关系和兼容视图。
- Related modules:
  - `knowledge_base/schemas.py`
  - `knowledge_base/storage.py`
  - `knowledge_base/service.py`
  - `knowledge_base/views.py`
  - `knowledge_base/search.py`
  - `knowledge_base/relations.py`
  - `knowledge_base/cli.py`
  - `knowledge_base_samples/sample_activities.json`
  - `tests/knowledge_base/test_service.py`
  - `docs/knowledgeBase/knowledge_base_interface.md`
  - `docs/knowledgeBase/README.md`
  - `docs/knowledgeBase/knowledge_base_dev_log.md`
  - `docs/project-log.md`
- Unfinished tail:
  - 当前 `export_view_bundle()` 仍然存在，说明兼容层尚未退役。
  - 当前搜索仍然是规则型字符串匹配，还没有分层为“前端轻筛选”和“知识库全文搜索”的独立策略。
  - 当前 CLI 主要用于无前端验证，尚未与真实上层协调器完成联调。
- Future optimization and technical debt:
  - 如果未来活动量变大，当前把完整文本直接存入 metadata 的方式可能需要配合索引或分页策略。
  - 当前知识图谱仍是规则驱动，后续可能需要结合真实活动数据重新调优关系权重和停用词。
  - 当前为了兼容旧验证链路仍保留 legacy bundle；后续如果前端完全切换到主接口，可以考虑逐步收缩该兼容层。

## Entry 005

Date: 2026-04-16
Task: 补充一份面向其他开发者的知识库 V2 通俗接口说明

- What changed:
  - 新增了 `knowledge_base_v2_plain_interface.md`，用更通俗的语言重新说明知识库 V2 的输入字段、默认值约定、输出接口、兼容层定位和搜索边界。
  - 在文档里明确区分了“完整活动入库”“核心数据输出”“图谱数据输出”和“legacy view bundle 兼容输出”，便于后续交接时快速建立统一理解。
  - 更新了 `README.md` 的文档索引，方便后续打包时直接把这份说明交给其他开发者阅读。
- Related modules:
  - `docs/knowledgeBase/knowledge_base_v2_plain_interface.md`
  - `docs/knowledgeBase/README.md`
  - `docs/knowledgeBase/knowledge_base_dev_log.md`
- Unfinished tail:
  - 当前这份文档主要面向“独立知识库模块”交接；如果后续真的并入完整应用，还需要再补一份“上层协调器如何适配旧输入”的集成说明。
  - `activity_name` 默认序号和 `activity_intro` 默认取值虽然已经写入文档约定，但当前仍更适合由上层协调器或适配层统一落实。
- Future optimization and technical debt:
  - 如果后续前端接口最终完全稳定，可以把 `knowledge_base_interface.md` 和这份通俗版说明做进一步分层，减少内容重复。
  - 如果后续加入字段更新接口，建议再补一份“可编辑字段生命周期说明”，把默认值、用户修改值和最终持久化值分开讲清楚。

## Entry 006

Date: 2026-04-16
Task: 打包知识库 V2 相关文档与代码，供外部开发者交接使用

- What changed:
  - 按当前知识库 V2 的边界整理了需要交付的目录范围，包括代码、样例数据、测试和文档。
  - 生成面向交接的压缩包，保存在项目根目录下，便于直接发送给其他开发者。
  - 打包时排除了 `__pycache__` 一类运行时缓存文件，避免把无关产物混入交付物。
- Related modules:
  - `knowledge_base/`
  - `knowledge_base_samples/`
  - `tests/knowledge_base/`
  - `docs/knowledgeBase/`
- Unfinished tail:
  - 当前压缩包主要面向“独立知识库模块”交接，不包含未来完整应用集成时可能需要的上层适配代码。
  - 如果后续知识库接口继续演进，需要重新打包新的交付版本。
- Future optimization and technical debt:
  - 如果后续交付频率变高，可以补一个自动化打包脚本，统一压缩包命名、打包范围和排除规则。
  - 如果后续需要长期维护多个交付版本，可以考虑增加交付清单或版本清单文件。
