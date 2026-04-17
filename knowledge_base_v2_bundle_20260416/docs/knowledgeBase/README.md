# 知识库模块文档

本目录用于存放独立知识库模块的实现文档、边界文档和开发记录。

## 当前文档清单

- `knowledge_base_interface.md`
  - 记录当前 v1 实现的输入输出契约与调用边界
- `knowledge_base_v2_plain_interface.md`
  - 用通俗语言说明当前知识库 V2 需要接收什么数据、会输出什么数据
  - 适合打包给其他开发者做交接阅读
- `knowledge_base_v2_contract.md`
  - 记录已经确认的下一阶段目标契约
  - 后续代码调整应优先对齐这份文档
- `knowledge_base_boundary.md`
  - 说明知识库与历史模块之间的职责边界
- `knowledge_base_dev_log.md`
  - 记录知识库开发过程中的阶段性修改
- `new_dev_require.txt`
  - 新一轮需求原始文本

## 当前实现位置

- 代码：`knowledge_base/`
- 样例输入：`knowledge_base_samples/`
- 测试：`tests/knowledge_base/`

## 当前状态说明

当前知识库已经完成第一轮 V2 落地，核心变化包括：

- 主输出已经增加为“核心数据输出 + 知识图谱输出”
- 旧的 `view bundle` 仍保留为兼容层
- 输入字段已经扩展到 `summary_of_summary`、`keywords_of_keywords`、`activity_intro`、`activity_name`
- 缺失关键字段时会返回明确的 `invalid_input` 结果
- 文本内容和文件路径会同时保存，兼顾知识库内部检索与前端文件预览

## 当前无前端验证方式

在不接入前端的情况下，推荐优先通过 CLI 验证当前实现：

```powershell
.\.venv\Scripts\python.exe -m knowledge_base.cli demo `
  --workspace .tmp_test_runs\kb_demo `
  --activities knowledge_base_samples\sample_activities.json `
  --output .tmp_test_runs\kb_demo\demo_output.json `
  --reset `
  --selected-activity activity-classroom-002
```

然后检查：

- 导出的 `core_data`
- 导出的 `graph_view`
- 导出的 `legacy_view_bundle`
- `tests/knowledge_base/` 下的自动化测试
- workspace 中生成的活动记录文件

说明：

- `demo` 当前导出的是“当前主接口 + 兼容层”的组合结果
- 如果只想导出当前主接口，可以使用 `export-data`
