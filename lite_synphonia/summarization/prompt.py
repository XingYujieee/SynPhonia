from __future__ import annotations

import re

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_CJK_THRESHOLD: float = 0.15
_MAX_CONSOLIDATION_INPUTS: int = 16


def _detect_language(text: str) -> str:
    stripped = text.replace(" ", "").replace("\n", "")
    if not stripped:
        return "en"
    cjk_ratio = len(_CJK_RE.findall(stripped)) / len(stripped)
    return "zh" if cjk_ratio >= _CJK_THRESHOLD else "en"


def _build_chinese_prompt(window_text: str, previous_summary: str = "") -> str:
    dedup_section = ""
    if previous_summary:
        dedup_section = (
            "## 上一段摘要（仅供去重参考，不要重复其内容）\n"
            f"{previous_summary}\n\n"
        )

    return (
        "你是一名专业的课程笔记整理员，擅长中文和中英混合讲座内容的精炼摘要。\n\n"
        "## 任务\n"
        "阅读下方【讲座片段】，严格输出以下 JSON，不得包含任何其他内容：\n"
        '{"keywords":["词1","词2","词3"],"summary":"摘要正文。"}\n\n'
        + dedup_section
        + "## 输出规则\n"
        "keywords（关键词数组，3～5 个）：\n"
        "  - 优先选取专业术语、核心概念、重要动词；\n"
        "  - 保留原文中的英文缩写（如 MSE、SSIM、FFT、LPF）；\n"
        "  - 排除虚词（的、了、是、在）和口语填充词（嗯、就是、那个）。\n\n"
        "summary（摘要正文，中文）：\n"
        "  - 总字数控制在 50 字左右（40～60 字）；\n"
        "  - 如果片段内容是罗列、枚举型（如列举多个名词、步骤、条目），只需一句话概括要点，控制在 30～40 字；\n"
        "  - 如果片段包含概念解释、原理推导或因果论证，可以适当展开，但也不超过 60 字；\n"
        "  - 提炼本段的核心论点或操作步骤，而不仅仅是重复片段开头；\n"
        "  - 保留英文专有名词原样（不翻译）；\n"
        "  - 如有因果或流程关系，用「因此」「从而」「进而」等连接词体现；\n"
        "  - 禁止续写或补充超出片段内容的内容；\n"
        "  - 不要重复上一段摘要已经涵盖的内容，聚焦本段新出现的要点。\n\n"
        "## 不得包含\n"
        "  - Markdown 标记（**、##、```）\n"
        "  - 代码块或代码围栏\n"
        "  - 任何解释性前缀（如「摘要：」「以下是」）\n"
        "  - JSON 之外的任何文字\n\n"
        "## 示例（格式参考，内容与本次无关）\n"
        '{"keywords":["低通滤波器","截止频率","MSE"],'
        '"summary":"讲座介绍了低通滤波器的设计方法，重点讨论截止频率选取原则，'
        '3×3 盒式滤波器在 MSE 上优于高斯核。"}\n\n'
        "## 讲座片段\n"
        f"{window_text}\n"
    )


def _build_english_prompt(window_text: str, previous_summary: str = "") -> str:
    dedup_section = ""
    if previous_summary:
        dedup_section = (
            "## Previous Summary (for deduplication only — do NOT repeat its content)\n"
            f"{previous_summary}\n\n"
        )

    return (
        "You are a professional lecture-notes editor specialising in dense technical content.\n\n"
        "## Task\n"
        "Read the [Transcript Window] below and output exactly this JSON — nothing else:\n"
        '{"keywords":["kw1","kw2","kw3"],"summary":"Summary text."}\n\n'
        + dedup_section
        + "## Rules\n"
        "keywords (array, 3-5 items):\n"
        "  - High-signal domain terms, technical nouns, key action verbs.\n"
        "  - Preserve technical abbreviations exactly (MSE, FFT, SSIM, LPF).\n"
        "  - Exclude filler words (will, may, have, with, just, basically).\n\n"
        "summary (string):\n"
        "  - 1-2 sentences, around 30-40 words.\n"
        "  - If the content is a list or enumeration, keep it to one concise sentence (20-30 words).\n"
        "  - If the content explains concepts or reasoning, allow slightly more detail (up to 40 words).\n"
        "  - Condense the core argument or procedure.\n"
        "  - Preserve English technical terms unchanged.\n"
        "  - Use connective words (therefore, thus, consequently) to capture cause-effect links.\n"
        "  - Do NOT continue the transcript or invent content beyond the window.\n"
        "  - Do NOT repeat content already covered in the previous summary; focus on new points.\n\n"
        "## Forbidden\n"
        "  - Markdown syntax (**, ##, ```)\n"
        "  - Code fences or blocks\n"
        "  - Explanatory prefixes (e.g. 'Summary:', 'Here is')\n"
        "  - Any text outside the JSON object\n\n"
        "## Transcript Window\n"
        f"{window_text}\n"
    )


def build_summarization_prompt(
    window_text: str, *, language: str = "auto", previous_summary: str = "",
) -> str:
    lang = _detect_language(window_text) if language == "auto" else language
    if lang == "zh":
        return _build_chinese_prompt(window_text, previous_summary)
    return _build_english_prompt(window_text, previous_summary)


def build_consolidation_prompt(summaries: list[str], *, language: str = "auto") -> str:
    capped = summaries[-_MAX_CONSOLIDATION_INPUTS:]
    bullet_list = "\n".join(f"[{i + 1}] {s.strip()}" for i, s in enumerate(capped) if s.strip())
    combined_sample = " ".join(capped[:3])
    lang = _detect_language(combined_sample) if language == "auto" else language

    if lang == "zh":
        return (
            "你是一名专业的课程内容整理员。\n"
            "以下是同一场讲座的多个分段摘要（按时间顺序编号）。\n"
            "请将它们整合为一篇连贯的整体摘要，严格输出以下 JSON，不得包含任何其他内容：\n"
            '{"keywords":["词1","词2","词3"],"summary":"整体摘要正文。"}\n\n'
            "## 整合规则\n"
            "keywords（5～8 个核心术语，覆盖全文主题，保留英文缩写）\n"
            "summary（整体摘要，中文）：3～5 句话，不超过 220 字，体现逻辑顺序。\n\n"
            "## 各分段摘要（按时序）\n"
            f"{bullet_list}\n"
        )
    return (
        "You are a professional lecture-notes editor.\n"
        "Below are time-ordered segment summaries from a single lecture.\n"
        "Consolidate them into one coherent final summary. Output exactly this JSON:\n"
        '{"keywords":["kw1","kw2","kw3"],"summary":"Consolidated summary."}\n\n'
        "## Segment Summaries (chronological)\n"
        f"{bullet_list}\n"
    )
