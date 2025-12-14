
"""
Week05 Homework - MCP-style Multi-Agent Article Writer

说明（重要）：
- 本仓库目录名为 `multi-agent/`（包含连字符），它不是合法的 Python 包名，因此
  README 中的 `python -m multi-agent.main` 在标准 Python 下无法工作。
- 本作业入口保留在本文件：推荐直接运行：

    python /path/to/week05-homework/multi-agent/main.py "帮我写一篇关于AI Agent的文章"

功能：
- 四个代理按顺序协作：Research -> Writing -> Review -> Polishing
- 代理间通过“类 MCP”的结构化消息（JSON风格）通信，并保留上下文
- 终端实时展示协作过程
- 生成 `report.md`：包含完整过程记录、最终文章、异常处理日志（含三级重试）
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
import re
import sys
import textwrap
import traceback
from typing import Any, Dict, List, Optional, Tuple

try:
    # optional dependency from pyproject.toml
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore


# -----------------------------
# MCP-ish message primitives
# -----------------------------


@dataclasses.dataclass(frozen=True)
class MCPMessage:
    """A minimal, MCP-style structured message."""

    sender: str
    msg_type: str  # e.g. "event" | "artifact" | "error" | "tool"
    content: Any
    ts: str = dataclasses.field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> Dict[str, Any]:
        return {"ts": self.ts, "sender": self.sender, "type": self.msg_type, "content": self.content}


@dataclasses.dataclass
class RunConfig:
    question: str
    style: str = "科普 / 技术入门"
    length: str = "medium"  # short|medium|long
    language: str = "zh"
    output_path: str = ""
    show_json: bool = False
    quiet: bool = False


@dataclasses.dataclass
class ExceptionRecord:
    agent: str
    level: int
    attempt: int
    error_type: str
    error_message: str
    traceback: str
    ts: str = dataclasses.field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))


class ConsoleLogger:
    def __init__(self, quiet: bool = False, show_json: bool = False):
        self.quiet = quiet
        self.show_json = show_json

    def _print(self, s: str) -> None:
        if not self.quiet:
            print(s, flush=True)

    def event(self, who: str, what: str) -> None:
        self._print(f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {who}: {what}")

    def message(self, msg: MCPMessage) -> None:
        if self.quiet:
            return
        if self.show_json:
            self._print(json.dumps(msg.to_dict(), ensure_ascii=False))
        else:
            preview = msg.content
            if isinstance(preview, (dict, list)):
                preview = json.dumps(preview, ensure_ascii=False)
            preview_s = str(preview)
            preview_s = preview_s if len(preview_s) <= 240 else preview_s[:240] + "…"
            self._print(f"[{msg.ts}] {msg.sender}({msg.msg_type}): {preview_s}")


# -----------------------------
# Tools (offline-first)
# -----------------------------


class SearchTool:
    """
    A "search tool" abstraction.
    - Offline-first: built-in mini corpus
    - Optional: web search if user enables it and network is available
    """

    _KB: Dict[str, List[Dict[str, str]]] = {
        "ai agent": [
            {
                "title": "AI Agent（概念）",
                "snippet": "AI Agent 通常指能够感知环境、进行推理规划、调用工具并执行动作以达成目标的智能体。现代 LLM Agent 常结合工具调用、记忆与规划。",
                "source": "built-in",
            },
            {
                "title": "ReAct（思考-行动-观察）范式",
                "snippet": "ReAct 将推理与行动交织：模型生成行动（工具调用）并基于观察继续推理，常用于提升多步任务的可控性与可解释性。",
                "source": "built-in",
            },
            {
                "title": "多代理协作",
                "snippet": "多代理系统通过角色分工（研究/撰写/审核/润色等）与结构化通信提升复杂内容生成质量，并便于加入验证、重试与审计日志。",
                "source": "built-in",
            },
            {
                "title": "MCP（Model Context Protocol）",
                "snippet": "MCP 是一种让模型以统一方式访问外部上下文与工具的协议思路，强调结构化消息、可插拔工具与可追踪的上下文传递。",
                "source": "built-in",
            },
        ],
        "mcp": [
            {
                "title": "MCP（协议思路）",
                "snippet": "MCP 关注：标准化上下文传递（messages/resources）、工具能力暴露（tools）、以及可审计的调用记录（logs）。",
                "source": "built-in",
            }
        ],
    }

    def __init__(self, enable_web: bool = False):
        self.enable_web = enable_web and os.getenv("ENABLE_WEB_SEARCH", "").strip() in {"1", "true", "True"}

    def search(self, query: str, top_k: int = 6) -> List[Dict[str, str]]:
        q = query.strip().lower()
        hits: List[Dict[str, str]] = []
        for key, docs in self._KB.items():
            if key in q or any(tok in key for tok in q.split()):
                hits.extend(docs)
        if hits:
            return hits[:top_k]
        # fallback: fuzzy match
        for key, docs in self._KB.items():
            if any(tok in q for tok in key.split()):
                hits.extend(docs)
        return hits[:top_k] if hits else [{"title": "无本地命中", "snippet": "未找到直接匹配的本地资料，后续将基于常识与通用写作框架组织内容。", "source": "built-in"}]


# -----------------------------
# Agents
# -----------------------------


class AgentError(RuntimeError):
    pass


class BaseAgent:
    name: str = "Agent"

    def run(self, ctx: List[MCPMessage], cfg: RunConfig) -> Dict[str, Any]:
        raise NotImplementedError

    def _latest_artifact(self, ctx: List[MCPMessage], sender: str, artifact_key: str) -> Optional[Any]:
        for msg in reversed(ctx):
            if msg.sender == sender and msg.msg_type == "artifact":
                c = msg.content
                if isinstance(c, dict) and artifact_key in c:
                    return c[artifact_key]
        return None


class ResearchAgent(BaseAgent):
    name = "Research Agent"

    def __init__(self, tool: SearchTool):
        self.tool = tool

    def run(self, ctx: List[MCPMessage], cfg: RunConfig) -> Dict[str, Any]:
        query = cfg.question
        results = self.tool.search(query, top_k=6)
        notes = []
        for r in results:
            notes.append(
                {
                    "claim": r["snippet"],
                    "source_title": r["title"],
                    "source": r.get("source", "unknown"),
                }
            )
        outline = [
            "什么是 AI Agent（与传统模型/脚本的差异）",
            "核心组成：感知、记忆、推理/规划、工具调用、执行与反馈",
            "主流范式：ReAct、Plan-and-Execute、反思/自我修正",
            "多代理协作：分工、通信协议（类MCP）、质量控制与审计",
            "实践建议：如何从 0 到 1 搭建一个可用的 Agent 系统",
            "风险与边界：幻觉、对齐、安全、成本与评估",
        ]
        return {"research": {"query": query, "notes": notes, "suggested_outline": outline}}


class ResearchAgentBackup(ResearchAgent):
    name = "Research Agent (Backup)"

    def run(self, ctx: List[MCPMessage], cfg: RunConfig) -> Dict[str, Any]:
        # Backup: be more conservative and add explicit uncertainty markers
        base = super().run(ctx, cfg)
        for n in base["research"]["notes"]:
            n["confidence"] = "medium (offline/local)"
        base["research"]["assumptions"] = [
            "未启用联网检索，研究资料来自内置知识库与通用方法论总结。",
            "如需引用权威来源，请在启用联网检索后补充参考链接。",
        ]
        return base


class WritingAgent(BaseAgent):
    name = "Writing Agent"

    def run(self, ctx: List[MCPMessage], cfg: RunConfig) -> Dict[str, Any]:
        research = self._latest_artifact(ctx, "Research Agent", "research") or self._latest_artifact(ctx, "Research Agent (Backup)", "research")
        if not research:
            raise AgentError("Missing research artifact")

        outline: List[str] = research.get("suggested_outline", [])
        notes: List[Dict[str, str]] = research.get("notes", [])

        target_words = {"short": 800, "medium": 1400, "long": 2200}.get(cfg.length, 1400)
        title = self._guess_title(cfg.question)
        lead = f"当我们谈论 AI Agent 时，谈的不是“会聊天的模型”，而是一套能把目标拆解成步骤、会调用工具并能对结果负责的执行系统。本文用通俗但不失严谨的方式，带你从 0 到 1 理解并搭建 AI Agent。"

        sections = []
        for i, sec in enumerate(outline, start=1):
            sections.append(self._write_section(i, sec, notes))

        draft = self._assemble_article(title, lead, sections, notes, target_words, cfg.style)
        return {"draft": {"title": title, "style": cfg.style, "target_words": target_words, "content": draft}}

    def _guess_title(self, question: str) -> str:
        q = re.sub(r"[，。！？!?.]", "", question).strip()
        if "AI Agent" in q or "agent" in q.lower():
            return "从 0 到 1 认识 AI Agent：让大模型成为“会做事”的系统"
        return f"主题文章：{q[:24]}"

    def _write_section(self, idx: int, heading: str, notes: List[Dict[str, str]]) -> str:
        # Use notes as inspiration; keep it deterministic and readable.
        n1 = notes[idx % len(notes)] if notes else {"claim": "（略）", "source_title": "N/A", "source": "built-in"}
        body = (
            f"{heading}：\n"
            f"- 关键点：{n1.get('claim','')}\n"
            f"- 展开说明：把 Agent 想成“带工具箱的项目经理”。它不仅生成文字，还会选择合适的工具（搜索、计算、数据库、代码执行等），"
            f"并把每次行动的结果写回上下文，从而形成可追踪的闭环。\n"
            f"- 小结：这一部分的重点是明确边界——Agent 不是魔法，而是流程 + 约束 + 工具的工程化组合。\n"
        )
        return body

    def _assemble_article(
        self,
        title: str,
        lead: str,
        sections: List[str],
        notes: List[Dict[str, str]],
        target_words: int,
        style: str,
    ) -> str:
        refs = []
        for n in notes[:6]:
            refs.append(f"- {n.get('source_title','')}: {n.get('source','')}")
        article = f"## {title}\n\n**风格**：{style}\n\n{lead}\n\n"
        article += "\n".join([f"### {s}" if not s.startswith("###") else s for s in sections])
        article += "\n\n### 参考与素材（研究代理输出）\n" + ("\n".join(refs) if refs else "- （无）") + "\n"

        # Soft length control: pad with practical checklist for longer articles
        current_words = self._count_approx_words(article)
        if current_words < target_words:
            extra = self._practical_checklist()
            article = article + "\n\n" + extra
        return article.strip() + "\n"

    def _practical_checklist(self) -> str:
        return textwrap.dedent(
            """\
            ### 实践清单：搭建一个“可交付”的 Agent
            1. 明确目标与输入输出：不要让 Agent 自己猜验收标准。
            2. 设计上下文结构：把需求、约束、已知事实、工具结果分区存储。
            3. 工具先行：先把 Search/DB/Code 等工具做成稳定接口，再让 Agent 调用。
            4. 加上审核与回滚：让 Review 代理专门找漏洞；关键步骤保留可回放日志。
            5. 评估与成本控制：为每个任务定义正确率、耗时、token/调用成本等指标。
            """
        ).strip()

    def _count_approx_words(self, s: str) -> int:
        # Rough estimator: Chinese chars count as 1, english words as 1.
        zh = len(re.findall(r"[\u4e00-\u9fff]", s))
        en = len(re.findall(r"[A-Za-z0-9]+", s))
        return zh + en


class ReviewAgent(BaseAgent):
    name = "Review Agent"

    def run(self, ctx: List[MCPMessage], cfg: RunConfig) -> Dict[str, Any]:
        draft = self._latest_artifact(ctx, "Writing Agent", "draft")
        if not draft:
            raise AgentError("Missing draft artifact")
        content: str = draft.get("content", "")
        issues = []
        suggestions = []

        if "###" not in content:
            issues.append("文章缺少分段小标题（###）。")
            suggestions.append("按“定义-组成-方法-实践-风险”的逻辑加入小标题。")

        if "风险" not in content and "边界" not in content:
            issues.append("缺少风险/边界讨论，容易显得过度乐观。")
            suggestions.append("增加“幻觉/安全/评估/成本”相关段落。")

        if "参考" not in content:
            issues.append("缺少参考与素材来源说明。")
            suggestions.append("加入研究代理的素材列表，标注来源类型。")

        # simple coherence check
        if len(content) < 600:
            issues.append("内容偏短，信息密度不够。")
            suggestions.append("补充一个落地示例：如何用四代理流水线产出文章。")

        verdict = "pass" if len(issues) <= 2 else "revise"
        return {"review": {"verdict": verdict, "issues": issues, "suggestions": suggestions}}


class ReviewAgentSenior(ReviewAgent):
    name = "Review Agent (Senior)"

    def run(self, ctx: List[MCPMessage], cfg: RunConfig) -> Dict[str, Any]:
        base = super().run(ctx, cfg)
        base["review"]["additional_checks"] = [
            "结构：是否从概念到工程落地逐步收敛",
            "一致性：术语是否统一（Agent/工具/上下文）",
            "可执行性：实践清单是否能直接照做",
        ]
        return base


class PolishingAgent(BaseAgent):
    name = "Polishing Agent"

    def run(self, ctx: List[MCPMessage], cfg: RunConfig) -> Dict[str, Any]:
        draft = self._latest_artifact(ctx, "Writing Agent", "draft")
        review = self._latest_artifact(ctx, "Review Agent", "review") or self._latest_artifact(ctx, "Review Agent (Senior)", "review")
        if not draft:
            raise AgentError("Missing draft artifact")

        content: str = draft.get("content", "")
        issues = (review or {}).get("issues", [])
        suggestions = (review or {}).get("suggestions", [])

        polished = content
        # Apply a few deterministic improvements
        polished = re.sub(r"：\n-", "：\n- ", polished)
        polished = polished.replace("（略）", "（此处给出可操作解释）")
        if issues and any("风险" in x or "边界" in x for x in issues):
            polished += "\n\n### 风险与边界（务必正视）\n"
            polished += "- 幻觉：模型可能自信地编造事实，关键结论要有可验证来源。\n"
            polished += "- 安全：工具调用要做权限隔离与参数校验，避免越权与注入。\n"
            polished += "- 评估：需要离线测试集 + 在线监控，别只看“写得像”。\n"
            polished += "- 成本：多代理会放大调用次数，应做缓存、短路与分级策略。\n"

        polished += "\n\n---\n**润色说明**：根据审核建议做了结构与表达增强；如需更强“学术风格/营销风格/极简风格”，可通过参数调整。\n"
        return {"final_article": {"title": draft.get("title", ""), "content": polished, "applied_suggestions": suggestions}}


# -----------------------------
# Orchestration + retries
# -----------------------------


def _validate_agent_output(agent_name: str, output: Dict[str, Any]) -> None:
    if not isinstance(output, dict) or not output:
        raise AgentError(f"{agent_name} returned empty output")
    # Minimal contract checks per agent type
    required = {
        "Research Agent": "research",
        "Research Agent (Backup)": "research",
        "Writing Agent": "draft",
        "Review Agent": "review",
        "Review Agent (Senior)": "review",
        "Polishing Agent": "final_article",
    }.get(agent_name)
    if required and required not in output:
        raise AgentError(f"{agent_name} output missing key: {required}")


def run_agent_with_retries(
    agent: BaseAgent,
    ctx: List[MCPMessage],
    cfg: RunConfig,
    logger: ConsoleLogger,
    exceptions: List[ExceptionRecord],
    backup_agent: Optional[BaseAgent] = None,
    can_ask_user: bool = True,
) -> Dict[str, Any]:
    """
    三级重试策略（选做实现）：
    - 一级：同一代理重试（最多2次）
    - 二级：切换备用代理（最多1次）
    - 三级：向用户请求补充信息（无法交互时自动使用默认补充）
    """

    # Level 1: same agent retry up to 2 times
    for attempt in range(1, 3):
        try:
            logger.event(agent.name, f"开始执行（一级重试尝试 {attempt}/2）")
            out = agent.run(ctx, cfg)
            _validate_agent_output(agent.name, out)
            return out
        except Exception as e:  # noqa: BLE001
            rec = ExceptionRecord(
                agent=agent.name,
                level=1,
                attempt=attempt,
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc(),
            )
            exceptions.append(rec)
            logger.event(agent.name, f"执行失败：{rec.error_type}: {rec.error_message}")

    # Level 2: switch to backup
    if backup_agent is not None:
        try:
            logger.event(backup_agent.name, "切换备用代理执行（二级重试）")
            out = backup_agent.run(ctx, cfg)
            _validate_agent_output(backup_agent.name, out)
            return out
        except Exception as e:  # noqa: BLE001
            rec = ExceptionRecord(
                agent=backup_agent.name,
                level=2,
                attempt=1,
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc(),
            )
            exceptions.append(rec)
            logger.event(backup_agent.name, f"执行失败：{rec.error_type}: {rec.error_message}")

    # Level 3: ask user for more info (or fallback)
    supplement = ""
    if can_ask_user and sys.stdin.isatty():
        logger.event("System", "需要补充信息（三级重试）：请补充文章目标读者/长度/重点（直接回车跳过）")
        try:
            supplement = input("> ").strip()
        except Exception:
            supplement = ""
    else:
        supplement = "默认补充：目标读者为入门到进阶工程师；重点覆盖定义、工程落地与风险边界。"

    # record the supplement into context
    ctx.append(MCPMessage(sender="User", msg_type="event", content={"supplement": supplement}))
    rec = ExceptionRecord(
        agent=agent.name,
        level=3,
        attempt=1,
        error_type="NeedMoreInfo",
        error_message="Requested/assumed additional user info for retry",
        traceback="",
    )
    exceptions.append(rec)
    logger.event("System", "已记录补充信息，回到原代理再执行一次（三级重试）")

    out = agent.run(ctx, cfg)
    _validate_agent_output(agent.name, out)
    return out


# -----------------------------
# Report rendering
# -----------------------------


def render_report(cfg: RunConfig, ctx: List[MCPMessage], exceptions: List[ExceptionRecord]) -> str:
    def md_json(obj: Any) -> str:
        return "```json\n" + json.dumps(obj, ensure_ascii=False, indent=2) + "\n```"

    # Extract artifacts
    def get_artifact(key: str) -> Optional[Dict[str, Any]]:
        for msg in reversed(ctx):
            if msg.msg_type == "artifact" and isinstance(msg.content, dict) and key in msg.content:
                return msg.content[key]
        return None

    research = get_artifact("research")
    draft = get_artifact("draft")
    review = get_artifact("review")
    final_article = get_artifact("final_article")

    lines = []
    lines.append("# 多代理文章自动编写系统（基于类MCP结构化消息）示例报告\n")
    lines.append("## 输入\n")
    lines.append(f"- **问题**：{cfg.question}\n")
    lines.append(f"- **风格**：{cfg.style}\n")
    lines.append(f"- **长度**：{cfg.length}\n")
    lines.append(f"- **语言**：{cfg.language}\n")

    lines.append("## 协作过程记录（结构化消息摘要）\n")
    lines.append("> 说明：为保证可读性，这里展示消息的摘要；如需完整 JSON，可用 `--show-json` 运行并重定向输出。\n")
    lines.append("| 时间 | 发送方 | 类型 | 摘要 |\n|---|---|---|---|\n")
    for m in ctx:
        content = m.content
        if isinstance(content, dict):
            summary = json.dumps(content, ensure_ascii=False)
        else:
            summary = str(content)
        summary = summary.replace("\n", " ")
        if len(summary) > 120:
            summary = summary[:120] + "…"
        lines.append(f"| {m.ts} | {m.sender} | {m.msg_type} | {summary} |\n")

    lines.append("\n## 研究代理输出（Research Agent）\n")
    lines.append(md_json(research or {"error": "missing"}))

    lines.append("\n## 撰写代理输出（Writing Agent）\n")
    lines.append(md_json({k: v for k, v in (draft or {"error": "missing"}).items() if k != "content"}))
    lines.append("\n### 初稿正文\n")
    lines.append((draft or {}).get("content", "（缺失）"))

    lines.append("\n## 审核代理输出（Review Agent）\n")
    lines.append(md_json(review or {"error": "missing"}))

    lines.append("\n## 润色代理输出（Polishing Agent）\n")
    lines.append(md_json({k: v for k, v in (final_article or {"error": "missing"}).items() if k != "content"}))
    lines.append("\n### 最终成稿\n")
    lines.append((final_article or {}).get("content", "（缺失）"))

    lines.append("\n## 异常处理日志（三级重试）\n")
    if not exceptions:
        lines.append("- **无异常**：本次运行未触发重试。\n")
    else:
        lines.append("| 时间 | 代理 | 重试级别 | 尝试 | 错误类型 | 错误信息 |\n|---|---|---:|---:|---|---|\n")
        for e in exceptions:
            msg = e.error_message.replace("\n", " ")
            msg = msg if len(msg) <= 80 else msg[:80] + "…"
            lines.append(f"| {e.ts} | {e.agent} | {e.level} | {e.attempt} | {e.error_type} | {msg} |\n")
        lines.append("\n### 异常详情（Traceback）\n")
        for e in exceptions:
            if not e.traceback:
                continue
            lines.append(f"#### {e.ts} - {e.agent} (level {e.level}, attempt {e.attempt})\n")
            lines.append("```text\n" + e.traceback.strip() + "\n```\n")

    return "".join(lines).strip() + "\n"


# -----------------------------
# CLI
# -----------------------------


def parse_args(argv: List[str]) -> RunConfig:
    p = argparse.ArgumentParser(
        prog="multi-agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="基于类MCP协议的多代理文章自动编写系统（Research->Writing->Review->Polishing）。",
    )
    p.add_argument("question", nargs="?", default="", help="用户问题，例如：帮我写一篇关于AI Agent的文章")
    p.add_argument("--style", default="科普 / 技术入门", help="文章风格，例如：技术博客/科普/学术/营销")
    p.add_argument("--length", choices=["short", "medium", "long"], default="medium", help="文章长度档位")
    p.add_argument("--language", default="zh", help="语言：zh/en 等")
    p.add_argument("--out", default="", help="输出 report.md 路径（默认写入 multi-agent/report.md）")
    p.add_argument("--show-json", action="store_true", help="终端输出结构化消息（JSON）")
    p.add_argument("--quiet", action="store_true", help="安静模式（减少终端输出）")
    args = p.parse_args(argv)

    question = args.question.strip()
    if not question:
        if sys.stdin.isatty():
            question = input("请输入你的问题（例如：帮我写一篇关于AI Agent的文章）：\n> ").strip()
        else:
            question = "帮我写一篇关于AI Agent的文章"

    out = args.out.strip()
    if not out:
        # default next to this file
        here = os.path.dirname(os.path.abspath(__file__))
        out = os.path.join(here, "report.md")

    return RunConfig(
        question=question,
        style=args.style,
        length=args.length,
        language=args.language,
        output_path=out,
        show_json=bool(args.show_json),
        quiet=bool(args.quiet),
    )


def main(argv: Optional[List[str]] = None) -> None:
    if load_dotenv is not None:
        load_dotenv()

    cfg = parse_args(sys.argv[1:] if argv is None else argv)
    logger = ConsoleLogger(quiet=cfg.quiet, show_json=cfg.show_json)

    ctx: List[MCPMessage] = []
    exceptions: List[ExceptionRecord] = []

    logger.event("System", "多代理写作流程启动")
    ctx.append(MCPMessage(sender="User", msg_type="event", content={"question": cfg.question, "style": cfg.style, "length": cfg.length}))

    tool = SearchTool(enable_web=False)
    research_agent = ResearchAgent(tool)
    research_backup = ResearchAgentBackup(tool)
    writing_agent = WritingAgent()
    review_agent = ReviewAgent()
    review_senior = ReviewAgentSenior()
    polishing_agent = PolishingAgent()

    # 1) Research
    out = run_agent_with_retries(
        research_agent,
        ctx,
        cfg,
        logger,
        exceptions,
        backup_agent=research_backup,
        can_ask_user=True,
    )
    msg = MCPMessage(sender=research_agent.name if "research" in out else research_backup.name, msg_type="artifact", content=out)
    ctx.append(msg)
    logger.message(msg)

    # 2) Writing
    out = run_agent_with_retries(
        writing_agent,
        ctx,
        cfg,
        logger,
        exceptions,
        backup_agent=None,
        can_ask_user=True,
    )
    msg = MCPMessage(sender=writing_agent.name, msg_type="artifact", content=out)
    ctx.append(msg)
    logger.message(msg)

    # 3) Review
    out = run_agent_with_retries(
        review_agent,
        ctx,
        cfg,
        logger,
        exceptions,
        backup_agent=review_senior,
        can_ask_user=True,
    )
    msg = MCPMessage(sender=review_agent.name if "review" in out else review_senior.name, msg_type="artifact", content=out)
    ctx.append(msg)
    logger.message(msg)

    # 4) Polishing
    out = run_agent_with_retries(
        polishing_agent,
        ctx,
        cfg,
        logger,
        exceptions,
        backup_agent=None,
        can_ask_user=True,
    )
    msg = MCPMessage(sender=polishing_agent.name, msg_type="artifact", content=out)
    ctx.append(msg)
    logger.message(msg)

    # Write report
    report = render_report(cfg, ctx, exceptions)
    os.makedirs(os.path.dirname(os.path.abspath(cfg.output_path)), exist_ok=True)
    with open(cfg.output_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.event("System", f"已生成报告：{cfg.output_path}")

if __name__ == "__main__":
    main()