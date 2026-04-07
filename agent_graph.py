from __future__ import annotations

import os
import traceback
import importlib
import re
import json
from pathlib import Path
import time
import random
from collections import deque
from typing import Any, TypedDict

from cad_tools import execute_cad_code
from config import create_llm
from planner_prompt import PLANNER_SYSTEM_PROMPT
from rag_context import retrieve_context

CODER_SYSTEM_PROMPT = """
你是一个精通 CadQuery 的高级脚本工程师。
你的任务是根据 [建模计划] 编写 Python 代码。

## 约束要求：
1. 必须使用 `import cadquery as cq`。
2. 必须将最终的几何对象赋值给变量 `result` (例如：result = ...)。
3. 使用参数化编程，将所有尺寸定义为变量放在代码开头。
4. 确保代码逻辑闭环，不要有未定义的引用。
5. 禁止使用 `cadquery.selectors.StringSelector` 或 `cq.selectors.StringSelector`。
6. 仅使用 CadQuery 通用选择器写法，例如：`faces(">Z")`、`edges("|Z")`、`workplane()`。
7. 禁止使用 `arcTo`（当前环境 Workplane 无此方法），需要圆弧时改用 `threePointArc` 或简化为线段。
"""


class AgentState(TypedDict, total=False):
    input: str
    plan: str
    history: list[str]
    chat_history: list[dict[str, str]]
    metadata: dict[str, Any]
    status: str
    rag_standards: str
    constraints: str
    code: str
    errors: str
    iterations: int
    cad_code: str
    cad_result: Any
    error: dict[str, Any] | None


# --- LLM 调用限流（进程内）---
_LLM_CALL_TIMES: "deque[float]" = deque()


def _rate_limit_per_minute() -> int:
    try:
        return int(os.getenv("LLM_RATE_LIMIT_PER_MIN", "15"))
    except Exception:
        return 15


def _llm_rate_limit_wait() -> None:
    """
    简易限流：确保近 60s 内 LLM 调用次数不超过阈值。
    适用于 Gemini 免费 key（例如 15 次/分钟）。
    """
    limit = _rate_limit_per_minute()
    if limit <= 0:
        return

    now = time.monotonic()
    window = 60.0
    # 清理窗口外记录
    while _LLM_CALL_TIMES and now - _LLM_CALL_TIMES[0] > window:
        _LLM_CALL_TIMES.popleft()

    if len(_LLM_CALL_TIMES) < limit:
        _LLM_CALL_TIMES.append(now)
        return

    # 需要等待到最早一次调用滑出窗口
    wait_for = window - (now - _LLM_CALL_TIMES[0]) + random.uniform(0.1, 0.5)
    if wait_for > 0:
        time.sleep(wait_for)
    # 记录本次
    _LLM_CALL_TIMES.append(time.monotonic())


def _retry_budget_seconds() -> float:
    """
    单次 invoke 的总重试时间预算（秒）。
    默认 300s（5min），可通过环境变量覆盖。
    """
    try:
        return float(os.getenv("LLM_RETRY_BUDGET_SEC", "300"))
    except Exception:
        return 300.0


def _extract_retry_delay_seconds(text: str) -> float | None:
    """
    从 Gemini/GenAI 错误文本里提取建议的 retry delay（秒）。
    覆盖两种常见格式：
    - "Please retry in 10.197563535s."
    - "'retryDelay': '10s'" 或 "retryDelay': '10s'"
    """
    if not text:
        return None

    m = re.search(r"Please retry in\s+([0-9]+(?:\.[0-9]+)?)s", text, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None

    m = re.search(r"retryDelay'\s*:\s*'(\d+)s'", text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None

    m = re.search(r"retryDelay\"\s*:\s*\"(\d+)s\"", text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None

    return None


def _smart_backoff_sleep(exc_text: str, attempt: int) -> None:
    """
    智能退避：
    - 若是 429/RESOURCE_EXHAUSTED 且包含 retry delay，则按建议 sleep
    - 否则按指数退避 sleep
    """
    lower = (exc_text or "").lower()
    # 免费层配额耗尽：不要在 UI 里无意义等待
    if _is_gemini_free_tier_quota_exhausted(exc_text):
        return
    is_rate_limited = ("resource_exhausted" in lower) or ("429" in lower)
    retry_delay = _extract_retry_delay_seconds(exc_text) if is_rate_limited else None

    if retry_delay is not None:
        # 轻微抖动，避免同一秒同时重试
        time.sleep(max(0.0, retry_delay) + random.uniform(0.1, 0.5))
        return

    # 默认指数退避（上限 8s）
    backoff = min(2 ** (attempt - 1), 8) + random.uniform(0.0, 0.5)
    time.sleep(backoff)


def _is_gemini_free_tier_quota_exhausted(exc_text: str) -> bool:
    """
    判断是否是“免费层配额已耗尽”一类的 429。
    这类错误等待 retryDelay 往往也无效（可能是按日/按项目配额用尽），应立即降级而不是继续重试/睡眠。
    """
    lower = (exc_text or "").lower()
    return (
        ("resource_exhausted" in lower or "429" in lower)
        and (
            "generaterequestsperdayperprojectpermodel-freetier" in lower
            or "generate_content_free_tier_requests" in lower
            or "free_tier_requests" in lower
        )
        and "quota exceeded" in lower
    )


PARSER_SYSTEM_PROMPT = """
你是一个工业 CAD 多 Agent 系统的“输入解析器（Parser Agent）”。
你的任务是：根据用户的自然语言对话，识别要建模的零件类型（part_type），提取尺寸参数与约束（extracted_params/constraints），并列出缺失参数（missing_params）与需要向用户追问的澄清问题（clarifying_questions）。

## 要求
1) 只输出严格 JSON，不要 Markdown，不要解释，不要代码块。
2) part_type 必须是以下之一：connecting_rod, flange, bracket, shaft, gear, unknown
3) 尺寸单位默认 mm；如果用户明确说明，则按用户为准。
4) 对缺失但常见的参数可以给“建议默认值”（放在 suggested_defaults），但仍应把它列入 missing_params 并提出确认问题。
5) clarifying_questions 必须像人一样“逐条提问”，每个问题都要明确告诉用户需要提供哪个参数（最好包含参数名+示例）。
6) 如果缺失参数很多，优先追问最关键的 3~6 个（能闭环建模的最小集合）。

## 输出 JSON schema
{
  "part_type": "connecting_rod|flange|bracket|shaft|gear|unknown",
  "intent_summary": "一句话总结用户想建什么",
  "extracted_params": { "param_name": value, ... },
  "constraints": { "material": "...", "process": "...", "tolerance": "...", "standards": ["..."], ... },
  "suggested_defaults": { "param_name": value, ... },
  "missing_params": ["..."],
  "clarifying_questions": ["..."],
  "confidence": 0.0
}
""".strip()


def _format_chat_history(chat_history: list[dict[str, str]] | None, max_turns: int = 12) -> str:
    if not chat_history:
        return ""
    trimmed = chat_history[-max_turns:]
    lines: list[str] = []
    for m in trimmed:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()


def _humanize_missing_param_questions(part_type: str, missing_params: list[str]) -> list[str]:
    """
    在模型未给出高质量追问时，根据 part_type + missing_params 生成“像人一样”的提问。
    """
    pt = (part_type or "unknown").strip()
    missing = [m for m in (missing_params or []) if str(m).strip()]

    # 通用问题模板（带示例）
    generic_map = {
        "thickness": "请确认厚度 `thickness` 是多少？例如：10mm。",
        "outer_diameter": "请确认外径 `outer_diameter` 是多少？例如：120mm。",
        "inner_diameter": "请确认内径/孔径 `inner_diameter` 是多少？例如：50mm。",
        "height": "请确认高度 `height` 是多少？例如：30mm。",
        "width": "请确认宽度 `width` 是多少？例如：40mm。",
        "length": "请确认长度 `length` 是多少？例如：150mm。",
        "hole_diameter": "请确认孔径 `hole_diameter` 是多少？例如：8.5mm。",
        "hole_count": "请确认孔数量 `hole_count` 是多少？例如：6。",
        "pcd": "请确认孔分布圆直径 `pcd` 是多少？例如：80mm。",
    }

    # 零件类型的关键参数优先级（用于截断到 3~6 个）
    priority_by_type: dict[str, list[str]] = {
        "connecting_rod": [
            "center_distance",
            "big_end_diameter",
            "small_end_diameter",
            "rod_thickness",
            "big_end_outer_diameter",
            "small_end_outer_diameter",
        ],
        "flange": ["outer_diameter", "inner_diameter", "thickness", "pcd", "hole_count", "hole_diameter"],
        "shaft": ["length", "outer_diameter", "inner_diameter"],
        "bracket": ["length", "width", "height", "thickness", "hole_count", "hole_diameter"],
        "gear": ["module", "teeth", "thickness", "bore_diameter"],
        "unknown": [],
    }

    # 先按优先级挑选缺失项
    prioritized = []
    for p in priority_by_type.get(pt, []):
        if p in missing:
            prioritized.append(p)

    # 再补齐剩余缺失项
    for m in missing:
        if m not in prioritized:
            prioritized.append(m)

    # 截断到 6 个问题，避免轰炸用户
    prioritized = prioritized[:6]

    questions: list[str] = []
    for m in prioritized:
        # 适配一些常见字段别名
        if m in generic_map:
            questions.append(generic_map[m])
            continue

        if m == "center_distance":
            questions.append("请确认连杆中心距 `center_distance` 是多少？例如：150mm。")
        elif m == "big_end_diameter":
            questions.append("请确认连杆大头孔径 `big_end_diameter` 是多少？例如：50mm。")
        elif m == "small_end_diameter":
            questions.append("请确认连杆小头孔径 `small_end_diameter` 是多少？例如：20mm。")
        elif m == "rod_thickness":
            questions.append("请确认连杆厚度 `rod_thickness` 是多少？例如：15mm。")
        elif m == "big_end_outer_diameter":
            questions.append("请确认连杆大头外径 `big_end_outer_diameter` 是多少？例如：80mm。")
        elif m == "small_end_outer_diameter":
            questions.append("请确认连杆小头外径 `small_end_outer_diameter` 是多少？例如：40mm。")
        else:
            questions.append(f"请补充参数 `{m}` 的数值（单位 mm），例如：`{m}=10`。")

    # 如果缺失列表为空但仍 need_clarification（低置信度），给一个更自然的兜底
    if not questions:
        questions = [
            "你希望建模的具体零件是什么（例如：连杆、法兰盘、支架）？",
            "请给出 2~3 个最关键尺寸（单位 mm），例如：外径、孔径、厚度/高度等。",
        ]
    return questions

def call_input_parser(state: AgentState) -> AgentState:
    """
    输入解析节点：意图识别 + 槽位填充，写入 state["metadata"]。
    若需要澄清问题，则设置 status="need_clarification"。
    """
    user_input = (state.get("input") or "").strip()
    history = state.get("history", [])
    chat_history = state.get("chat_history", [])

    if not user_input and not chat_history:
        return {
            **state,
            "metadata": {
                "part_type": "unknown",
                "intent_summary": "",
                "extracted_params": {},
                "constraints": {},
                "suggested_defaults": {},
                "missing_params": ["input"],
                "clarifying_questions": ["请描述你想建模的零件类型与关键尺寸（单位 mm）。"],
                "confidence": 0.0,
            },
            "status": "need_clarification",
            "history": [*history, "Parser 失败：输入为空"],
        }

    use_real_llm = os.getenv("USE_REAL_LLM", "0") == "1"
    if not use_real_llm:
        # 最简 mock：把整个输入当作 unknown
        return {
            **state,
            "metadata": {
                "part_type": "unknown",
                "intent_summary": user_input[:80],
                "extracted_params": {},
                "constraints": {},
                "suggested_defaults": {},
                "missing_params": [],
                "clarifying_questions": [],
                "confidence": 0.1,
            },
            "status": "parsed",
            "history": [*history, "Parser 已解析（mock）"],
        }

    transcript = _format_chat_history(chat_history)
    parser_user_prompt = f"""
[对话上下文]
{transcript if transcript else "(无)"}

[本轮用户输入]
{user_input}
""".strip()

    last_tb = ""
    start_ts = time.monotonic()
    max_parser_attempts = int(os.getenv("PARSER_MAX_ATTEMPTS", "2"))
    for attempt in range(1, max_parser_attempts + 1):
        try:
            _llm_rate_limit_wait()
            messages_module = importlib.import_module("langchain_core.messages")
            SystemMessage = getattr(messages_module, "SystemMessage")
            HumanMessage = getattr(messages_module, "HumanMessage")
            llm = create_llm()
            resp = llm.invoke(
                [
                    SystemMessage(content=PARSER_SYSTEM_PROMPT),
                    HumanMessage(content=parser_user_prompt),
                ]
            )
            text = (str(resp.content) or "").strip()
            json_text = _extract_json_text(text)
            data = json.loads(json_text)

            confidence = float(data.get("confidence") or 0.0)
            missing = data.get("missing_params") or []
            clarifying = data.get("clarifying_questions") or []
            suggested = data.get("suggested_defaults") or {}
            if not isinstance(suggested, dict):
                suggested = {}

            # 如果缺失参数都已给出 suggested_defaults，则认为“可继续建模”，不阻塞流程
            # 另外，一些“非几何关键”的约束（材料/公差/工艺等）可在后续迭代再确认，这里不阻塞生成。
            optional_missing = {
                "material",
                "process",
                "tolerance",
                "standard",
                "standards",
                "surface_finish",
            }
            blocking_missing = [
                str(m)
                for m in missing
                if (str(m) not in suggested) and (str(m) not in optional_missing)
            ]
            data["blocking_missing_params"] = blocking_missing

            # PRD：confidence < 0.8 或 missing_params 非空 -> 立即中断追问
            need_clarification = (confidence < 0.8) or (len(blocking_missing) > 0)
            # 若需要追问但模型未给问题，或问题太笼统，则生成更“像人一样”的提问
            if need_clarification:
                # 判定“太笼统”的条件：只有 1 条且很短
                too_generic = (len(clarifying) <= 1) and all(len(str(q)) < 18 for q in clarifying)
                if (not clarifying) or too_generic:
                    clarifying = _humanize_missing_param_questions(
                        str(data.get("part_type") or "unknown"), [str(x) for x in blocking_missing]
                    )
                    data["clarifying_questions"] = clarifying

            status = "need_clarification" if need_clarification else "parsed"
            return {
                **state,
                "metadata": data,
                "status": status,
                "errors": "",
                "history": [*history, f"Parser 已解析（attempt {attempt}）"],
            }
        except Exception:
            last_tb = traceback.format_exc()
            if _is_gemini_free_tier_quota_exhausted(last_tb):
                break
            if time.monotonic() - start_ts > _retry_budget_seconds():
                break
            _smart_backoff_sleep(last_tb, attempt)

    # LLM 失败：回退到规则解析，依然给出“像人一样”的追问（确保交互可用）
    # 当前先做最小规则：识别关键词决定 part_type；不做复杂数值抽取
    part_type = "unknown"
    if "连杆" in user_input:
        part_type = "connecting_rod"
    elif "法兰" in user_input or "法兰盘" in user_input:
        part_type = "flange"
    elif "轴" in user_input:
        part_type = "shaft"
    elif "齿轮" in user_input:
        part_type = "gear"
    elif "支架" in user_input:
        part_type = "bracket"

    # 缺参：按类型给关键参数列表，用于生成具体追问
    missing_params = {
        "connecting_rod": ["center_distance", "big_end_diameter", "small_end_diameter", "rod_thickness"],
        "flange": ["outer_diameter", "inner_diameter", "thickness", "pcd", "hole_count", "hole_diameter"],
        "shaft": ["length", "outer_diameter"],
        "bracket": ["length", "width", "height", "thickness"],
        "gear": ["module", "teeth", "thickness", "bore_diameter"],
        "unknown": [],
    }.get(part_type, [])

    return {
        **state,
        "metadata": {
            "part_type": part_type,
            "intent_summary": user_input[:80],
            "extracted_params": {},
            "constraints": {},
            "suggested_defaults": {},
            "missing_params": missing_params,
            "clarifying_questions": _humanize_missing_param_questions(part_type, missing_params),
            "confidence": 0.0,
        },
        "status": "need_clarification",
        "errors": last_tb,
        "history": [*history, "Parser 失败：LLM 调用异常（已回退规则解析）"],
    }


def execute_cad_code_node(state: AgentState) -> AgentState:
    """
    执行 CAD 代码并将结果或错误写回 State。

    约定:
    - 输入: state["cad_code"]
    - 成功: 写入 state["cad_result"], 清空 state["error"]
    - 失败: 保留原始 state, 写入结构化错误 state["error"]
    """
    cad_code = state.get("cad_code", "")
    if not cad_code.strip():
        return {
            **state,
            "error": {
                "type": "ValueError",
                "message": "cad_code 为空，无法执行 CAD 代码。",
                "traceback": None,
            },
        }

    try:
        result = execute_cad_code(cad_code)
        # execute_cad_code 会将大多数执行失败以字符串形式返回
        if isinstance(result, str) and result.startswith("Error"):
            return {
                **state,
                "cad_result": None,
                "error": {
                    "type": "CadExecutionError",
                    "message": result,
                    "traceback": None,
                },
            }

        return {
            **state,
            "cad_result": result,
            "error": None,
        }
    except Exception as exc:
        return {
            **state,
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }


def call_cad_executor(state: AgentState) -> AgentState:
    """
    调用 CAD 执行工具运行 state["code"]，并将错误写入 state["errors"]。

    约定:
    - 输入: state["code"]
    - 成功: state["errors"] 置为空字符串
    - 失败: state["errors"] 写入 traceback 字符串
    """
    code = state.get("code", "")
    iterations = state.get("iterations", 0)
    if not code.strip():
        try:
            raise ValueError("state['code'] 为空，无法执行 CAD 代码。")
        except Exception:
            return {
                **state,
                "errors": traceback.format_exc(),
                "iterations": iterations + 1,
            }

    try:
        result = execute_cad_code(code)
        # execute_cad_code 可能将执行错误包装为字符串返回
        if isinstance(result, str) and result.startswith("Error"):
            raise RuntimeError(result)

        return {
            **state,
            "errors": "",
        }
    except Exception:
        return {
            **state,
            "errors": traceback.format_exc(),
            "iterations": iterations + 1,
        }


def _mock_planner_response(user_input: str, rag_standards: str) -> str:
    """在无可用 LLM 时返回稳定的 Mock 计划文本。"""
    return (
        "### 1) 需求理解\n"
        f"- 功能目标: 基于输入生成连杆建模方案（需求：{user_input}）\n"
        "- 关键尺寸/范围: 待由用户或标准补全\n"
        "- 材料与工艺: 默认钢件机加工\n"
        "- 装配与受力约束: 关注大小头孔与中心距\n\n"
        "### 2) 国标约束映射（基于 RAG）\n"
        f"- 条款A: {rag_standards} -> 影响大头厚度与孔径比例 -> 满足(待参数确认)\n\n"
        "### 3) 参数表（用于 CadQuery）\n"
        "- `big_end_diameter`: 待定\n"
        "- `small_end_diameter`: 待定\n"
        "- `center_distance`: 待定\n"
        "- `rod_thickness`: 待定\n"
        "- `fillet_radius`: 待定\n\n"
        "### 4) CadQuery 建模步骤（可执行级）\n"
        "1. 创建 XY 工作平面并定义参数。\n"
        "2. 绘制大小头与连杆身过渡轮廓。\n"
        "3. 拉伸形成基体。\n"
        "4. 切除大小头孔。\n"
        "5. 添加圆角与倒角并复核约束。\n\n"
        "### 5) 风险与待确认\n"
        "- 风险: 关键尺寸缺失会影响最终可制造性。\n"
        "- 假设: 默认单位 mm。\n"
        "- 待用户确认问题: 目标功率级别、材料牌号、公差等级。\n\n"
        "### 6) 交付给 Code Agent 的指令草案\n"
        "- 建模顺序: 先基体后特征再修饰。\n"
        "- 必要参数: 孔径、中心距、厚度、圆角半径。\n"
        "- 禁止事项: 禁止硬编码、禁止忽略标准约束。\n"
    )


def _extract_python_code(text: str) -> str:
    """
    从 LLM 返回文本中提取 Python 代码块。
    若无代码块，则回退为原始文本。
    """
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_json_text(text: str) -> str:
    """
    从 LLM 返回中尽量提取 JSON 文本：
    - 优先提取 ```json ... ```
    - 否则取第一个 '{' 到最后一个 '}' 的子串
    """
    raw = (text or "").strip()
    if not raw:
        return ""

    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1].strip()
    return raw


def _safe_py_value(v: Any) -> str:
    """把常见 JSON 值转成稳定的 Python 字面量字符串。"""
    if isinstance(v, bool):
        return "True" if v else "False"
    if v is None:
        return "None"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return repr(v)
    # 兜底：复杂结构用 json.dumps，保证可执行
    try:
        return repr(json.dumps(v, ensure_ascii=False))
    except Exception:
        return repr(str(v))


def _render_param_block(params: dict[str, Any]) -> str:
    """
    把参数 dict 渲染成 Python 赋值块，确保 Coder 能“照抄”进代码顶部。
    """
    if not params:
        return ""
    lines = ["# ========== 参数（来自 Parser 提取/建议默认值） =========="]
    for k in sorted(params.keys()):
        key = str(k).strip()
        if not key:
            continue
        # 仅允许合法标识符
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            continue
        lines.append(f"{key} = {_safe_py_value(params[k])}")
    return "\n".join(lines).strip()

def _get_cadquery_version() -> str:
    """读取本地 cadquery 版本，失败时返回 unknown。"""
    try:
        import cadquery as cq  # pyright: ignore[reportMissingImports]

        return str(getattr(cq, "__version__", "unknown"))
    except Exception:
        return "unknown"


def _check_generated_cad_code(code_text: str) -> str | None:
    """
    对生成代码做静态校验，命中禁用模式时返回错误原因，否则返回 None。
    """
    banned_patterns = [
        "cadquery.selectors.StringSelector",
        "cq.selectors.StringSelector",
        ".arcTo(",
    ]
    for pattern in banned_patterns:
        if pattern in code_text:
            return f"检测到禁用 API: {pattern}"

    if "import cadquery as cq" not in code_text:
        return "缺少必要导入: import cadquery as cq"
    if "result" not in code_text:
        return "缺少最终输出变量: result"
    return None


def _allow_mock_fallback() -> bool:
    """
    是否允许在真实 LLM 失败时回退到 mock 代码。
    - 1/true/yes: 允许
    - 0/false/no: 禁止（默认，便于测试根因）
    """
    return os.getenv("ALLOW_MOCK_FALLBACK", "0").strip().lower() in {"1", "true", "yes"}


def _load_memory_rag_notes(max_chars: int = 12000) -> str:
    """
    轻量内存 RAG：直接加载 docs/rag 下所有 markdown 内容。
    用于在 prompt 中注入“强制约束”。
    """
    rag_dir = Path(__file__).parent / "docs" / "rag"
    if not rag_dir.exists():
        return ""

    blocks: list[str] = []
    for md in sorted(rag_dir.glob("*.md")):
        try:
            text = md.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if text:
            blocks.append(f"## Source: {md.name}\n{text}")

    merged = "\n\n".join(blocks).strip()
    if len(merged) > max_chars:
        return merged[:max_chars] + "\n\n[Truncated]"
    return merged


def _error_to_fix_strategy(error_text: str) -> str:
    """
    执行错误 -> 修复策略映射，供 Coder 重试时使用。
    """
    e = error_text.lower()
    strategies: list[str] = []

    if "stringselector" in e:
        strategies.append(
            "- 禁止 selectors.StringSelector；改为 `faces(\">Z\")`/`edges(\"|Z\")` 等字符串选择器。"
        )
    if "arcTo".lower() in e:
        strategies.append("- 禁止 `arcTo`；改用 `threePointArc` 或线段 + 圆角。")
    if "no attribute" in e and "workplane" in e:
        strategies.append("- 仅使用 CadQuery 2.7 常见 Workplane API，避免可疑方法。")
    if "can not return the nth element of an empty list" in e:
        strategies.append("- 说明选择器结果为空；所有 fillet/chamfer 前先确保选择范围不为空。")
    if "cannot find a solid on the stack" in e or "at least one solid on the stack" in e:
        strategies.append("- 布尔运算前确保对象是 solid；先构建基体，再对基体 cut/union。")
    if "brep_api: command not done" in e:
        strategies.append("- OCC 布尔失败；简化轮廓、减少复杂布尔、分步构建并逐步 cut。")
    if ("brepadaptor_curve" in e and "no geometry" in e) or ("no geometry" in e and "brep" in e):
        strategies.append(
            "- OCC 曲线/边退化（No geometry）：确保草图轮廓闭合且无零长度边/重复点；优先用稳健基元（`rect`/`slot2D`/`polyline(...).close()`）再 `extrude`；避免在同一草图里混用多段 moveTo/lineTo 导致自交或断裂。"
        )
    if "no pending wires present" in e:
        strategies.append(
            "- 未生成可挤出的 wire（No pending wires）：在 `extrude` 前确保当前工作平面上有闭合轮廓；若使用 `polyline/lineTo`，务必 `.close()` 或回到起点；避免 `circle()` 与开放折线混在同一链条里导致 wire 不成立。"
        )
    if "name '" in e and "' is not defined" in e:
        strategies.append(
            "- NameError（变量未定义）：所有用到的参数必须在代码顶部显式赋值；优先从 Parser 的 `extracted_params/suggested_defaults` 读取并生成同名变量（例如 `bolt_hole_count`）。"
        )

    if not strategies:
        strategies.append("- 基于报错做最小改动修复，保持参数化与 `result` 输出约束。")
    return "\n".join(strategies)


def call_planner(state: AgentState) -> AgentState:
    """
    Planner 节点：根据用户需求生成建模计划。
    统一通过 create_llm() 创建模型，不可用时自动回退到 Mock。
    """
    user_input = state.get("input", "").strip()
    history = state.get("history", [])
    iterations = state.get("iterations", 0)
    rag_standards = state.get(
        "rag_standards", "[这里暂时填入连杆大头厚度标准：厚度应为孔径的0.5-0.7倍]"
    )
    constraints = state.get("constraints", "")
    metadata = state.get("metadata") or {}

    if not user_input:
        try:
            raise ValueError("state['input'] 为空，Planner 无法生成计划。")
        except Exception:
            return {
                **state,
                "plan": "",
                "errors": traceback.format_exc(),
                "history": [*history, "Planner 失败：输入为空"],
            }

    user_requirement = (
        "【原始用户输入】\n"
        + user_input
        + "\n\n【解析结果（metadata）】\n"
        + json.dumps(metadata, ensure_ascii=False, indent=2)
    )

    prompt = PLANNER_SYSTEM_PROMPT.format(
        user_requirement=user_requirement,
        rag_standards=rag_standards,
        history="\n".join(history),
        constraints=constraints,
    )

    # 兼容无 Key 场景：默认 mock；设置 USE_REAL_LLM=1 时启用真实调用。
    use_real_llm = os.getenv("USE_REAL_LLM", "0") == "1"
    if not use_real_llm:
        return {
            **state,
            "plan": _mock_planner_response(user_input, rag_standards),
            "errors": "",
            "history": [*history, "Planner 已生成建模计划（mock）"],
        }

    # LLM 短重试，处理偶发网络/限流抖动
    max_planner_attempts = int(os.getenv("PLANNER_MAX_ATTEMPTS", "2"))
    last_tb = ""
    start_ts = time.monotonic()
    for attempt in range(1, max_planner_attempts + 1):
        try:
            _llm_rate_limit_wait()
            messages_module = importlib.import_module("langchain_core.messages")
            SystemMessage = getattr(messages_module, "SystemMessage")
            HumanMessage = getattr(messages_module, "HumanMessage")
            llm = create_llm()
            response = llm.invoke(
                [
                    SystemMessage(content=prompt),
                    HumanMessage(content=user_input),
                ]
            )
            return {
                **state,
                "plan": str(response.content),
                "errors": "",
                "history": [*history, f"Planner 已生成建模计划（attempt {attempt}）"],
            }
        except Exception:
            last_tb = traceback.format_exc()
            if _is_gemini_free_tier_quota_exhausted(last_tb):
                break
            if time.monotonic() - start_ts > _retry_budget_seconds():
                break
            _smart_backoff_sleep(last_tb, attempt)

    return {
        **state,
        "plan": "",
        "errors": last_tb,
        "iterations": iterations + 1,
        "history": [*history, "Planner 失败：LLM 调用异常"],
    }


def call_coder(state: AgentState) -> AgentState:
    """
    Code Agent 节点：将建模计划转译为可执行 CadQuery 代码。

    统一通过 create_llm() 创建模型，不可用时自动回退到 Mock。
    """
    history = state.get("history", [])
    plan = state.get("plan", "").strip()
    user_input = state.get("input", "").strip()
    cadquery_version = _get_cadquery_version()
    previous_error = state.get("errors", "").strip()
    retry_index = state.get("iterations", 0)
    allow_mock_fallback = _allow_mock_fallback()
    rag_query = f"{user_input}\n{plan}\ncadquery {cadquery_version} compatibility"
    rag_context = retrieve_context(rag_query, top_k=4)
    memory_rag_notes = _load_memory_rag_notes()
    fix_strategy = _error_to_fix_strategy(previous_error) if previous_error else "(无)"
    metadata = state.get("metadata") or {}
    extracted_params = {}
    suggested_defaults = {}
    try:
        if isinstance(metadata, dict):
            extracted_params = metadata.get("extracted_params") or {}
            suggested_defaults = metadata.get("suggested_defaults") or {}
    except Exception:
        extracted_params = {}
        suggested_defaults = {}
    if not isinstance(extracted_params, dict):
        extracted_params = {}
    if not isinstance(suggested_defaults, dict):
        suggested_defaults = {}

    # 参数合并：建议默认值作为底，再用提取值覆盖
    merged_params: dict[str, Any] = {**suggested_defaults, **extracted_params}
    forced_param_block = _render_param_block(merged_params)

    mock_code = """
import cadquery as cq

# 参数定义
big_end_dist = 150.0
big_hole_dia = 50.0
small_hole_dia = 20.0
thickness = 15.0

# 建模开始
# 1. 创建连杆主体（简化版：两个圆柱加一个连杆身）
big_end = cq.Workplane("XY").circle(big_hole_dia/2 + 10).extrude(thickness)
small_end = cq.Workplane("XY").center(big_end_dist, 0).circle(small_hole_dia/2 + 8).extrude(thickness)

# 2. 布尔运算结合并打孔
result = big_end.union(small_end).faces(">Z").workplane().circle(big_hole_dia/2).cutThruAll()
result = result.faces(">Z").workplane().center(big_end_dist, 0).circle(small_hole_dia/2).cutThruAll()
""".strip()

    # 兼容无 Key 场景：默认 mock；设置 USE_REAL_LLM=1 时启用真实调用。
    use_real_llm = os.getenv("USE_REAL_LLM", "0") == "1"
    if not use_real_llm:
        return {
            **state,
            "code": mock_code,
            "errors": "",
            "history": [*history, "Code Agent 已生成建模脚本（mock）"],
        }

    if not plan:
        if not allow_mock_fallback:
            return {
                **state,
                "code": state.get("code", ""),
                "errors": "Code Agent 无法生成代码：plan 为空，且已禁用 mock 回退。",
                "history": [*history, "Code Agent 失败：plan 为空（未回退 mock）"],
            }
        return {
            **state,
            "code": mock_code,
            "errors": "",
            "history": [*history, "Code Agent 回退 mock：plan 为空"],
        }

    coder_user_prompt = f"""
[用户需求]
{user_input}

[建模计划]
{plan}

[Parser 结构化参数（必须体现在代码开头的变量定义中）]
{json.dumps(merged_params, ensure_ascii=False, indent=2)}

[必须原样包含的参数赋值块（请直接粘贴到代码顶部）]
{forced_param_block if forced_param_block else "(无)"}

[运行环境]
- cadquery 版本: {cadquery_version}
- 注意：当前环境不支持 `cadquery.selectors.StringSelector`，请勿使用。

[RAG 检索片段]
{rag_context}

[内存 RAG 强制约束（必须遵守）]
{memory_rag_notes}

[上一轮执行错误（如有）]
{previous_error if previous_error else "(无)"}

[错误修复策略映射]
{fix_strategy}

请输出可直接执行的 CadQuery Python 代码，遵守系统约束。
仅返回代码（可使用 ```python 包裹）。
""".strip()

    max_coder_attempts = int(os.getenv("CODER_MAX_ATTEMPTS", "2"))
    last_tb = ""
    start_ts = time.monotonic()
    for attempt in range(1, max_coder_attempts + 1):
        try:
            _llm_rate_limit_wait()
            messages_module = importlib.import_module("langchain_core.messages")
            SystemMessage = getattr(messages_module, "SystemMessage")
            HumanMessage = getattr(messages_module, "HumanMessage")
            llm = create_llm()
            response = llm.invoke(
                [
                    SystemMessage(content=CODER_SYSTEM_PROMPT),
                    HumanMessage(content=coder_user_prompt),
                ]
            )
            code_text = _extract_python_code(str(response.content))
            if not code_text:
                raise ValueError("Coder 返回空代码。")

            check_error = _check_generated_cad_code(code_text)
            if check_error:
                # 触发一次“定向修复重试”
                repair_prompt = f"""
你的上一个版本未通过静态校验：{check_error}

请基于原需求和建模计划重新生成完整 CadQuery 代码，必须：
1) 保留 `import cadquery as cq`
2) 最终变量为 `result`
3) 禁止使用 `cadquery.selectors.StringSelector` / `cq.selectors.StringSelector`
4) 禁止使用 `arcTo`，圆弧请改为 `threePointArc` 或线段
5) 使用 `faces(">Z")` / `edges("|Z")` 等通用选择器写法
6) 兼容 cadquery {cadquery_version}
7) 必须遵守 [内存 RAG 强制约束]
8) 若上一轮执行错误不为空，必须应用 [错误修复策略映射]

仅返回 Python 代码。
""".strip()
                _llm_rate_limit_wait()
                repaired = llm.invoke(
                    [
                        SystemMessage(content=CODER_SYSTEM_PROMPT),
                        HumanMessage(content=coder_user_prompt),
                        HumanMessage(content=repair_prompt),
                    ]
                )
                code_text = _extract_python_code(str(repaired.content))
                second_check_error = _check_generated_cad_code(code_text)
                if second_check_error:
                    raise ValueError(f"Coder 代码静态校验失败: {second_check_error}")

            return {
                **state,
                "code": code_text,
                "errors": "",
                "history": [
                    *history,
                    (
                        f"Code Agent 已生成建模脚本（重试轮次: {retry_index}，attempt {attempt}）"
                        if retry_index > 0
                        else f"Code Agent 已生成建模脚本（attempt {attempt}）"
                    ),
                ],
            }
        except Exception:
            last_tb = traceback.format_exc()
            if _is_gemini_free_tier_quota_exhausted(last_tb):
                break
            if time.monotonic() - start_ts > _retry_budget_seconds():
                break
            _smart_backoff_sleep(last_tb, attempt)

    # 多次尝试仍失败：交给图的重试环路（优先于 mock）
    if not allow_mock_fallback:
        return {
            **state,
            "code": state.get("code", ""),
            "errors": last_tb,
            "history": [*history, "Code Agent 失败：LLM 调用异常（未回退 mock）"],
        }
    return {
        **state,
        "code": mock_code,
        "errors": last_tb,
        "history": [*history, "Code Agent 失败：LLM 调用异常，已回退 mock"],
    }


def decide_to_retry(state: AgentState) -> str:
    """
    执行后逻辑闸门：
    - 有错误且 iterations < 3: retry
    - 有错误且 iterations >= 3: fail
    - 无错误: complete
    """
    errors = state.get("errors", "")
    iterations = state.get("iterations", 0)

    if errors and iterations < 3:
        print(f"[Retry Gate] 检测到错误，准备第 {iterations + 1} 次重试。")
        return "retry"
    if errors:
        print("[Retry Gate] 达到最大重试次数，任务失败。")
        return "fail"
    print("[Retry Gate] 校验通过，准备交付。")
    return "complete"


def decide_after_planner(state: AgentState) -> str:
    """
    Planner 后逻辑闸门：
    - 有 plan: to_coder
    - 无 plan 且 iterations < 3: retry_planner
    - 无 plan 且 iterations >= 3: fail
    """
    plan = (state.get("plan") or "").strip()
    errors = state.get("errors", "")
    iterations = state.get("iterations", 0)

    if plan:
        return "to_coder"
    if errors and iterations < 3:
        print(f"[Planner Gate] Planner 失败，准备第 {iterations + 1} 次重试。")
        return "retry_planner"
    print("[Planner Gate] Planner 达到最大重试次数，任务失败。")
    return "fail"


def decide_after_coder(state: AgentState) -> str:
    """
    Coder 后逻辑闸门：
    - 有 code 且无 errors: to_executor
    - 有 errors 且 iterations < 3: retry_coder
    - 有 errors 且 iterations >= 3: fail
    """
    code = (state.get("code") or "").strip()
    errors = state.get("errors", "")
    iterations = state.get("iterations", 0)

    if code and not errors:
        return "to_executor"
    if errors and iterations < 3:
        print(f"[Coder Gate] Coder 失败，准备第 {iterations + 1} 次重试。")
        return "retry_coder"
    print("[Coder Gate] Coder 达到最大重试次数，任务失败。")
    return "fail"


def decide_after_parser(state: AgentState) -> str:
    """
    Parser 后逻辑闸门：
    - need_clarification: complete（交给前端展示追问）
    - parsed: to_planner
    """
    status = (state.get("status") or "").strip()
    if status == "need_clarification":
        return "complete"
    return "to_planner"


def build_workflow():
    """
    构建 LangGraph 工作流：
    planner -> coder -> executor -> END
    """
    graph_module = importlib.import_module("langgraph.graph")
    StateGraph = getattr(graph_module, "StateGraph")
    END = getattr(graph_module, "END")

    workflow = StateGraph(AgentState)
    workflow.add_node("parser", call_input_parser)
    workflow.add_node("planner", call_planner)
    workflow.add_node("coder", call_coder)
    workflow.add_node("executor", call_cad_executor)
    workflow.set_entry_point("parser")
    workflow.add_conditional_edges(
        "parser",
        decide_after_parser,
        {
            "to_planner": "planner",
            "complete": END,
        },
    )
    workflow.add_conditional_edges(
        "planner",
        decide_after_planner,
        {
            "to_coder": "coder",
            "retry_planner": "planner",
            "fail": END,
        },
    )
    workflow.add_conditional_edges(
        "coder",
        decide_after_coder,
        {
            "to_executor": "executor",
            "retry_coder": "coder",
            "fail": END,
        },
    )
    workflow.add_conditional_edges(
        "executor",
        decide_to_retry,
        {
            "retry": "coder",
            "complete": END,
            "fail": END,
        },
    )
    return workflow.compile()


try:
    app = build_workflow()
except Exception:
    # 允许在未安装 langgraph 时导入本模块，便于先开发其余节点逻辑。
    app = None
