from __future__ import annotations

import os
import sys
from pathlib import Path


def _candidate_cad311_pythons() -> list[Path]:
    """返回可能的 cad311 Python 可执行文件路径（按优先级）。"""
    candidates: list[Path] = []

    # 1) 由当前解释器推断 Anaconda 根目录
    current_python = Path(sys.executable)
    try:
        anaconda_root = current_python.parent.parent
        candidates.append(anaconda_root / "envs" / "cad311" / "python.exe")
    except Exception:
        pass

    # 2) 用户常见安装路径
    user_profile = os.environ.get("USERPROFILE", "")
    if user_profile:
        candidates.append(Path(user_profile) / "anaconda3" / "envs" / "cad311" / "python.exe")

    # 去重并仅保留存在路径
    unique: list[Path] = []
    seen = set()
    for path in candidates:
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return [p for p in unique if p.exists()]


def _ensure_prefer_cad311() -> None:
    """
    若当前不在 cad311 环境，且本机存在 cad311 Python，则自动切换并重启当前脚本。
    """
    # 防止重复重启导致循环
    if os.environ.get("CAD311_BOOTSTRAPPED") == "1":
        return

    current_python = Path(sys.executable)
    if "envs\\cad311\\" in str(current_python).lower().replace("/", "\\"):
        return

    candidates = _candidate_cad311_pythons()
    if not candidates:
        print("提示：未找到 cad311 环境，继续使用当前 Python 运行。")
        return

    target_python = candidates[0]
    print(f"提示：检测到 cad311 环境，自动切换解释器：{target_python}")
    os.environ["CAD311_BOOTSTRAPPED"] = "1"
    os.execv(str(target_python), [str(target_python), *sys.argv])


def main():
    _ensure_prefer_cad311()
    from agent_graph import app
    from config import validate_llm_startup

    print("=== 欢迎使用工业级多 Agent 建模助手 ===")

    # 避免误把旧的 part.step 当作本轮输出
    try:
        step_path = Path("part.step")
        if step_path.exists():
            step_path.unlink()
    except Exception:
        pass

    initial_state = {
        "input": "设计一个中心距为150mm，大头孔径50mm，小头孔径20mm的发动机连杆。",
        "plan": "",
        "code": "",
        "critique": "",
        "history": [],
        "chat_history": [],
        "errors": "",
        "iterations": 0,
    }

    if app is None:
        print("❌ 工作流未初始化成功，请先安装并检查 langgraph 依赖。")
        return

    startup_issues = validate_llm_startup()
    if startup_issues:
        print("\n【启动前校验失败】")
        for issue in startup_issues:
            print(f"- {issue}")
        print("请修正 `.env` 后重试。")
        return

    final_output = app.invoke(initial_state)
    history = final_output.get("history", []) or []
    status = final_output.get("status", "")
    plan = final_output.get("plan", "") or ""
    code = final_output.get("code", "") or ""
    errors = final_output.get("errors", "") or ""
    metadata = final_output.get("metadata") or {}

    print("\n" + "=" * 30)
    print("【0. LLM 调用状态】")
    print(f"- status: {status}")
    print(f"- plan_len: {len(plan)}")
    print(f"- code_len: {len(code)}")
    if isinstance(metadata, dict) and metadata:
        part_type = metadata.get("part_type")
        confidence = metadata.get("confidence")
        missing = metadata.get("missing_params")
        blocking = metadata.get("blocking_missing_params")
        print(f"- part_type: {part_type}")
        print(f"- confidence: {confidence}")
        print(f"- missing_params: {missing}")
        print(f"- blocking_missing_params: {blocking}")

    print("【1. 最终计划】")
    print(plan)

    print("\n【2. 生成的代码预览】")
    if code:
        print(code[:200] + "...")
    else:
        print("(无代码输出)")

    print("\n【3. 执行结果】")
    if errors == "":
        if Path("part.step").exists():
            print("成功：part.step 文件已在本地生成！")
        else:
            print("执行未报错，但未发现 part.step（请检查 cad_tools 导出逻辑）。")
    else:
        print(f"失败：{errors}")
    print("\n【4. 历史轨迹】")
    print(history)
    print("=" * 30)


if __name__ == "__main__":
    main()
