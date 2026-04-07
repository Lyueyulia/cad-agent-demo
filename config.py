from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Provider = Literal["gemini", "openai", "deepseek", "qwen"]


def _load_dotenv_if_available() -> None:
    """
    自动加载项目根目录 `.env` 文件（如果存在）。

    优先级策略：
    - 使用 override=True，让 `.env` 覆盖系统环境变量
    - 因此：`.env` 变量 > 系统变量
    """
    dotenv_path = Path(__file__).with_name(".env")
    if not dotenv_path.exists():
        return

    try:
        dotenv_module = importlib.import_module("dotenv")
        load_dotenv = getattr(dotenv_module, "load_dotenv")
        load_dotenv(dotenv_path=dotenv_path, override=True)
    except Exception:
        # 无 python-dotenv 或加载失败时静默跳过，保持兼容性
        return


_load_dotenv_if_available()


@dataclass(frozen=True)
class LLMConfig:
    """
    LLM 统一配置（通过环境变量覆盖）。

    通用:
    - LLM_PROVIDER: gemini/openai/deepseek/qwen（默认 gemini）
    - LLM_MODEL:    模型名（按 provider 给默认值）
    - LLM_TEMPERATURE: 采样温度（默认 0）
    - LLM_MAX_TOKENS:  可选

    Provider 专用:
    - Gemini: GOOGLE_API_KEY
    - OpenAI: OPENAI_API_KEY
    - DeepSeek: DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL(可选，默认官方)
    - Qwen(通义千问): DASHSCOPE_API_KEY, QWEN_BASE_URL(可选，默认阿里兼容地址)
    """

    provider: Provider
    model: str
    temperature: float = 0.0
    max_tokens: int | None = None

    @staticmethod
    def from_env() -> "LLMConfig":
        provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
        if provider not in {"gemini", "openai", "deepseek", "qwen"}:
            raise ValueError(
                "LLM_PROVIDER 非法，请使用: gemini/openai/deepseek/qwen"
            )

        default_model_map = {
            "gemini": "gemini-2.0-flash",
            "openai": "gpt-4o-mini",
            "deepseek": "deepseek-chat",
            "qwen": "qwen-plus",
        }
        model = os.getenv("LLM_MODEL", default_model_map[provider]).strip()
        temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
        max_tokens_raw = os.getenv("LLM_MAX_TOKENS", "").strip()
        max_tokens = int(max_tokens_raw) if max_tokens_raw else None

        return LLMConfig(
            provider=provider,  # type: ignore[arg-type]
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def get_llm_config() -> LLMConfig:
    """读取当前环境变量并生成 LLMConfig。"""
    return LLMConfig.from_env()


def validate_llm_startup() -> list[str]:
    """
    启动前校验 LLM 关键配置，返回问题列表（为空代表通过）。
    """
    issues: list[str] = []
    use_real_llm = os.getenv("USE_REAL_LLM", "0").strip() == "1"
    if not use_real_llm:
        return issues

    try:
        cfg = get_llm_config()
    except Exception as exc:
        return [f"LLM 配置读取失败: {exc}"]

    placeholder_values = {
        "",
        "your_google_api_key_here",
        "your_openai_api_key_here",
        "your_deepseek_api_key_here",
        "your_dashscope_api_key_here",
    }

    if cfg.provider == "gemini":
        key = os.getenv("GOOGLE_API_KEY", "").strip()
        if key in placeholder_values:
            issues.append("GOOGLE_API_KEY 未设置（或仍是占位符）。")
        elif not key.startswith("AIza"):
            issues.append("GOOGLE_API_KEY 格式可疑（通常以 AIza 开头）。")
    elif cfg.provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if key in placeholder_values:
            issues.append("OPENAI_API_KEY 未设置（或仍是占位符）。")
    elif cfg.provider == "deepseek":
        key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if key in placeholder_values:
            issues.append("DEEPSEEK_API_KEY 未设置（或仍是占位符）。")
    else:
        key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if key in placeholder_values:
            issues.append("DASHSCOPE_API_KEY 未设置（或仍是占位符）。")

    return issues


def create_llm(config: LLMConfig | None = None) -> Any:
    """
    按 provider 创建 LangChain ChatModel 实例。

    说明：
    - Gemini 依赖: langchain-google-genai
    - OpenAI/DeepSeek/Qwen 统一走 langchain-openai（OpenAI 兼容接口）
    """
    cfg = config or get_llm_config()

    if cfg.provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise ValueError("缺少 GOOGLE_API_KEY，无法调用 Gemini。")
        module = importlib.import_module("langchain_google_genai")
        chat_cls = getattr(module, "ChatGoogleGenerativeAI")
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "google_api_key": api_key,
            "temperature": cfg.temperature,
        }
        if cfg.max_tokens is not None:
            kwargs["max_output_tokens"] = cfg.max_tokens
        return chat_cls(**kwargs)

    # 其余 provider 统一按 OpenAI 兼容接口创建
    module = importlib.import_module("langchain_openai")
    chat_cls = getattr(module, "ChatOpenAI")

    if cfg.provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("缺少 OPENAI_API_KEY，无法调用 OpenAI。")
        kwargs = {
            "model": cfg.model,
            "api_key": api_key,
            "temperature": cfg.temperature,
        }
    elif cfg.provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise ValueError("缺少 DEEPSEEK_API_KEY，无法调用 DeepSeek。")
        kwargs = {
            "model": cfg.model,
            "api_key": api_key,
            "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "temperature": cfg.temperature,
        }
    else:
        api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            raise ValueError("缺少 DASHSCOPE_API_KEY，无法调用千问。")
        kwargs = {
            "model": cfg.model,
            "api_key": api_key,
            "base_url": os.getenv(
                "QWEN_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            "temperature": cfg.temperature,
        }

    if cfg.max_tokens is not None:
        kwargs["max_tokens"] = cfg.max_tokens
    return chat_cls(**kwargs)
