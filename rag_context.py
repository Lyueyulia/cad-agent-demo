from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class KnowledgeChunk:
    source: str
    title: str
    content: str


def _split_markdown_chunks(text: str) -> list[tuple[str, str]]:
    """
    将 markdown 按标题切分为小块，便于轻量检索。
    返回 (title, content) 列表。
    """
    lines = text.splitlines()
    chunks: list[tuple[str, str]] = []
    current_title = "Document"
    buff: list[str] = []

    for line in lines:
        if re.match(r"^#{1,4}\s+", line):
            if buff:
                chunks.append((current_title, "\n".join(buff).strip()))
                buff = []
            current_title = re.sub(r"^#{1,4}\s+", "", line).strip()
            continue
        buff.append(line)

    if buff:
        chunks.append((current_title, "\n".join(buff).strip()))
    return chunks


def _load_knowledge_chunks(base_dir: Path) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    if not base_dir.exists():
        return chunks

    for md in sorted(base_dir.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        for title, content in _split_markdown_chunks(text):
            if content:
                chunks.append(
                    KnowledgeChunk(
                        source=md.name,
                        title=title,
                        content=content,
                    )
                )
    return chunks


def _score_chunk(query: str, chunk: KnowledgeChunk) -> int:
    q_tokens = [t for t in re.split(r"[\s,.;:()\[\]{}<>/\-_\n]+", query.lower()) if t]
    hay = f"{chunk.title}\n{chunk.content}".lower()
    score = 0
    for token in q_tokens:
        if len(token) < 2:
            continue
        if token in hay:
            score += 1
    return score


def retrieve_context(query: str, top_k: int = 4) -> str:
    """
    轻量 RAG 检索入口：从 docs/rag 下检索最相关片段并拼接。
    """
    rag_dir = Path(__file__).parent / "docs" / "rag"
    chunks = _load_knowledge_chunks(rag_dir)
    if not chunks:
        return ""

    ranked = sorted(chunks, key=lambda c: _score_chunk(query, c), reverse=True)
    selected = [c for c in ranked[:top_k] if _score_chunk(query, c) > 0]
    if not selected:
        selected = ranked[: min(top_k, 2)]

    parts = []
    for c in selected:
        parts.append(f"[{c.source} :: {c.title}]\n{c.content}")
    return "\n\n".join(parts).strip()
