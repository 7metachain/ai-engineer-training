"""
Milvus FAQ retrieval system (LlamaIndex + Milvus) + FastAPI.

Run:
  - CLI query:      python -m milvus_faq.main query "如何退货？" --top-k 3
  - Reindex:        python -m milvus_faq.main reindex --kb ./milvus_faq/data/faqs.jsonl
  - Start API:      python -m milvus_faq.main api --host 0.0.0.0 --port 8000

Env (optional):
  - MILVUS_URI (default: http://localhost:19530)
  - MILVUS_DB (default: default)
  - MILVUS_COLLECTION (default: faq)
  - AUTO_REINDEX_ON_QUERY (default: false)
  - OPENAI_API_KEY (optional, if using OpenAI embedding/llm)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_KB = Path(__file__).with_suffix("").parent / "data" / "faqs.jsonl"


@dataclass(frozen=True)
class Settings:
    milvus_uri: str = os.getenv("MILVUS_URI", "http://localhost:19530")
    milvus_db: str = os.getenv("MILVUS_DB", "default")
    milvus_collection: str = os.getenv("MILVUS_COLLECTION", "faq")
    auto_reindex_on_query: bool = os.getenv("AUTO_REINDEX_ON_QUERY", "false").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def _ensure_sample_kb(kb_path: Path) -> None:
    kb_path.parent.mkdir(parents=True, exist_ok=True)
    if kb_path.exists():
        return
    sample = [
        {"question": "如何退货？", "answer": "请在订单详情页点击“申请退货”，按提示填写原因并提交，客服审核后会安排取件。"},
        {"question": "多久可以退款？", "answer": "退货仓库签收后 1-3 个工作日原路退款；具体到账时间以银行/支付渠道为准。"},
        {"question": "如何修改收货地址？", "answer": "若订单未发货，可在订单详情页修改地址；若已发货，请联系快递或客服协助拦截。"},
        {"question": "发票怎么开？", "answer": "在“我的-发票管理”中申请开票，支持电子发票，开具后可下载或邮件发送。"},
    ]
    with kb_path.open("w", encoding="utf-8") as f:
        for row in sample:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_faqs_jsonl(kb_path: Path) -> list[dict[str, str]]:
    faqs: list[dict[str, str]] = []
    with kb_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = (obj.get("question") or "").strip()
            a = (obj.get("answer") or "").strip()
            if q and a:
                faqs.append({"question": q, "answer": a})
    if not faqs:
        raise ValueError(f"KB is empty or invalid: {kb_path}")
    return faqs


def _try_build_index(*, faqs: list[dict[str, str]], settings: Settings) -> Any:
    """
    Build (or rebuild) a VectorStoreIndex backed by Milvus.
    """
    try:
        from llama_index.core import Document, StorageContext, VectorStoreIndex
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.vector_stores.milvus import MilvusVectorStore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Missing dependencies. Please install: llama-index, llama-index-vector-stores-milvus, pymilvus"
        ) from e

    # Build documents: put Q+A into text and keep structured metadata for answer display.
    docs: list[Any] = []
    for i, row in enumerate(faqs):
        text = f"Q: {row['question']}\nA: {row['answer']}"
        docs.append(
            Document(
                text=text,
                metadata={"faq_id": str(i), "question": row["question"], "answer": row["answer"]},
            )
        )

    # Chunking optimization: prefer semantic splitting when available, fallback to overlap sentence splitter.
    node_parser = None
    try:
        from llama_index.core.node_parser import SemanticSplitterNodeParser

        # Default embedding model used by LlamaIndex; user can override via global settings or env.
        node_parser = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95)
    except Exception:
        node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=80)

    vector_store = MilvusVectorStore(
        uri=settings.milvus_uri,
        collection_name=settings.milvus_collection,
        database_name=settings.milvus_db,
        dim=None,  # let backend infer; requires embedding model to be configured in llama_index settings
        overwrite=True,  # hot update via rebuild
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, node_parser=node_parser)
    return index


def _try_query_index(*, index: Any, question: str, top_k: int) -> dict[str, Any]:
    try:
        from llama_index.core import QueryBundle
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing llama-index core dependency.") from e

    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(QueryBundle(question))
    items: list[dict[str, Any]] = []
    for n in nodes:
        meta = getattr(n.node, "metadata", {}) or {}
        items.append(
            {
                "score": float(getattr(n, "score", 0.0) or 0.0),
                "question": meta.get("question"),
                "answer": meta.get("answer"),
                "faq_id": meta.get("faq_id"),
                "text": getattr(n.node, "text", None),
            }
        )
    best = items[0] if items else None
    return {"query": question, "top_k": top_k, "best_answer": (best or {}).get("answer"), "matches": items}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="milvus_faq")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_reindex = sub.add_parser("reindex", help="Rebuild Milvus index from KB.")
    sp_reindex.add_argument("--kb", type=str, default=str(DEFAULT_KB), help="Path to faqs.jsonl")

    sp_query = sub.add_parser("query", help="Query from CLI.")
    sp_query.add_argument("question", type=str)
    sp_query.add_argument("--kb", type=str, default=str(DEFAULT_KB), help="KB path used for auto-reindex check")
    sp_query.add_argument("--top-k", type=int, default=3)

    sp_api = sub.add_parser("api", help="Start FastAPI server.")
    sp_api.add_argument("--kb", type=str, default=str(DEFAULT_KB))
    sp_api.add_argument("--host", type=str, default="127.0.0.1")
    sp_api.add_argument("--port", type=int, default=8000)

    return p.parse_args(argv)


def _build_app(*, kb_path: Path, settings: Settings) -> Any:
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependencies. Please install: fastapi, pydantic, uvicorn") from e

    app = FastAPI(title="Milvus FAQ Retrieval", version="0.1.0")

    class QueryIn(BaseModel):
        question: str = Field(..., description="User natural language question")
        top_k: int = Field(3, ge=1, le=20)

    class ReindexIn(BaseModel):
        kb_path: str | None = None

    state: dict[str, Any] = {"index": None, "kb_mtime": None, "kb_path": str(kb_path)}

    def ensure_index() -> Any:
        path = Path(state["kb_path"])
        _ensure_sample_kb(path)
        mtime = path.stat().st_mtime
        if state["index"] is None:
            faqs = _load_faqs_jsonl(path)
            state["index"] = _try_build_index(faqs=faqs, settings=settings)
            state["kb_mtime"] = mtime
        elif settings.auto_reindex_on_query and state["kb_mtime"] != mtime:
            faqs = _load_faqs_jsonl(path)
            state["index"] = _try_build_index(faqs=faqs, settings=settings)
            state["kb_mtime"] = mtime
        return state["index"]

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "milvus_uri": settings.milvus_uri, "collection": settings.milvus_collection}

    @app.post("/query")
    def query(inp: QueryIn) -> dict[str, Any]:
        try:
            index = ensure_index()
            return _try_query_index(index=index, question=inp.question, top_k=inp.top_k)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/admin/reindex")
    def reindex(inp: ReindexIn) -> dict[str, Any]:
        try:
            if inp.kb_path:
                state["kb_path"] = inp.kb_path
            path = Path(state["kb_path"])
            _ensure_sample_kb(path)
            faqs = _load_faqs_jsonl(path)
            state["index"] = _try_build_index(faqs=faqs, settings=settings)
            state["kb_mtime"] = path.stat().st_mtime
            return {"ok": True, "kb_path": state["kb_path"], "reindexed_at": int(time.time())}
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(list(argv) if argv is not None else sys.argv[1:])
    settings = Settings()

    if args.cmd == "reindex":
        kb_path = Path(args.kb)
        _ensure_sample_kb(kb_path)
        faqs = _load_faqs_jsonl(kb_path)
        _ = _try_build_index(faqs=faqs, settings=settings)
        print(json.dumps({"ok": True, "kb": str(kb_path), "collection": settings.milvus_collection}, ensure_ascii=False))
        return

    if args.cmd == "query":
        kb_path = Path(args.kb)
        _ensure_sample_kb(kb_path)
        faqs = _load_faqs_jsonl(kb_path)
        index = _try_build_index(faqs=faqs, settings=settings)
        out = _try_query_index(index=index, question=args.question, top_k=args.top_k)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    if args.cmd == "api":
        kb_path = Path(args.kb)
        app = _build_app(kb_path=kb_path, settings=settings)
        try:
            import uvicorn
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("Missing dependency: uvicorn") from e
        uvicorn.run(app, host=args.host, port=args.port)
        return

    raise AssertionError(f"Unhandled cmd: {args.cmd}")


if __name__ == "__main__":
    main()