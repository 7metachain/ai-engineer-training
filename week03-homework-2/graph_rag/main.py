"""
Graph-RAG: Document retrieval (LlamaIndex) + Neo4j graph reasoning (Cypher) for multi-hop QA.

Run:
  - Ingest sample graph into Neo4j:
      python -m graph_rag.main ingest --edges ./graph_rag/data/ownership_edges.jsonl
  - Ask:
      python -m graph_rag.main ask "A公司的最大股东是谁？" --top-k 3 --max-hops 3

Env:
  - NEO4J_URI (default: bolt://localhost:7687)
  - NEO4J_USER (default: neo4j)
  - NEO4J_PASSWORD (required for Neo4j mode)

Notes:
  - If Neo4j connection fails, this script will fallback to an in-memory graph built from edges.jsonl
    so you can still run an end-to-end demo.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, NamedTuple


DEFAULT_DOCS = Path(__file__).with_suffix("").parent / "data" / "docs.jsonl"
DEFAULT_EDGES = Path(__file__).with_suffix("").parent / "data" / "ownership_edges.jsonl"


@dataclass(frozen=True)
class Neo4jSettings:
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "")


class PathExplain(NamedTuple):
    nodes: list[str]
    percents: list[float]
    weight: float


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_company(question: str) -> str | None:
    # handle: “A 公司的最大股东是谁？” / “A公司的最大股东是谁？”
    q = question.replace(" ", "")
    m = re.search(r"^(?P<name>.+?)公司的最大股东是谁[？?]?$", q)
    if m:
        return m.group("name") + "公司" if not m.group("name").endswith("公司") else m.group("name")
    # fallback: find first "...公司"
    m2 = re.search(r"(?P<name>[^，。？?]+?公司)", q)
    return m2.group("name") if m2 else None


def _build_rag_index(docs_path: Path) -> Any:
    try:
        from llama_index.core import Document, VectorStoreIndex  # type: ignore[import-not-found]
        from llama_index.core.node_parser import SentenceSplitter  # type: ignore[import-not-found]
    except Exception as e:  # noqa: BLE001
        # Fallback: dependency-free "index" that uses simple token overlap scoring.
        docs_rows = _load_jsonl(docs_path)
        simple_docs = []
        for r in docs_rows:
            simple_docs.append(
                {
                    "doc_id": r.get("doc_id"),
                    "title": r.get("title"),
                    "text": r.get("text"),
                    "full_text": (f"{r.get('title','')}\n{r.get('text','')}".strip()),
                }
            )
        return {"mode": "simple", "docs": simple_docs, "error": str(e)}

    docs_rows = _load_jsonl(docs_path)
    docs: list[Any] = []
    for r in docs_rows:
        text = f"{r.get('title','')}\n{r.get('text','')}".strip()
        docs.append(Document(text=text, metadata={"doc_id": r.get("doc_id"), "title": r.get("title")}))

    # Simple chunking with overlap (document retrieval only, not generation)
    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=80)
    return {"mode": "llama_index", "index": VectorStoreIndex.from_documents(docs, node_parser=node_parser)}


def _rag_retrieve(index: Any, query: str, top_k: int) -> list[dict[str, Any]]:
    if isinstance(index, dict) and index.get("mode") == "simple":
        # Very small, dependency-free baseline: character-level overlap scoring for Chinese
        q = query.strip()
        qset = set(q)
        scored = []
        for d in index["docs"]:
            txt = d.get("full_text") or ""
            tset = set(txt)
            inter = len(qset & tset)
            denom = max(1, len(qset))
            score = inter / denom
            scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, d in scored[:top_k]:
            out.append({"score": float(score), "doc_id": d.get("doc_id"), "title": d.get("title"), "text": d.get("full_text")})
        return out

    if isinstance(index, dict) and index.get("mode") == "llama_index":
        try:
            from llama_index.core import QueryBundle  # type: ignore[import-not-found]
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("Missing llama-index core dependency.") from e

        retriever = index["index"].as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(QueryBundle(query))
        out: list[dict[str, Any]] = []
        for n in nodes:
            meta = getattr(n.node, "metadata", {}) or {}
            out.append(
                {
                    "score": float(getattr(n, "score", 0.0) or 0.0),
                    "doc_id": meta.get("doc_id"),
                    "title": meta.get("title"),
                    "text": getattr(n.node, "text", None),
                }
            )
        return out

    raise TypeError("Unknown index type for retrieval.")


def _normalize_percent(p: float) -> float:
    # Accept [0,1] or [0,100]
    if p > 1.0:
        return p / 100.0
    return p


def _best_shareholder_path_in_memory(
    *, company: str, edges: list[dict[str, Any]], max_hops: int
) -> tuple[str | None, PathExplain | None]:
    # edges: from -> to, meaning 'from owns to'
    adj_in: dict[str, list[tuple[str, float]]] = {}
    for e in edges:
        frm = str(e["from"])
        to = str(e["to"])
        pct = _normalize_percent(float(e["percent"]))
        adj_in.setdefault(to, []).append((frm, pct))

    best: PathExplain | None = None

    def dfs(cur: str, path_nodes: list[str], path_pcts: list[float], depth: int) -> None:
        nonlocal best
        if depth >= max_hops:
            return
        for owner, pct in adj_in.get(cur, []):
            nodes2 = [owner] + path_nodes
            pcts2 = [pct] + path_pcts
            weight = 1.0
            for x in pcts2:
                weight *= x
            cand = PathExplain(nodes=nodes2, percents=pcts2, weight=weight)
            if best is None or cand.weight > best.weight:
                best = cand
            dfs(owner, nodes2, pcts2, depth + 1)

    # path is owner -> ... -> company
    dfs(company, [company], [], 0)
    if best is None:
        return None, None
    return best.nodes[0], best


def _try_connect_neo4j(settings: Neo4jSettings) -> Any | None:
    try:
        from neo4j import GraphDatabase  # type: ignore[import-not-found]
    except Exception:
        return None
    if not settings.password:
        return None
    try:
        driver = GraphDatabase.driver(settings.uri, auth=(settings.user, settings.password))
        # verify connectivity
        driver.verify_connectivity()
        return driver
    except Exception:
        return None


def _neo4j_init_schema(driver: Any) -> None:
    cyphers = [
        "CREATE CONSTRAINT company_name IF NOT EXISTS FOR (c:Entity) REQUIRE c.name IS UNIQUE",
        "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
    ]
    with driver.session() as s:
        for c in cyphers:
            s.run(c)


def _neo4j_ingest_edges(driver: Any, edges: list[dict[str, Any]]) -> None:
    # Store everything as :Entity; OWNS directed edge with percent [0,1]
    query = """
UNWIND $rows AS row
MERGE (src:Entity {name: row.from})
MERGE (dst:Entity {name: row.to})
MERGE (src)-[r:OWNS]->(dst)
SET r.percent = row.percent
"""
    rows = [{"from": str(e["from"]), "to": str(e["to"]), "percent": _normalize_percent(float(e["percent"]))} for e in edges]
    with driver.session() as s:
        s.run(query, rows=rows)


def _neo4j_best_path(*, driver: Any, company: str, max_hops: int) -> tuple[str | None, PathExplain | None, str]:
    # Multi-hop "ownership penetration": maximize product of percentages along path.
    cypher = f"""
MATCH p=(root:Entity {{name:$company}})<-[:OWNS*1..{max_hops}]-(s:Entity)
WITH s, nodes(p) AS ns, relationships(p) AS rs,
     reduce(acc=1.0, r IN rs | acc * coalesce(r.percent, 0.0)) AS weight
RETURN s.name AS shareholder, [n IN ns | n.name] AS nodes,
       [r IN rs | r.percent] AS percents, weight
ORDER BY weight DESC
LIMIT 1
"""
    with driver.session() as s:
        rec = s.run(cypher, company=company).single()
        if not rec:
            return None, None, cypher.strip()
        nodes = list(rec["nodes"])
        percents = [float(x) for x in rec["percents"]]
        weight = float(rec["weight"])
        shareholder = rec["shareholder"]
        return shareholder, PathExplain(nodes=nodes, percents=percents, weight=weight), cypher.strip()


def _joint_score(*, rag_score: float, graph_weight: float, alpha: float) -> float:
    # rag_score: similarity (often [0,1] but not guaranteed). We'll clamp roughly to [0,1] for fusion.
    r = max(0.0, min(1.0, rag_score))
    g = max(0.0, min(1.0, graph_weight))
    return alpha * r + (1.0 - alpha) * g


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="graph_rag")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_ingest = sub.add_parser("ingest", help="Create schema and ingest edges into Neo4j.")
    sp_ingest.add_argument("--edges", type=str, default=str(DEFAULT_EDGES))

    sp_ask = sub.add_parser("ask", help="Ask a multi-hop question.")
    sp_ask.add_argument("question", type=str)
    sp_ask.add_argument("--docs", type=str, default=str(DEFAULT_DOCS))
    sp_ask.add_argument("--edges", type=str, default=str(DEFAULT_EDGES), help="Used for fallback in-memory reasoning.")
    sp_ask.add_argument("--top-k", type=int, default=3)
    sp_ask.add_argument("--max-hops", type=int, default=3)
    sp_ask.add_argument("--alpha", type=float, default=0.6, help="Fusion weight for RAG similarity vs graph confidence.")
    sp_ask.add_argument("--company", type=str, default=None, help="Override extracted company name.")

    return p.parse_args(argv)


def ask(*, question: str, docs_path: Path, edges_path: Path, top_k: int, max_hops: int, alpha: float, company: str | None) -> dict[str, Any]:
    company_name = company or _extract_company(question)
    if not company_name:
        raise ValueError("Cannot extract company name from question; please pass --company.")

    # 1) RAG retrieve (evidence)
    idx = _build_rag_index(docs_path)
    rag_hits = _rag_retrieve(idx, query=company_name + " 股权 结构 最大 股东", top_k=top_k)
    best_rag_score = rag_hits[0]["score"] if rag_hits else 0.0

    # 2) Graph reasoning (Neo4j multi-hop)
    settings = Neo4jSettings()
    driver = _try_connect_neo4j(settings)
    graph_mode = "neo4j" if driver else "in_memory"

    edges = _load_jsonl(edges_path)
    cypher_used = None
    shareholder = None
    explain = None
    graph_weight = 0.0

    if driver:
        shareholder, explain, cypher_used = _neo4j_best_path(driver=driver, company=company_name, max_hops=max_hops)
        if explain:
            graph_weight = explain.weight
        driver.close()
    else:
        shareholder, explain = _best_shareholder_path_in_memory(company=company_name, edges=edges, max_hops=max_hops)
        if explain:
            graph_weight = explain.weight

    final_score = _joint_score(rag_score=best_rag_score, graph_weight=graph_weight, alpha=alpha)

    # 3) Explainable output
    path_str = None
    if explain:
        # nodes are ordered: shareholder -> ... -> company (in-memory), or root->...?? from cypher nodes(p) gives [root,...,shareholder]?
        # normalize to "shareholder -> ... -> company"
        nodes = list(explain.nodes)
        percents = list(explain.percents)
        if nodes and nodes[0] == company_name:
            nodes = list(reversed(nodes))
            percents = list(reversed(percents))
        hops = []
        for i in range(len(percents)):
            hops.append(f"{nodes[i]} -[{percents[i]:.4f}]-> {nodes[i+1]}")
        path_str = " ; ".join(hops) + f" ; weight={explain.weight:.6f}"

    return {
        "question": question,
        "company": company_name,
        "rag": {"top_k": top_k, "hits": rag_hits},
        "graph": {
            "mode": graph_mode,
            "neo4j_uri": settings.uri if graph_mode == "neo4j" else None,
            "max_hops": max_hops,
            "shareholder": shareholder,
            "path": path_str,
            "weight": graph_weight,
            "cypher": cypher_used,
        },
        "fusion": {"alpha": alpha, "score": final_score},
        "answer": f"{company_name} 的（穿透后）最大股东是：{shareholder or '未知'}",
        "explain": {"reasoning_path": path_str, "notes": "融合分数=alpha*RAG相似度+(1-alpha)*图谱路径权重(股权乘积)"},
    }


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(list(argv) if argv is not None else sys.argv[1:])

    if args.cmd == "ingest":
        edges_path = Path(args.edges)
        edges = _load_jsonl(edges_path)
        settings = Neo4jSettings()
        driver = _try_connect_neo4j(settings)
        if not driver:
            raise RuntimeError(
                "Neo4j not available. Please set NEO4J_PASSWORD and ensure Neo4j is reachable at NEO4J_URI."
            )
        _neo4j_init_schema(driver)
        _neo4j_ingest_edges(driver, edges)
        driver.close()
        print(json.dumps({"ok": True, "ingested_edges": len(edges), "neo4j_uri": settings.uri}, ensure_ascii=False))
        return

    if args.cmd == "ask":
        out = ask(
            question=args.question,
            docs_path=Path(args.docs),
            edges_path=Path(args.edges),
            top_k=args.top_k,
            max_hops=args.max_hops,
            alpha=args.alpha,
            company=args.company,
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    raise AssertionError(f"Unhandled cmd: {args.cmd}")


if __name__ == "__main__":
    main()