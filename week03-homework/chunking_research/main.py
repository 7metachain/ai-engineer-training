
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    SentenceSplitter,
    SentenceWindowNodeParser,
    TokenTextSplitter,
)
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
)
from llama_index.llms.openai_like import OpenAILike


def configure_llamaindex() -> None:
    """Configure LlamaIndex to use Qwen + DashScope embeddings."""
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing DASHSCOPE_API_KEY. Please export it before running the script."
        )

    Settings.llm = OpenAILike(
        model="qwen-plus",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
        is_chat_model=True,
    )
    Settings.embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
        embed_batch_size=6,
        embed_input_length=8192,
    )


def load_documents(data_dir: Path) -> List:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    reader = SimpleDirectoryReader(input_dir=str(data_dir), recursive=True)
    return reader.load_data()


def evaluate_splitter(
    name: str,
    docs: List,
    question: str,
    ground_truth: str,
    splitter=None,
    window_parser: Optional[SentenceWindowNodeParser] = None,
    similarity_top_k: int = 5,
) -> Dict[str, Any]:
    if window_parser:
        nodes = window_parser.get_nodes_from_documents(docs)
        index = VectorStoreIndex(nodes)
        postprocessors = [
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ]
    else:
        index = VectorStoreIndex.from_documents(
            docs, transformations=[splitter] if splitter else None
        )
        postprocessors = []

    query_engine = index.as_query_engine(
        similarity_top_k=similarity_top_k,
        streaming=False,
        node_postprocessors=postprocessors,
    )

    response = query_engine.query(question)
    contexts = [sn.node.get_content() for sn in response.source_nodes]
    context_hit = False
    if ground_truth:
        lowered = ground_truth.lower()
        context_hit = any(lowered in ctx.lower() for ctx in contexts)

    return {
        "name": name,
        "answer": str(response),
        "num_contexts": len(contexts),
        "context_hit_ground_truth": context_hit,
        "contexts": contexts,
    }


def run_experiments(
    data_dir: Path,
    question: str,
    ground_truth: str,
    include_markdown: bool,
    similarity_top_k: int,
) -> List[Dict[str, Any]]:
    docs = load_documents(data_dir)

    experiments: List[Tuple[str, Dict[str, Any]]] = [
        (
            "Sentence",
            dict(
                splitter=SentenceSplitter(
                    chunk_size=512,
                    chunk_overlap=50,
                )
            ),
        ),
        (
            "Token",
            dict(
                splitter=TokenTextSplitter(
                    chunk_size=32,
                    chunk_overlap=4,
                    separator="\n",
                )
            ),
        ),
        (
            "Sentence Window",
            dict(
                window_parser=SentenceWindowNodeParser.from_defaults(
                    window_size=3,
                    window_metadata_key="window",
                    original_text_metadata_key="original_text",
                )
            ),
        ),
    ]

    if include_markdown:
        experiments.append(
            (
                "Markdown",
                dict(
                    splitter=None,
                    window_parser=None,
                    markdown_parser=MarkdownNodeParser(),
                ),
            )
        )

    results: List[Dict[str, Any]] = []
    for name, config in experiments:
        if name == "Markdown":
            parser: MarkdownNodeParser = config["markdown_parser"]
            nodes = parser.get_nodes_from_documents(docs)
            index = VectorStoreIndex(nodes)
            query_engine = index.as_query_engine(similarity_top_k=similarity_top_k)
            response = query_engine.query(question)
            contexts = [sn.node.get_content() for sn in response.source_nodes]
            context_hit = (
                ground_truth
                and any(ground_truth.lower() in ctx.lower() for ctx in contexts)
            )
            results.append(
                {
                    "name": name,
                    "answer": str(response),
                    "num_contexts": len(contexts),
                    "context_hit_ground_truth": bool(context_hit),
                    "contexts": contexts,
                }
            )
            continue

        result = evaluate_splitter(
            name=name,
            docs=docs,
            question=question,
            ground_truth=ground_truth,
            splitter=config.get("splitter"),
            window_parser=config.get("window_parser"),
            similarity_top_k=similarity_top_k,
        )
        results.append(result)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run chunking experiments with LlamaIndex."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/chunking"),
        help="Directory containing text/markdown documents.",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=False,
        default="文章的核心观点是什么？",
        help="Evaluation question to ask.",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        required=False,
        default="",
        help="Reference answer snippet for simple context check.",
    )
    parser.add_argument(
        "--similarity-top-k",
        type=int,
        default=5,
        help="Top-k nodes for retrieval.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("chunking_research/results.json"),
        help="Where to save experiment results.",
    )
    parser.add_argument(
        "--include-markdown",
        action="store_true",
        help="Include MarkdownNodeParser experiment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_llamaindex()

    results = run_experiments(
        data_dir=args.data_dir,
        question=args.question,
        ground_truth=args.ground_truth,
        include_markdown=args.include_markdown,
        similarity_top_k=args.similarity_top_k,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Saved results to {args.output}")
    for r in results:
        print(f"[{r['name']}] context_hit={r['context_hit_ground_truth']}")


if __name__ == "__main__":
    main()