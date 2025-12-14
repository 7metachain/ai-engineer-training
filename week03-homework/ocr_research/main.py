
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Union

from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
)
from llama_index.llms.openai_like import OpenAILike
from paddleocr import PaddleOCR


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def configure_llamaindex() -> None:
    """Configure LlamaIndex to use Qwen + DashScope when we need to query."""
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY for LlamaIndex querying.")

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


class ImageOCRReader(BaseReader):
    """Use PP-OCR to extract text from images and return LlamaIndex Documents."""

    def __init__(self, lang: str = "ch", use_gpu: bool = False, **ocr_kwargs):
        self.lang = lang
        self.use_gpu = use_gpu
        self.ocr = PaddleOCR(lang=lang, use_gpu=use_gpu, **ocr_kwargs)

    def _normalize_file_input(self, file: Union[str, Sequence[str]]) -> List[Path]:
        if isinstance(file, (str, Path)):
            paths = [Path(file)]
        else:
            paths = [Path(p) for p in file]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"Image not found: {p}")
        return paths

    def _format_blocks(self, raw_result) -> (str, List[float]):
        text_blocks: List[str] = []
        confidences: List[float] = []
        if not raw_result or not raw_result[0]:
            return "", confidences

        for idx, line in enumerate(raw_result[0]):
            text, conf = line[1]
            confidences.append(float(conf))
            text_blocks.append(
                f"[Text Block {idx + 1}] (conf: {float(conf):.2f}): {text}"
            )

        return "\n".join(text_blocks), confidences

    def load_data(self, file: Union[str, Sequence[str]]) -> List[Document]:
        paths = self._normalize_file_input(file)
        documents: List[Document] = []

        for path in paths:
            result = self.ocr.ocr(str(path), cls=False)
            text_content, confidences = self._format_blocks(result)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            metadata = {
                "image_path": str(path),
                "ocr_model": "PP-OCR",
                "language": self.lang,
                "num_text_blocks": len(confidences),
                "avg_confidence": avg_conf,
            }
            documents.append(Document(text=text_content, metadata=metadata))

        return documents


def gather_images(directory: Path) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    images: List[Path] = []
    for path in directory.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    if not images:
        raise RuntimeError(f"No images found under {directory}")
    return images


def run_query(documents: List[Document], question: str) -> str:
    configure_llamaindex()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    return str(response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image OCR Reader demo.")
    parser.add_argument(
        "--images",
        nargs="*",
        help="Image paths to process.",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        help="Directory containing images to OCR.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="ch",
        help="OCR language model (e.g., ch, en).",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for PaddleOCR.",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=Path("ocr_research/ocr_results.json"),
        help="Path to save OCR output.",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Optional question to ask with LlamaIndex after OCR.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_paths: List[Path] = []
    if args.images:
        image_paths.extend([Path(p) for p in args.images])
    if args.dir:
        image_paths.extend(gather_images(args.dir))

    if not image_paths:
        raise RuntimeError("Please provide --images or --dir with images.")

    reader = ImageOCRReader(lang=args.lang, use_gpu=args.use_gpu)
    documents = reader.load_data(image_paths)

    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    args.save_json.write_text(
        json.dumps(
            [dict(text=doc.text, metadata=doc.metadata) for doc in documents],
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"OCR results saved to {args.save_json}")

    if args.query:
        answer = run_query(documents, args.query)
        print(f"Query answer: {answer}")


if __name__ == "__main__":
    main()