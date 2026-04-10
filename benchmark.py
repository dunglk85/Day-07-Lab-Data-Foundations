"""
Phase 2 — Benchmark Script (dùng compute_similarity để đánh giá)
Chạy: python benchmark.py

Hai cách đánh giá song song:
  - is_hit()         : keyword check (nhanh, dễ hiểu)
  - semantic_score() : cosine similarity giữa gold_answer và chunk content
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
import os

from src.chunking import compute_similarity
from src.chunking import FixedSizeChunker, SentenceChunker, RecursiveChunker
from src.store import EmbeddingStore
from src.models import Document
from src.embeddings import LocalEmbedder, OpenAIEmbedder, OPENAI_EMBEDDING_MODEL
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------ #
# CONFIG                                                              #
# ------------------------------------------------------------------ #

YOUR_NAME         = "your_name"
CHUNKING_STRATEGY = "fixed"   # "fixed" | "sentence" | "recursive"
CHUNK_SIZE        = 300
OVERLAP           = 50
MAX_SENTENCES     = 3
DATA_DIR          = Path("data/phapluat/")

# Ngưỡng để coi một chunk là "semantically relevant"
SIMILARITY_THRESHOLD = 0.4

BENCHMARK = [
    {
        "query": "Mức phạt khi vi phạm nồng độ cồn khi lái xe là bao nhiêu?",
        "gold_answer": (
            "Người điều khiển xe ô tô có nồng độ cồn vượt mức cho phép "
            "có thể bị phạt tiền từ 30 đến 40 triệu đồng và tước giấy phép "
            "lái xe từ 22 đến 24 tháng."
        ),
        "gold_chunks": ["nồng độ cồn", "phạt tiền", "tước giấy phép", "lái xe"],
    },
    {
        "query": "Thủ tục khởi kiện dân sự tại tòa án được thực hiện như thế nào?",
        "gold_answer": (
            "Người khởi kiện nộp đơn kèm tài liệu chứng cứ tại tòa án có thẩm quyền, "
            "tòa án xem xét thụ lý trong vòng 3 ngày làm việc và thông báo nộp tạm ứng án phí."
        ),
        "gold_chunks": ["khởi kiện", "đơn kiện", "thụ lý", "án phí", "tòa án"],
    },
    {
        "query": "Tội lừa đảo chiếm đoạt tài sản bị xử lý như thế nào theo Bộ luật Hình sự?",
        "gold_answer": (
            "Tội lừa đảo chiếm đoạt tài sản có thể bị phạt tù từ 6 tháng đến 3 năm "
            "với trường hợp thông thường, và lên đến 20 năm hoặc tù chung thân "
            "nếu chiếm đoạt tài sản có giá trị đặc biệt lớn."
        ),
        "gold_chunks": ["lừa đảo", "chiếm đoạt tài sản", "phạt tù", "bộ luật hình sự"],
    },
    {
        "query": "Quyền và nghĩa vụ của người bị tạm giam là gì?",
        "gold_answer": (
            "Người bị tạm giam có quyền được thông báo lý do tạm giam, gặp luật sư, "
            "nhận đồ tiếp tế từ thân nhân và khiếu nại quyết định tạm giam nếu cho là trái pháp luật."
        ),
        "gold_chunks": ["tạm giam", "luật sư", "khiếu nại", "quyền", "thân nhân"],
    },
    {
        "query": "Tranh chấp đất đai giữa các hộ gia đình được giải quyết theo trình tự nào?",
        "gold_answer": (
            "Tranh chấp đất đai phải qua hòa giải tại UBND cấp xã trước, "
            "nếu không hòa giải được thì các bên có thể khởi kiện ra tòa án "
            "hoặc yêu cầu UBND cấp huyện giải quyết."
        ),
        "gold_chunks": ["tranh chấp đất đai", "hòa giải", "UBND", "khởi kiện", "tòa án"],
    },
]


def semantic_score(gold_answer: str, chunk_content: str) -> float:
    """
    Tính cosine similarity giữa gold_answer và chunk_content
    """
    embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
    
    vec_gold  = embedder(gold_answer)
    vec_chunk = embedder(chunk_content)
    return compute_similarity(vec_gold, vec_chunk)


# ------------------------------------------------------------------ #
# Helpers load / chunk / index                                        #
# ------------------------------------------------------------------ #

def load_documents(data_dir: Path) -> list[Document]:
    docs  = []
    paths = sorted(data_dir.glob("**/*.txt")) + sorted(data_dir.glob("**/*.md"))
    for i, path in enumerate(paths):
        content = path.read_text(encoding="utf-8")
        docs.append(Document(
            id=f"doc_{i}",
            content=content,
            metadata={"source": path.name, "path": str(path)},
        ))
    print(f"[load] {len(docs)} tài liệu từ {data_dir}")
    return docs


def make_chunker():
    if CHUNKING_STRATEGY == "fixed":
        return FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    elif CHUNKING_STRATEGY == "sentence":
        return SentenceChunker(max_sentences_per_chunk=MAX_SENTENCES)
    elif CHUNKING_STRATEGY == "recursive":
        return RecursiveChunker(chunk_size=CHUNK_SIZE)
    raise ValueError(f"Unknown strategy: {CHUNKING_STRATEGY}")


def chunk_documents(docs: list[Document]) -> list[Document]:
    chunker  = make_chunker()
    result   = []
    chunk_id = 0
    for doc in docs:
        for idx, text in enumerate(chunker.chunk(doc.content)):
            result.append(Document(
                id=f"chunk_{chunk_id}",
                content=text,
                metadata={**doc.metadata, "doc_id": doc.id, "chunk_index": idx},
            ))
            chunk_id += 1
    print(f"[chunk] strategy={CHUNKING_STRATEGY}, chunk_size={CHUNK_SIZE} → {len(result)} chunks")
    return result


def is_hit(content: str, gold_kws: list[str]) -> bool:
    """Keyword-based check (baseline)."""
    lower = content.lower()
    return any(kw.lower() in lower for kw in gold_kws)


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main() -> None:
    print("=" * 65)
    print(f"  Benchmark — {YOUR_NAME} — strategy: {CHUNKING_STRATEGY}")
    print("=" * 65)

    docs = load_documents(DATA_DIR)
    if not docs:
        print("[ERROR] Không tìm thấy tài liệu. Kiểm tra DATA_DIR.")
        return

    chunks = chunk_documents(docs)
    store  = EmbeddingStore(collection_name=YOUR_NAME)
    store.add_documents(chunks)
    print(f"[store] Đã index {store.get_collection_size()} chunks\n")

    total_sem_hits = 0
    query_sem_avgs: list[float] = []

    for i, item in enumerate(BENCHMARK, 1):
        query    = item["query"]
        gold     = item["gold_answer"]

        print(f"─── Query {i}: {query}")
        print(f"    Gold: {gold[:90]}...")

        results    = store.search(query, top_k=3)
        sem_scores: list[float] = []

        for rank, r in enumerate(results, 1):
            content = r.get("content", "")
            score   = r.get("score",   0.0)
            source  = r.get("metadata", {}).get("source", "?")

            sim    = semantic_score(gold, content)

            sem_scores.append(sim)

            sem_mark = "✓" if sim >= SIMILARITY_THRESHOLD else "✗"
            preview  = content[:100].replace("\n", " ")

            print(f"similarity={sim:.4f} {sem_mark}  src={source}")
            print(f"        {preview}...")

        avg_sim  = sum(sem_scores) / len(sem_scores) if sem_scores else 0.0
        sem_hits = sum(1 for s in sem_scores if s >= SIMILARITY_THRESHOLD)

        total_sem_hits += sem_hits

        print(f"    → Semantic Precision@3 = {sem_hits}/{len(results)} = {sem_hits/len(results):.2f}"
              f"  (avg sim = {avg_sim:.4f})\n")

    n           = len(BENCHMARK) * 3
    overall_sem = total_sem_hits / n

    print("=" * 65)
    print(f"  Strategy : {CHUNKING_STRATEGY} | chunk_size={CHUNK_SIZE}")
    print(f"  Overall Semantic Precision@3 = {total_sem_hits}/{n}  = {overall_sem:.2f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
