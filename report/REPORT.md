# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Lê Kim Dũng
**Nhóm:** 04-E403
**Ngày:** 10/4/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> *Viết 1-2 câu:* Hai vector “gần giống nhau” về hướng → nội dung của chúng giống nhau về mặt ngữ nghĩa (hoặc từ vựng, tùy embedding).

**Ví dụ HIGH similarity:**
- Sentence A: "Người lái xe có nồng độ cồn vượt mức sẽ bị phạt tiền."
- Sentence B: "Tài xế vi phạm nồng độ cồn có thể bị xử phạt bằng tiền."
- Tại sao tương đồng: Cùng nói về vi phạm nồng độ cồn khi lái xe, cùng ý bị phạt tiền

**Ví dụ LOW similarity:**
- Sentence A: "Người lái xe có nồng độ cồn vượt mức sẽ bị phạt tiền."
- Sentence B: "Thủ tục khởi kiện dân sự tại tòa án gồm nhiều bước."
- Tại sao khác: Sentence A: nói về giao thông / vi phạm nồng độ cồn. Sentence B: nói về pháp lý / thủ tục tòa án. Hai câu không chung chủ đề, không chung từ khóa quan trọng.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> *Viết 1-2 câu:* Cosine similarity đo “hướng”, Euclidean distance đo “khoảng cách tuyệt đối”. Trong text embeddings, ý nghĩa câu nằm ở hướng vector, không phải độ dài vector.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* $num\_chunks = \left\lceil \frac{N - chunk\_size}{chunk\_size - overlap} \right\rceil + 1$
> *Đáp án:* 23

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Viết 1-2 câu:* Chunks tăng: 23 → 25. Muốn overlap nhiều hơn vì: tránh mất ngữ cảnh ở “biên chunk”, tăng khả năng retrieve đúng (recall ↑), tốt hơn cho semantic embedding.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Pháp luật

**Tại sao nhóm chọn domain này?**
> *Viết 2-3 câu:* Lĩnh vực pháp luật có khối lượng tài liệu lớn, nhiều quy định phức tạp và thường xuyên được người dùng tra cứu. Các câu hỏi trong domain này thường mang tính lặp lại và có cấu trúc rõ ràng, rất phù hợp để áp dụng hệ thống RAG nhằm hỗ trợ truy xuất thông tin nhanh và chính xác.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | data/phapluat/01_pháp_luật.md| https://vnexpress.net/phap-luat| 4241| doc_id, source |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| doc_id| integer | 1 | Giúp xác định chunk thuộc về tài liệu nào, từ đó có thể group các chunk liên quan, tránh trả về nhiều đoạn trùng lặp từ cùng một nguồn và hỗ trợ truy vết (traceability) về tài liệu gốc khi hiển thị kết quả cho người dùng.|
| | | | |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| data/phapluat/01_pháp_luật.md | FixedSizeChunker (`fixed_size`) | 41|298.8 | Partially|
| | SentenceChunker (`by_sentences`) | 21| 485.6| Partially|
| | RecursiveChunker (`recursive`) |98| 102.6| Partially|

### Strategy Của Tôi

**Loại:** [FixedSizeChunker / SentenceChunker / RecursiveChunker / custom strategy]

**Mô tả cách hoạt động:**
> *Viết 3-4 câu: strategy chunk thế nào? Dựa trên dấu hiệu gì?*

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Viết 2-3 câu: domain có pattern gì mà strategy khai thác?*

**Code snippet (nếu custom):**
```python
# Paste implementation here
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| | best baseline | | 298.8| |
| | **của tôi** | | | |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | | | | |
| [Tên] | | | | |
| [Tên] | | | | |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:*

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> *Viết 2-3 câu: dùng regex gì để detect sentence? Xử lý edge case nào?*

**`RecursiveChunker.chunk` / `_split`** — approach:
> *Viết 2-3 câu: algorithm hoạt động thế nào? Base case là gì?*

### EmbeddingStore

**`add_documents` + `search`** — approach:
> *Viết 2-3 câu: lưu trữ thế nào? Tính similarity ra sao?*

**`search_with_filter` + `delete_document`** — approach:
> *Viết 2-3 câu: filter trước hay sau? Delete bằng cách nào?*

### KnowledgeBaseAgent

**`answer`** — approach:
> *Viết 2-3 câu: prompt structure? Cách inject context?*

### Test Results

```
# Paste output of: pytest tests/ -v
```

**Số tests pass:** __ / __

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | | | high / low | | |
| 2 | | | high / low | | |
| 3 | | | high / low | | |
| 4 | | | high / low | | |
| 5 | | | high / low | | |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:*

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Bao nhiêu queries trả về chunk relevant trong top-3?** __ / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:*

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | / 5 |
| Document selection | Nhóm | / 10 |
| Chunking strategy | Nhóm | / 15 |
| My approach | Cá nhân | / 10 |
| Similarity predictions | Cá nhân | / 5 |
| Results | Cá nhân | / 10 |
| Core implementation (tests) | Cá nhân | / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **/ 100** |
