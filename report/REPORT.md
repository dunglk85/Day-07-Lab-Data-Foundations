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
|---|---|---|---|---|
| 1 | data/phapluat/01_pháp_luật.md | https://vnexpress.net/phap-luat | 10137 | doc_id, source, path, chunk_index |
| 2 | data/phapluat/02_pháp_luật.md | https://vietnamnet.vn/phap-luat | 549 | doc_id, source, path, chunk_index |
| 3 | data/phapluat/03_pháp_luật.md | https://tuoitre.vn/phap-luat.htm | 1664 | doc_id, source, path, chunk_index |

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
|---|---|---|---|---|
| data/phapluat/01_pháp_luật.md | FixedSizeChunker (`fixed_size`) | 41 | 296.0 | Yes |
| data/phapluat/01_pháp_luật.md | SentenceChunker (`by_sentences`) | 20 | 504.2 | Yes |
| data/phapluat/01_pháp_luật.md | RecursiveChunker (`recursive`) | 98 | 101.5 | Partially |
| data/phapluat/02_pháp_luật.md | FixedSizeChunker (`fixed_size`) | 2 | 299.5 | Yes |
| data/phapluat/02_pháp_luật.md | SentenceChunker (`by_sentences`) | 1 | 550.0 | Yes |
| data/phapluat/02_pháp_luật.md | RecursiveChunker (`recursive`) | 37 | 13.8 | Partially |
| data/phapluat/03_pháp_luật.md | FixedSizeChunker (`fixed_size`) | 7 | 280.6 | Yes |
| data/phapluat/03_pháp_luật.md | SentenceChunker (`by_sentences`) | 4 | 414.5 | Yes |
| data/phapluat/03_pháp_luật.md | RecursiveChunker (`recursive`) | 30 | 53.5 | Partially |

### Strategy Của Tôi

**Loại:** [FixedSizeChunker / SentenceChunker / RecursiveChunker / custom strategy]
SentenceChunker

**Mô tả cách hoạt động:**
> *Viết 3-4 câu: strategy chunk thế nào? Dựa trên dấu hiệu gì?*
SentenceChunker tách văn bản thành các câu bằng regex `[^.!?]+[.!?]?`, sau đó gom tối đa 3 câu thành 1 chunk. Cách này dựa vào dấu kết câu (`.`, `!`, `?`) để giữ nghĩa ở mức câu thay vì cắt cứng theo ký tự. Với dữ liệu markdown có tiêu đề ngắn, chunk theo câu giúp giảm tình trạng một chunk chỉ chứa nửa ý. Edge case chính là các câu thiếu dấu kết thúc; regex vẫn lấy được nhờ phần dấu câu là tùy chọn.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Viết 2-3 câu: domain có pattern gì mà strategy khai thác?*
Domain pháp luật/điều khoản cần giữ mạch nghĩa theo câu hoặc cụm câu. Nếu chunk quá ngắn (recursive), ngữ cảnh pháp lý bị vỡ; nếu chunk cố định, câu dễ bị cắt ngang. SentenceChunker cân bằng giữa độ dài chunk và tính mạch lạc nên phù hợp hơn cho truy xuất theo câu hỏi tự nhiên.

**Code snippet (nếu custom):**
```python
# Paste implementation here
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| data/phapluat (3 docs) | best baseline (SentenceChunker) | 26 | ~295.4 | Top-3 relevant: 2/5 |
| data/phapluat (3 docs) | **của tôi (SentenceChunker)** | 26 | ~295.4 | Top-3 relevant: 2/5 |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | SentenceChunker | 4.0 | Chunk dài hơn, giữ ngữ cảnh tốt | Top-1 vẫn nhiễu vì data là trang tin tổng hợp |
| Ngô Gia Bảo | FixedSizeChunker | 2.0 | Triển khai đơn giản, ổn định | Cắt ngang câu, dễ mất ý |
| Nguyễn Dương Ninh | RecursiveChunker | 0.0 | Linh hoạt theo separators | Nhiều chunk quá ngắn, nhiễu retrieval |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:*
Với dữ liệu hiện tại, SentenceChunker là lựa chọn tốt nhất trong 3 baseline vì giữ ngữ cảnh tốt hơn và có tỉ lệ top-3 relevant cao nhất (2/5). Tuy nhiên chất lượng retrieval tổng thể vẫn thấp do nguồn dữ liệu đang là trang tổng hợp tin pháp luật, chưa phải bộ tài liệu điều khoản chuyên sâu.
---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> *Viết 2-3 câu: dùng regex gì để detect sentence? Xử lý edge case nào?*
Dùng regex `[^.!?]+[.!?]?` để tách theo ranh giới câu cơ bản, rồi `strip()` từng câu và bỏ câu rỗng. Sau đó gom theo block `max_sentences_per_chunk` để tạo chunk ổn định.

**`RecursiveChunker.chunk` / `_split`** — approach:
> *Viết 2-3 câu: algorithm hoạt động thế nào? Base case là gì?*
Hàm `chunk` gọi đệ quy `_split` với danh sách separator ưu tiên: `

`, `
`, `. `, ` `, `""`. Base case là khi đoạn hiện tại ngắn hơn `chunk_size` thì trả về luôn; nếu hết separator thì fallback cắt cứng theo độ dài.
### EmbeddingStore

**`add_documents` + `search`** — approach:
> *Viết 2-3 câu: lưu trữ thế nào? Tính similarity ra sao?* Khi add, mỗi Document được chuẩn hóa thành record gồm `id`, `document`, `embedding`, `metadata`. Search tạo embedding cho query, tính cosine similarity bằng `compute_similarity`, rồi sort giảm dần theo score để lấy top-k.

**`search_with_filter` + `delete_document`** — approach:
> *Viết 2-3 câu: filter trước hay sau? Delete bằng cách nào?* Với in-memory, filter metadata được áp dụng trước rồi mới chạy similarity để giảm candidate set. `delete_document` xóa toàn bộ record có `metadata.doc_id` trùng id cần xóa và trả về bool theo số phần tử giảm.

### KnowledgeBaseAgent

**`answer`** — approach:
> *Viết 2-3 câu: prompt structure? Cách inject context?* Agent retrieve top-k chunks từ store, ghép thành `Context` theo từng dòng, rồi build prompt dạng `Context -> Question -> Answer`. Sau đó gọi `llm_fn(prompt)` để sinh câu trả lời.

### Test Results

```
# Paste output of: pytest tests/ -v
```

**Số tests pass:** 41 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Người lái xe có nồng độ cồn vượt mức sẽ bị phạt tiền. | Tài xế vi phạm nồng độ cồn có thể bị xử phạt bằng tiền. | high | 0.0152 | Sai |
| 2 | Thủ tục khởi kiện dân sự tại tòa án gồm nhiều bước. | Cách nấu bún bò Huế ngon tại nhà. | low | -0.2179 | Đúng |
| 3 | Người bị tạm giam có quyền gặp luật sư. | Bị can có thể được gặp người bào chữa trong giai đoạn điều tra. | high | 0.1037 | Đúng |
| 4 | Tranh chấp đất đai cần hòa giải ở xã trước. | Các bên tranh chấp đất thường phải hòa giải tại UBND cấp xã. | high | 0.2029 | Đúng |
| 5 | Lừa đảo chiếm đoạt tài sản có thể bị phạt tù. | Hôm nay thời tiết Hà Nội có mưa rào và dông rải rác. | low | -0.1564 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:*  Cặp 1 bất ngờ nhất vì cùng nghĩa nhưng score thấp do dùng mock embedding. Điều này cho thấy chất lượng biểu diễn nghĩa phụ thuộc mạnh vào embedding model backend.


---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Mức phạt khi vi phạm nồng độ cồn khi lái xe là bao nhiêu? | Người điều khiển xe ô tô có nồng độ cồn vượt mức cho phép có thể bị phạt tiền từ 30 đến 40 triệu đồng và tước giấy phép lái xe từ 22 đến 24 tháng. |
| 2 | Thủ tục khởi kiện dân sự tại tòa án được thực hiện như thế nào? | Người khởi kiện nộp đơn kèm tài liệu chứng cứ tại tòa án có thẩm quyền, tòa án xem xét thụ lý trong vòng 3 ngày làm việc và thông báo nộp tạm ứng án phí. |
| 3 | Tội lừa đảo chiếm đoạt tài sản bị xử lý như thế nào theo Bộ luật Hình sự? | Tội lừa đảo chiếm đoạt tài sản có thể bị phạt tù từ 6 tháng đến 3 năm với trường hợp thông thường, và lên đến 20 năm hoặc tù chung thân nếu chiếm đoạt tài sản có giá trị đặc biệt lớn. |
| 4 | Quyền và nghĩa vụ của người bị tạm giam là gì? | Người bị tạm giam có quyền được thông báo lý do tạm giam, gặp luật sư, nhận đồ tiếp tế từ thân nhân và khiếu nại quyết định tạm giam nếu cho là trái pháp luật. |
| 5 | Tranh chấp đất đai giữa các hộ gia đình được giải quyết theo trình tự nào? | Tranh chấp đất đai phải qua hòa giải tại UBND cấp xã trước, nếu không hòa giải được thì các bên có thể khởi kiện ra tòa án hoặc yêu cầu UBND cấp huyện giải quyết. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |

| 1 | Mức phạt khi vi phạm nồng độ cồn khi lái xe là bao nhiêu? | w=240&h=144&q=100&dpr=1&fit=crop&s=jFtmUClBtzlXHYGifgXMZA&t=image  ### Nhân viên quán ăn bị chĩa súng dọa giết khi rót nhầm nước chấm  AnhM... (src: 01_pháp_luật.md) | 0.1844 | No | Chưa có LLM run trong script này; dùng top-1 chunk làm context dự kiến. |
| 2 | Thủ tục khởi kiện dân sự tại tòa án được thực hiện như thế nào? | ### Giám đốc Công ty Mua bán nợ Nhất Tín bị bắt vì 'khủng bố' để đòi 70 tỷ đồng  TP HCMLợi dụng danh nghĩa hợp đồng mua bán nợ, Hồ Minh Đạt ... (src: 01_pháp_luật.md) | 0.2173 | No | Chưa có LLM run trong script này; dùng top-1 chunk làm context dự kiến. |
| 3 | Tội lừa đảo chiếm đoạt tài sản bị xử lý như thế nào theo Bộ luật Hình sự? | 000 SGD nhưng doanh nghiệp từ chối'  TP HCMCựu lãnh đạo Tập đoàn Cao su Việt Nam khai từng trả lại 200. 000 SGD nhưng doanh nghiệp không nhậ... (src: 01_pháp_luật.md) | 0.3003 | No | Chưa có LLM run trong script này; dùng top-1 chunk làm context dự kiến. |
| 4 | Quyền và nghĩa vụ của người bị tạm giam là gì? | w=240&h=144&q=100&dpr=1&fit=crop&s=jFtmUClBtzlXHYGifgXMZA&t=image)  ### Nhân viên quán ăn bị chĩa súng dọa giết khi rót nhầm nước chấm  AnhM... (src: 01_pháp_luật.md) | 0.2527 | Yes | Chưa có LLM run trong script này; dùng top-1 chunk làm context dự kiến. |
| 5 | Tranh chấp đất đai giữa các hộ gia đình được giải quyết theo trình tự nào? | 000 SGD nhưng doanh nghiệp từ chối'  TP HCMCựu lãnh đạo Tập đoàn Cao su Việt Nam khai từng trả lại 200. 000 SGD nhưng doanh nghiệp không nhậ... (src: 01_pháp_luật.md) | 0.1697 | No | Chưa có LLM run trong script này; dùng top-1 chunk làm context dự kiến. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 2 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:* Học cách ưu tiên separator hợp lý khi recursive chunking để tránh tạo quá nhiều chunk ngắn gây nhiễu.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:*Học cách đánh giá retrieval theo Precision@k và phân tích lỗi theo từng query thay vì chỉ nhìn score trung bình.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:*Tôi sẽ dùng nguồn luật/nghị định chính thức thay cho trang tin tổng hợp, đồng thời gắn metadata điều/khoản/chương để lọc tốt hơn.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 12 / 15 |
| My approach | Cá nhân | 9 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 7 / 10 |
| Core implementation (tests) | Cá nhân | 29 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **79 / 100** |
