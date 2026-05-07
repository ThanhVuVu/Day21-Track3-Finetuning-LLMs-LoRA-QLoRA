# Lab 21 Report — LoRA Rank Experiment
**Học viên:** Vũ Phúc Thành
**Model:** Qwen2.5-3B-bnb-4bit (Unsloth)
**Dataset:** `duyet/vietnamese-legal-instruct` (300 samples, Alpaca format)

---

## 1. Setup
- **Base model:** `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset:** `duyet/vietnamese-legal-instruct` (300 samples, Alpaca format)
- **max_seq_length:** 1024 (p95 = 2525, rounded up to power of 2 but capped at 1024 for T4 compatibility)
- **GPU:** Tesla T4 (16GB VRAM)
- **Training cost:** ~$0.37 (~63.7 phút @ $0.35/hr)
- **HF Hub link:** [ThanhVuVu/vietnamese-legal-qwen2.5-3b-lora](https://huggingface.co/ThanhVuVu/vietnamese-legal-qwen2.5-3b-lora)
- **Modules trained:** All linear layers (q, k, v, o, gate, up, down)

---

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|------------|-----------|-----------|------------|
| 8    | 14,966,784      | 20.1 min   | 9.30 GB   | 0.8419    | 2.32       |
| 16   | 29,933,568      | 20.6 min   | 8.83 GB   | 0.8400    | 2.32       |
| 64   | 119,734,272     | 23.0 min   | 11.06 GB  | 0.8525    | 2.35       |
| Base | -               | -          | -         | 1.1339    | 3.11       |

---

## 3. Loss Curve Analysis
![loss_curve.png](results/loss_curve.png)
- **Quan sát:** Loss curve giảm ổn định từ ~1.1 xuống ~0.7 sau 3 epochs. Không có dấu hiệu overfitting rõ rệt vì training loss và eval loss khá sát nhau. Việc sử dụng `r=16` cho thấy sự hội tụ tốt nhất trong 3 rank thử nghiệm với Eval Loss thấp nhất (0.8400).

## 4. Qualitative Comparison (5 examples)

| STT | Prompt (Câu hỏi Luật) | Base Model Response | Fine-tuned Model Response |
| :--- | :--- | :--- | :--- |
| 1 | Thế nào là hành vi vi phạm pháp luật dân sự? | Hành vi vi phạm pháp luật dân sự là hành vi cụ thể, có dấu hiệu vi phạm... | Hành vi vi phạm pháp luật dân sự là hành vi vi phạm pháp luật về quyền và nghĩa vụ... |
| 2 | Hợp đồng lao động có thể giao kết bằng lời nói không? | Hợp đồng lao động không bắt buộc phải viết bằng văn bản... | Trong trường hợp người lao động, người sử dụng lao động không ký hợp đồng... |
| 3 | Thời gian thử việc tối đa cho cao đẳng trở lên? | ## Thời gian thử việc tối đa đối với công việc có chức danh... | ## Thời gian thử việc tối đa đối với công việc có chức danh... (Kèm trích dẫn chi tiết Bộ luật lao động) |
| 4 | Quyền đơn phương chấm dứt HĐLĐ không báo trước? | Tình trạng: Hết hiệu lực... (Trả lời chung chung) | Nếu bạn có một trong những lý do sau, bạn có quyền đơn phương... (Chi tiết điều khoản) |
| 5 | Thủ tục đăng ký kết hôn cần giấy tờ gì? | ## Thủ tục đăng ký kết hôn... (UBND cấp xã) | ## Thủ tục đăng ký kết hôn... (Hướng dẫn chi tiết đối tượng, cơ quan và các loại giấy tờ tùy thân) |

---

## 5. Conclusion về Rank Trade-off

Dựa trên kết quả thực nghiệm từ notebook, rank **r=16** cho ROI (Return on Investment) tốt nhất trên bộ dữ liệu pháp luật Việt Nam. Với rank r=16, model đạt Eval Loss thấp nhất (0.8400) và Perplexity tối ưu (2.32) trong khi vẫn duy trì mức tiêu thụ VRAM thấp (8.83 GB).

Hiện tượng **diminishing returns** xuất hiện rõ rệt khi tăng rank lên **r=64**. Tại mức rank này, mặc dù số lượng tham số huấn luyện tăng gấp 4 lần, nhưng Eval Loss lại tăng lên (0.8525) và Perplexity cũng cao hơn. Điều này cho thấy với tập dữ liệu đặc thù và kích thước vừa phải (300 mẫu), việc tăng rank quá cao không giúp model học tốt hơn mà trái lại có thể gây nhiễu.

**Recommendation:** Nếu triển khai thực tế, tôi khuyến nghị chọn rank **r=16**. Đây là mức cân bằng hoàn hảo giữa khả năng hội tụ, tốc độ huấn luyện và hiệu quả sử dụng tài nguyên phần cứng.

## 6. What I learned
- **Sự phù hợp của Rank:** Hiểu rõ rằng rank lớn (r=64) không phải lúc nào cũng tốt hơn rank trung bình (r=16) đối với các domain-specific dataset nhỏ.
- **Tối ưu VRAM:** Kỹ thuật QLoRA và Unsloth giúp việc huấn luyện model 3B trên Tesla T4 trở nên cực kỳ mượt mà, ngay cả với seq length dài (1024).
- **Độ chính xác Domain:** Dữ liệu pháp luật giúp model chuyển từ trả lời mang tính liệt kê sang trả lời có trích dẫn điều khoản và cấu trúc Markdown chuyên nghiệp.
