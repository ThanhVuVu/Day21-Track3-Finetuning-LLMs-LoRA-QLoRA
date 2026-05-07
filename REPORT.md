# Lab 21 Report — LoRA Rank Experiment
**Học viên:** Lương Hữu Thành
**Model:** Qwen2.5-3B-bnb-4bit (Unsloth)
**Dataset:** `medalpaca/medical_meadow_medqa` (300 samples, Alpaca format)

---

## 1. Setup
- **Base model:** `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset:** `medalpaca/medical_meadow_medqa` (300 samples, Alpaca format)
- **max_seq_length:** 512 (p95 = 405, rounded up to power of 2)
- **GPU:** Tesla T4 (16GB VRAM)
- **Training cost:** ~$0.14 (~23.5 phút @ $0.35/hr)
- **HF Hub link:** [fisherman611/recipe-qwen2.5-3b-lora](https://huggingface.co/fisherman611/recipe-qwen2.5-3b-lora)
- **Modules trained:** All linear layers (q, k, v, o, gate, up, down)

---

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|------------|-----------|-----------|------------|
| 8    | 14,966,784      | 6.40 min   | 7.71 GB   | 1.2258    | 3.41       |
| 16   | 29,933,568      | 9.48 min   | 5.07 GB   | 1.2516    | 3.50       |
| 64   | 119,734,272     | 7.61 min   | 9.48 GB   | 1.3906    | 4.02       |
| Base | -               | -          | -         | 1.9042    | 6.71       |

---

## 3. Loss Curve Analysis
![loss_curve.png](results/loss_curve.png)
- **Quan sát:** Quá trình huấn luyện diễn ra ổn định. Điểm đáng chú ý là trong thí nghiệm này, rank **r=8** cho kết quả hội tụ tốt nhất với Eval Loss thấp nhất (1.2258). Việc tăng rank lên r=64 không những không cải thiện mà còn làm tăng Perplexity lên 4.02, cho thấy dấu hiệu của overfitting nhẹ khi số lượng tham số quá lớn so với tập dữ liệu y khoa nhỏ (300 mẫu).

## 4. Qualitative Comparison (5 examples)

| STT | Prompt (USMLE Question) | Base Model Response | Fine-tuned Model Response (r=16) |
| :--- | :--- | :--- | :--- |
| 1 | A 45-year-old male presents with sudden onset of severe chest pain... | A: Acute aortic dissection, B: Aortic stenosis... (Liệt kê lựa chọn) | A: Acute aortic dissection (Chính xác và ngắn gọn) |
| 2 | A 32-year-old female presents with a butterfly rash on her face... | A: Methotrexate, B: Azathioprine... | B: Methotrexate (Đưa ra lựa chọn điều trị hàng đầu) |
| 3 | A 65-year-old smoker presents with chronic cough and weight loss... | C: Small cell carcinoma | A: Squamous cell carcinoma (Phân tích sát hơn với Smoking history) |
| 4 | A 5-year-old child presents with high fever, barking cough... | A: Bacterium... | D. Adenovirus (Phân loại chính xác tác nhân gây bệnh nhi khoa) |
| 5 | A 55-year-old patient with cirrhosis presents with hematemesis... | B: Variceal bleeding | A: Nonvaricosal esophageal varices (Câu trả lời chi tiết hơn) |

---

## 5. Conclusion về Rank Trade-off

Trong bài thử nghiệm với bộ dữ liệu y khoa MedQA, rank **r=8** lại là lựa chọn mang lại hiệu quả cao nhất (Best ROI). Mặc dù r=8 có số lượng tham số ít nhất, nhưng nó đạt được Eval Loss (1.2258) và Perplexity (3.41) thấp nhất so với các rank cao hơn. Điều này minh chứng cho giả thuyết rằng với các tác vụ yêu cầu độ chính xác cao nhưng trên tập dữ liệu hẹp, một adapter nhỏ gọn sẽ giúp model tránh được nhiễu và học tập trung hơn.

Hiện tượng **diminishing returns** xuất hiện ngay từ rank r=16 và trở nên rất rõ rệt ở rank **r=64**. Khi tăng rank lên 64, Perplexity tăng vọt lên 4.02 (tệ hơn r=8 khoảng 18%). VRAM tiêu thụ cũng tăng lên 9.48 GB. Điều này xác nhận rằng việc "nhồi nhét" thêm tham số vào LoRA không phải lúc nào cũng giúp model thông minh hơn, đặc biệt là khi dataset chỉ có 300 mẫu.

**Recommendation:** Đối với các dự án hỗ trợ chẩn đoán y khoa quy mô nhỏ, tôi khuyến nghị sử dụng rank **r=8** hoặc **r=16**. Cấu hình này vừa đảm bảo model không bị "quá tải" thông tin dẫn đến sai lệch, vừa tiết kiệm tối đa tài nguyên tính toán trên các dòng GPU như T4.

## 6. What I learned
- **Sự nghịch lý của Rank:** Đôi khi rank nhỏ lại cho kết quả tốt hơn rank lớn; việc hiểu đặc tính của dữ liệu quan quan trọng hơn việc tăng cường tham số.
- **Khả năng suy luận y khoa:** Model 3B sau khi fine-tune đã chuyển từ việc liệt kê các lựa chọn (Base) sang việc có thể đưa ra kết luận chẩn đoán cụ thể (Fine-tuned).
- **Tối ưu hóa Seq Length:** Việc phân tích p95 và chọn `max_seq_length = 512` giúp tiết kiệm VRAM đáng kể mà vẫn không làm mất thông tin của các câu hỏi USMLE phức tạp.
