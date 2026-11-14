# Thử nghiệm sơ bộ (pilot)

Thử nghiệm sơ bộ kiểm tra nhanh: chỉ số DOS có giúp một luật chọn đơn đơn giản kiếm được nhiều tiền hơn không,
trước khi huấn luyện D3QN. Kết quả tương ứng Bảng 4.2 trong báo cáo.

- `sim.py`: môi trường mô phỏng rút gọn theo bài báo (lưới 10 × 10 dặm, 10 nhà hàng, ca 8 giờ, phân đơn mỗi 2 phút
  cho shipper rảnh gần nhất, 8 shipper đối thủ dùng chính sách Centroid và nhận 80% đơn).
- `experiment.py`: chọn tham số θ, λ trên 80 ca huấn luyện, rồi đánh giá 4 chính sách
  (Centroid, Ngưỡng, Ngưỡng + DOS, Ngưỡng + DOS + tái định vị) trên 400 ca kiểm tra.
- `results.json`, `results_full.jsonl`: kết quả đã chạy.

Chạy lại (từ thư mục gốc của repo):

```bash
cd pilot
python experiment.py
```
