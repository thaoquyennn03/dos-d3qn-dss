@'
# DOS-aware D3QN – Hệ hỗ trợ quyết định cho shipper giao đồ ăn

Mã nguồn, dữ liệu mô phỏng và kết quả thực nghiệm của tiểu luận môn **Hệ hỗ trợ quyết định (CS2208) – UIT**.

- Học viên: Trần Thảo Quyên – MSHV 26848185
- Giảng viên: TS. Tôn Thất Hòa An

## Bài toán

Đồ án dựa trên bài báo của Zhou, Fotouhi & Miller-Hooks (2025), *Decision Support Systems* 190, 114388. Bài báo dùng D3QN để giúp shipper quyết định nhận/từ chối đơn và di chuyển khi rảnh, nhằm tối đa thu nhập.

**Hướng mở rộng:** trạng thái của D3QN được bổ sung chỉ số cơ hội tại điểm đến **DOS = D̂ / (Ĉ + 1)**, trong đó D̂ là số đơn gần đây và Ĉ là số shipper cạnh tranh. DOS được tính tại 6 điểm, với cửa sổ 30 phút và bán kính 2,5 km.

**Giả thuyết:** thêm DOS giúp D3QN đánh giá đơn tốt hơn, nhất là khi nhu cầu biến động theo không gian và thời gian.

## Kết quả chính

| Kịch bản (dư đơn · điểm nóng ngẫu nhiên, 1.500 ca) | Thu nhập ròng / ca (CNY) |
|---|---|
| D3QN gốc | 211,8 ± 5,6 |
| D3QN + DOS | 220,0 ± 6,6 |
| D3QN + nhiễu (đối chứng) | 218,2 ± 3,4 |
| Luật ngưỡng | 237,6 ± 12,0 |

- D3QN gốc thắng Centroid ở cả 9 kịch bản.
- D3QN + DOS hơn D3QN gốc 3,9%, nhưng chênh lệch không có ý nghĩa thống kê (p = 0,135). Nhánh nhiễu cũng hơn 3,0%.
- Khi dư đơn, luật ngưỡng "nhận nếu lợi nhuận/phút ≥ θ" hơn D3QN 10,9–13,3%.

Kết quả đầy đủ xem trong `results/KET_QUA.md`, `results/summary.csv`, `results/paired_tests.csv` và `results/budget_1500ep/`.

## Chạy lại

Cần Python 3.10 trở lên. Không cần GPU, vì D3QN được viết bằng NumPy.

```bash
git clone https://github.com/thaoquyennn03/dos-d3qn-dss.git
cd dos-d3qn-dss
pip install -r requirements.txt

python src/run_experiments.py --plan smoke                # kiểm tra cài đặt (~1 phút)
python src/run_experiments.py --plan core --episodes 600  # các lượt tạo ra results/
python src/analyze.py                                     # tạo lại bảng và biểu đồ
```

Hướng dẫn chi tiết (chạy trong VS Code, giải thích các nhánh thực nghiệm, khác biệt so với bài báo) xem trong `README_VI.md`.

## Cấu trúc
