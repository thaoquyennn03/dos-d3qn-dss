# Kết quả đã chạy (14/11/2026, máy 2 nhân CPU)

Thiết lập: 30 đối thủ, 600 ca huấn luyện mỗi lượt, 3 seed. Đánh giá trên 20 ngày test của mỗi kịch bản; mọi policy dùng chung các ngày này.
Đơn vị: thu nhập ròng mỗi ca 8 giờ (CNY), trung bình trên 20 ngày test (D3QN: trung bình thêm trên 3 seed).
File gốc: `summary.csv`, `paired_tests.csv` (paired t-test theo ngày, hiệu chỉnh Holm), `convergence.csv`.

## Thu nhập ròng mỗi ca (CNY)

| Kịch bản | A0 D3QN gốc | A2 D3QN + DOS | B0 Centroid | B1 Ngưỡng | B2 Ngưỡng + DOS |
|---|---|---|---|---|---|
| ít đơn · tĩnh | 32,4 | 27,2 | −14,5 | −5,4 | −11,5 |
| ít đơn · cao điểm cố định | 28,1 | 23,1 | −10,1 | −40,4 | −28,5 |
| ít đơn · điểm nóng ngẫu nhiên | 27,7 | 21,0 | −12,0 | −16,4 | −19,0 |
| đủ đơn · tĩnh | 74,4 | 69,0 | 26,2 | 40,5 | 39,8 |
| đủ đơn · cao điểm cố định | 80,2 | 73,0 | 16,9 | 32,7 | 31,6 |
| đủ đơn · điểm nóng ngẫu nhiên | 74,7 | 68,5 | 28,3 | 39,7 | 32,1 |
| dư đơn · tĩnh | 205,7 | 198,4 | 87,9 | 237,4 | 252,0 |
| dư đơn · cao điểm cố định | 213,8 | 217,3 | 98,4 | 239,8 | 244,0 |
| dư đơn · điểm nóng ngẫu nhiên | 206,4 | 196,2 | 91,8 | 237,6 | 241,5 |

Ở hai kịch bản dư đơn (tĩnh / điểm nóng ngẫu nhiên), thêm các nhánh kiểm soát:
A1 D̂ 192,3 / 193,6 · A3 oracle 199,0 / 200,0 · A4 nhiễu 186,4 / 181,4.

## Kiểm tra ngân sách huấn luyện: 1.500 ca, dư đơn · điểm nóng ngẫu nhiên (`budget_1500ep/`)

| Nhánh | Thu nhập (± CI95) |
|---|---|
| A0 D3QN gốc | 211,8 ± 5,6 |
| A2 D3QN + DOS | 220,0 ± 6,6 |
| A4 D3QN + nhiễu | 218,2 ± 3,4 |
| B1 Ngưỡng | 237,6 ± 12,0 |

## Đọc kết quả

1. **D3QN gốc vượt Centroid ở cả 9 kịch bản**, khớp với kết luận của paper.
2. **Khi dư đơn, luật ngưỡng thắng D3QN** khoảng 11–13% (có ý nghĩa ở 2/3 kịch bản dư đơn). Ngược lại, khi ít/đủ đơn D3QN thắng rất xa.
   Đây đúng là baseline mà paper thiếu.
3. **DOS chưa cải thiện D3QN có ý nghĩa thống kê ở bất kỳ kịch bản nào.**
   - Với 600 ca, mọi nhánh có thêm input đều thấp hơn A0, kể cả nhánh nhiễu (−9 đến −12%, có ý nghĩa).
     Thứ tự giữa các nhánh rất nhất quán: oracle > DOS > D̂ > nhiễu.
   - Với 1.500 ca, cái giá của việc thêm input biến mất: DOS +3,9% so với A0 (p_Holm = 0,135), nhưng nhánh nhiễu cũng +3,0%.
   - Như vậy, trong simulator này, D3QN gốc đã gần như tự học được phần thông tin mà DOS mang lại.
4. **Ngưỡng + DOS so với Ngưỡng:** +1,6% đến +6,1% khi dư đơn, chưa có ý nghĩa với 20 ngày test.
   Pilot trước đó dùng 400 ngày test thì thấy +4,9% / +7,5% có ý nghĩa. Muốn khẳng định cần nhiều ngày test hơn.
5. **Độ mạnh thống kê:** khoảng tin cậy hiện tại là ±3–4%. Để phát hiện chắc chắn một hiệu ứng cỡ 4% cần khoảng 5 seed × 50 ngày test
   (`python src/data_gen.py --test-days 50`, rồi chạy `--plan full --episodes 1500`).
