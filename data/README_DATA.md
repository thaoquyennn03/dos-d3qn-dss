# Dataset v2 cho hướng DOS (100% mô phỏng)

Dataset này được sinh bằng `src/data_gen.py` với seed 20261108, dựa trên dataset mô phỏng gốc của bạn.
**Đây không phải dữ liệu Meituan.**

## Giữ nguyên từ dataset gốc
- 30 nhà hàng: vị trí, độ phổ biến (tỉ lệ số đơn trong `orders.csv` gốc), thời gian chuẩn bị (mean, sd, chặn 5–40 phút).
- Phí giao = 4 + 0,5 × km (nhà hàng → khách). Chi phí = 0,9 CNY/km. Tip có độ lệch chuẩn 2.
- Trọng số nhu cầu theo giờ [2.5, 3, 4, 7, 6, 4, 3, 2]. Lưới 10 × 10 km. Tốc độ 28 km/h.

## Thay đổi để phục vụ hướng DOS
| Thay đổi | Lý do |
|---|---|
| 540 ngày thay vì 1 ngày | Cần nhiều ngày để huấn luyện và để có ngày test riêng |
| 3 mức nhu cầu: few 150, sufficient 400, excess 1.200 đơn/ngày (với 30 đối thủ) | Giả thuyết H4. Với Centroid, thời gian rảnh mỗi ca ≈ 294 / 156 / 58 phút |
| 3 kiểu nhu cầu: `stationary`, `peak_fixed`, `hotspot_random` | Giả thuyết H1 và H2. Trong mỗi khung 2 giờ, 6/30 nhà hàng có nhu cầu ×4: cố định qua các ngày (`peak_fixed`), hoặc chọn ngẫu nhiên lại mỗi ngày (`hotspot_random`) |
| Tip trung bình riêng theo nhà hàng (2,5–6,5 CNY) | Fig. 6 của paper cho thấy tip khác nhau theo nhà hàng; dataset gốc chỉ có một phân phối chung N(4.5, 2) |
| Chia 40 ngày train / 20 ngày test cho mỗi kịch bản | Mọi policy được đánh giá trên cùng các ngày test |
| Bỏ `order_offers`, `mdp_transitions`, `batch_pair_candidates` | Offer và transition do simulator sinh ra theo luật phân đơn của paper. Ghép 2 đơn nằm ngoài phạm vi đề tài |

## Các file
| File | Khóa | Nội dung |
|---|---|---|
| `source_restaurants.csv` | restaurant_id | Hồ sơ trích từ `orders.csv` gốc |
| `restaurants.csv` | restaurant_id | Như trên, thêm cột `tip_mean_cny` |
| `riders.csv` | rider_id | Vị trí đầu ca; R000 là shipper chính, còn lại là đối thủ (dùng 30 người đầu) |
| `days.csv` | day_id | Mức nhu cầu, kiểu nhu cầu, split train/test, seed, số đơn |
| `orders.csv` | order_id | day_id, thời điểm tạo đơn, vị trí nhà hàng/khách, km, thời gian chuẩn bị, ready_min, phí, tip |
| `demand_profile.csv` | (day_id, block_2h, restaurant_id) | Hệ số nhu cầu thật của từng ngày. **Chỉ nhánh oracle (A3) được dùng file này** |

Định dạng `day_id`: `<level>-<pattern>-<split>-<số thứ tự>`, ví dụ `exc-hots-test-003`.
Muốn sinh lại dataset hoặc đổi kích thước: `python src/data_gen.py --train-days 40 --test-days 20 --seed 20261108`.
