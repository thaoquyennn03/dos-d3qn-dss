# Destination Opportunity-Aware D3QN (DOS) — mã nguồn và dữ liệu

Hướng phát triển dựa trên Zhou, Fotouhi & Miller-Hooks (2025), *Decision Support Systems* 190, 114388.
Bài toán, thuật toán D3QN, hàm thưởng và tập hành động giữ nguyên như paper. Phần thay đổi duy nhất
là state có thêm 6 đặc trưng DOS.

## 1. Chạy trong VS Code

**Cần có:** Python 3.10 trở lên, VS Code, và extension **Python** (Microsoft).
Không cần GPU hay PyTorch: D3QN được viết bằng NumPy.

1. Giải nén `dos_d3qn.zip`, rồi trong VS Code chọn **File → Open Folder…** và mở thư mục `dos_d3qn`.
2. Tạo môi trường ảo: `Ctrl+Shift+P` → **Python: Create Environment** → **Venv** → chọn Python 3.10+
   → tick `requirements.txt`. VS Code sẽ tự cài thư viện.
   Nếu muốn dùng terminal (`` Ctrl+` ``) thay vì giao diện:
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Mở tab **Run and Debug** (`Ctrl+Shift+D`) và chọn một cấu hình có sẵn:
   - **2. Smoke test**: chạy khoảng 1 phút, dùng để kiểm tra cài đặt đã đúng chưa.
   - **3. Train 1 nhánh**: huấn luyện D3QN + DOS cho kịch bản dư đơn, điểm nóng ngẫu nhiên.
   - **4. Chạy kế hoạch core**: chính là các lượt đã tạo ra `results/` trong gói này.
   - **5. Chạy kế hoạch full**: thiết kế đầy đủ.
   - **6. Phân tích kết quả**: tạo lại bảng và biểu đồ.

Cũng có thể chạy bằng dòng lệnh, đặt thư mục làm việc là `dos_d3qn`:
```bash
python src/run_experiments.py --plan smoke
python src/train.py --level excess --pattern hotspot_random --feature dos --seed 0 --episodes 600
python src/evaluate.py --level excess --pattern hotspot_random
python src/analyze.py
python src/run_experiments.py --plan core --episodes 600 --jobs 4     # --jobs = số nhân CPU muốn dùng
```
Script tự bỏ qua những model đã huấn luyện xong (file `.npz` đã có trong `results/models`).
Vì vậy nếu bị dừng giữa chừng, chỉ cần chạy lại lệnh cũ để tiếp tục.
Muốn huấn luyện lại từ đầu thì đổi tên hoặc chuyển thư mục `results/` đi chỗ khác.

**Thời gian chạy tham khảo** (đo trên máy 2 nhân, mỗi lượt 600 ca): khoảng 1–2,5 phút mỗi lượt huấn luyện.
Kế hoạch core gồm 72 lượt, mất khoảng 1 giờ với `--jobs 2`.
Kế hoạch full gồm 225 lượt, cần vài giờ; trên máy 8 nhân, dùng `--jobs 7` sẽ nhanh hơn nhiều.

## 2. Cấu trúc thư mục

```
dos_d3qn/
├─ data/                  dataset v2 (xem data/README_DATA.md)
├─ src/
│  ├─ config.py           hằng số: lưới, tốc độ, chi phí, mức nhu cầu, thông số DOS
│  ├─ data_gen.py         sinh dataset v2 từ hồ sơ 30 nhà hàng của dữ liệu gốc
│  ├─ env.py              môi trường shipper theo Mục 3 của paper + 5 bộ đặc trưng
│  ├─ agent.py            D3QN (Dueling + Double DQN) bằng NumPy, siêu tham số theo Bảng B-1
│  ├─ baselines.py        Centroid, luật ngưỡng, luật ngưỡng + DOS
│  ├─ train.py            huấn luyện 1 nhánh trên 1 kịch bản
│  ├─ evaluate.py         đánh giá trên các ngày test (mọi policy dùng chung các ngày test)
│  ├─ analyze.py          bảng tổng hợp, paired t-test + Holm, tốc độ hội tụ, biểu đồ
│  └─ run_experiments.py  chạy cả kế hoạch smoke / core / full
├─ results/               model, learning curve, eval_*.csv, summary.csv, paired_tests.csv, PNG
└─ .vscode/               cấu hình Run & Debug
```

## 3. Các nhánh thực nghiệm

| Nhánh | Mô tả | Giả thuyết kiểm tra |
|---|---|---|
| A0_d3qn | D3QN gốc: state = vị trí shipper, vị trí khách, phí, tip, thời gian, one-hot nhà hàng | Mốc so sánh |
| A1_d3qn_dhat | A0 + số đơn gần đây D̂ tại 6 điểm | H3 (vai trò của Ĉ) |
| A2_d3qn_dos | A0 + DOS = D̂/(Ĉ+1) tại 6 điểm | H1, H2, H4 |
| A3_d3qn_oracle | A0 + nhu cầu kỳ vọng thật/(Ĉ+1) | Cận trên |
| A4_d3qn_noise | A0 + 6 số ngẫu nhiên | Kiểm soát việc mở rộng input |
| B0_centroid | Nhận mọi đơn, khi rảnh đi về tâm nhà hàng | Chính sách tĩnh tốt nhất của paper |
| B1_threshold | Nhận nếu lợi nhuận/phút ≥ θ | Baseline paper không có |
| B2_threshold_dos | Nhận nếu lợi nhuận/phút + λ·ΔDOS ≥ θ | Phiên bản DSS dễ diễn giải |

6 điểm tính DOS gồm: vị trí hiện tại, điểm giao (khách) và 4 ô lân cận cách 1,5 km.
DOS dùng số đơn trong 30 phút gần nhất và số shipper rảnh trong bán kính 2,5 km (Manhattan).

## 4. Khác biệt so với paper
- Đơn vị km và CNY theo dữ liệu gốc của bạn; paper dùng dặm và USD.
  Kết quả chỉ dùng để so sánh tương đối giữa các nhánh.
- Mạng dueling dùng thân chung 2 lớp × 70 neuron rồi tách đầu V và A (mỗi đầu thêm 1 lớp × 70).
  Paper dùng hai mạng tách hẳn. Các siêu tham số khác giữ đúng Bảng B-1.
- Đối thủ (30 shipper) luân phiên 4 chính sách tĩnh của paper và nhận 80% đơn.
  Không để tất cả cùng chạy về tâm, vì khi đó họ chồng lên một điểm và lợi thế của shipper chính bị phóng đại giả tạo.
- `--reject-penalty` thêm phạt cho mỗi lần từ chối, dùng làm ràng buộc tỉ lệ từ chối.
  Mặc định là 0, tức giống paper.

## 5. Kết quả đã chạy sẵn

Xem `results/KET_QUA.md`. Tóm tắt:

- D3QN gốc thắng Centroid ở cả 9 kịch bản.
- Khi dư đơn, luật ngưỡng thắng D3QN khoảng 11–13%.
- DOS chưa cải thiện D3QN có ý nghĩa thống kê. Với 1.500 ca huấn luyện, DOS hơn D3QN gốc +3,9% (không có ý nghĩa), trong khi nhánh nhiễu cũng hơn +3,0%.
- Để phát hiện chắc chắn một hiệu ứng cỡ 4% cần khoảng 5 seed × 50 ngày test.
