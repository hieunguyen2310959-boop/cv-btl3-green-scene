# Green Scene — CO3057 BTL3

Pipeline phân tích cảnh đường phố và vùng cây xanh: tiền xử lý RGB → hình học/đặc trưng → panorama (tùy chọn) → phân vùng → metric và trực quan hóa.

**Sản phẩm:** code Python, demo Streamlit, bộ test, thực nghiệm trên CamVid có nhãn, báo cáo PDF tiếng Việt sinh từ kết quả chạy thật. Không cần Docker hoặc GPU.

## Đọc kết quả trước

- **Báo cáo:** [`reports/Bao-cao-BTL3-CV.pdf`](reports/Bao-cao-BTL3-CV.pdf).
- [`reports/results/summary.json`](reports/results/summary.json): số liệu thật và môi trường.
- [`docs/VERIFICATION_BTL3.md`](docs/VERIFICATION_BTL3.md): các cổng kiểm chứng.
- [`THIRD_PARTY.md`](THIRD_PARTY.md): nguồn dữ liệu và phạm vi tái sử dụng.

### Phạm vi và giới hạn quan trọng

- CamVid: **40 frame thực có nhãn**, 20 dev và 20 test, tách theo video sequence. Không phải 40 môi trường độc lập.
- Nhãn dương là **Tree=5**, bỏ qua Unlabelled=11. Không tuyên bố nhận mọi loài thực vật hoặc đo diện tích ngoài thực địa.
- HSV và ExG là **baseline màu**, chất lượng semantic còn thấp. Macro IoU test lần chạy đã ghi nhận: HSV khoảng **0.1089**, ExG khoảng **0.1547**. Không dùng như công cụ đo cây xanh chính xác.
- Panorama ORB/SIFT được đo trên **crop chồng lấn từ cùng một ảnh thực**, không phải các góc camera độc lập. Lỗi góc có ground truth translation biết trước; kết quả không chứng minh robustness với parallax/chuyển động.
- Demo nhận ảnh riêng để nhóm thử camera, nhưng không có kết quả benchmark camera độc lập trong báo cáo.
- Không sửa/push lên repo BTL1 hoặc BTL2; implementation mới dùng OpenCV/NumPy, có AI hỗ trợ.

## Cài đặt Windows

Cần Git và [uv](https://docs.astral.sh/uv/) trên PATH. Từ thư mục project, chạy trong PowerShell:

```powershell
cmd /c setup.cmd
.\.venv\Scripts\python.exe scripts/download_data.py
.\.venv\Scripts\python.exe -m pytest tests -q
```

`setup.cmd` tạo môi trường Python 3.13 riêng, cài đúng `requirements-btl3.lock`, rồi cài package editable. Không cài vào Python toàn máy. Ảnh/nhãn tải trực tiếp từ commit cố định của SegNet-Tutorial; raw data bị git-ignore. Cần mạng lúc cài và tải dữ liệu lần đầu.

Linux/macOS (chưa có chứng nhận thực thi trong lần bàn giao Windows):

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements-btl3.lock
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python scripts/download_data.py
.venv/bin/python -m pytest tests -q
```

## Chạy một cảnh

```powershell
# Một ảnh thực: edges/lines/corners + phân vùng + đối chiếu nhãn Tree
.\.venv\Scripts\python.exe -m cv_final.runner --scene 0001TP_008550 --config configs/selected.json --output outputs/my-scene

# Panorama từ crop có kiểm soát (không phải camera views)
.\.venv\Scripts\python.exe -m cv_final.runner --scene 0001TP_008550 --controlled-crops --config configs/selected.json --output outputs/my-crops

# Ảnh tự chụp, trái -> phải; cần cùng cảnh, đủ texture và overlap
.\.venv\Scripts\python.exe -m cv_final.runner --views "left.jpg" "middle.jpg" "right.jpg" --config configs/selected.json --output outputs/my-camera
```

Output khi thành công: `status.json`, `config.json`, `geometry_metrics.json`, `panorama.png`, `valid_mask.png`, `vegetation_mask.png`, `overlay.png`, `edges.png`, `lines.png`, `inputs.png`. Nếu có GT: thêm `evaluation.json`, `ground_truth.png`.

Khi hình học thất bại: status failed, lý do rõ, mã thoát CLI **2**, không có ảnh panorama giả thành công. Mã thoát thành công **0**. Rerun chỉ dọn các artifact do runner sở hữu; thư mục có file lạ bị từ chối để tránh xóa dữ liệu người dùng. Dùng thư mục output riêng.

## Demo tương tác

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Mở URL local được in ra. Chọn cảnh/method → Chạy phân tích → xem ảnh, hình học, mask, metric và tải JSON. Ảnh upload được lưu local trong `.cache/uploads/`; không gửi lên inference API. Không bật public binding cho demo khi chưa bổ sung kiểm soát upload/tài nguyên.

## Tái sinh thực nghiệm và báo cáo

```powershell
.\.venv\Scripts\python.exe scripts/run_experiments.py
.\.venv\Scripts\python.exe scripts/build_report.py
```

- 29 cấu hình HSV/ExG được thử **chỉ trên dev**; khóa tham số rồi mới đo test.
- Split dev: `0006R0`, `0016E5`; test: `0001TP`, `Seq05VD`.
- Đánh giá cả 20 frame test với mỗi baseline; panorama 20 bộ crop với mỗi ORB/SIFT. Không bỏ cảnh thất bại khỏi mẫu số.
- `reports/results/` gồm aggregate, từng frame, từng cặp geometry và các trial dev.
- Timing là một lượt đo local, không cam kết benchmark tốc độ ổn định.
- PDF dùng font Arial trên Windows; Linux cài DejaVu Sans hoặc đặt `CV_REPORT_FONT_DIR`.
- Lệnh thực nghiệm ghi đè kết quả sinh tự động trong `outputs/experiments/` và `reports/`; không để file thủ công quan trọng tại đó.

## Cấu trúc

```text
src/cv_final/      preprocess, geometry, segmentation, evaluate, runner
configs/          baseline mặc định và cấu hình đã chọn từ dev
data/manifest.json  split, URL, class mapping và SHA256
scripts/          download_data, run_experiments, build_report
tests/            unit/integration/demo smoke tests
app.py            demo Streamlit
reports/          PDF, Markdown, figures, JSON/CSV đo thật
docs/             verification và hướng dẫn bảo vệ
```

## Tài liệu lịch sử trên máy

`plan.md`, `VERIFICATION_REPORT.md`, `verify.cmd`, `scripts/verify_sources.py`, `requirements.in`, `requirements.lock.txt` thuộc lượt **kiểm chứng BTL1/BTL2 trước khi làm BTL3**, không phải workflow hiện tại. Không cần clone upstream để chạy BTL3. Các file này được giữ local để đối chiếu, không nằm trong bộ nộp Git mới.

## Thành viên và minh bạch

Nhóm 02: Hứa Thanh Tâm (2313026), Hoàng Ngô Thiên Phúc (2212612), Nguyễn Minh Hiếu (2310959), Nguyễn Thị Kim Thanh (2433204). Bảng bàn giao trong báo cáo là **phân công đề xuất để nhóm review/xác nhận**, không phải khẳng định mỗi người đã viết code. Code và tài liệu có AI hỗ trợ; nhóm cần tự chạy lại, hiểu thuật toán, xác nhận đóng góp thực tế và kiểm tra quy định AI của môn học trước khi nộp.
