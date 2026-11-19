# Green Scene — CO3057, nhóm 02

Green Scene dùng HSV và ExG để khoanh vùng xanh trong ảnh đường phố. Demo cho xem mặt nạ, ảnh tô vùng dự đoán và tỷ lệ pixel được chọn. Nếu có nhiều ảnh chồng lấn, có thể thử ghép bằng ORB hoặc SIFT.

## Chạy demo trên Windows

Cài [uv](https://docs.astral.sh/uv/getting-started/installation/). Mở PowerShell trong thư mục repo và chạy:

```powershell
cmd /c setup.cmd
.\.venv\Scripts\python.exe scripts/download_data.py
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Mở địa chỉ Streamlit in ra trong PowerShell. Chọn ảnh `Seq05VD_f01140` và bấm **Chạy phân tích** với HSV. Sau đó đổi sang ExG và chạy lại để so sánh trên cùng ảnh. Muốn thử ORB và SIFT, chọn chế độ panorama từ ảnh cắt. Nếu dùng ảnh riêng, tải ảnh lên theo thứ tự trái sang phải.

## Chạy từ dòng lệnh

```powershell
.\.venv\Scripts\python.exe -m cv_final.runner --scene Seq05VD_f01140 --config configs/selected.json --output outputs/sample
```

Kết quả nằm trong thư mục `outputs/sample`, gồm ảnh, mặt nạ và số liệu JSON. Thêm `--controlled-crops` để thử ghép panorama từ cùng một ảnh, hoặc thay `--scene Seq05VD_f01140` bằng `--views left.jpg right.jpg` để dùng ảnh riêng. Dùng một thư mục đầu ra riêng cho mỗi ví dụ.

## Kiểm thử và thực nghiệm

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts/run_experiments.py
```

Lệnh thứ hai chạy lại thực nghiệm và cập nhật cấu hình đã chọn, bảng số liệu và hình minh họa. Repo lưu [kết quả tổng hợp](reports/results/summary.json), [điểm phân vùng từng ảnh](reports/results/segmentation_per_scene.csv) và [kết quả ghép từng bộ ảnh](reports/results/geometry_per_scene.json).

Thực nghiệm dùng 40 ảnh CamVid có nhãn: 20 ảnh chọn tham số và 20 ảnh kiểm thử, tách theo chuỗi video. Mặt nạ màu được đối chiếu với lớp `Tree`, bỏ qua `Unlabelled` khi tính chỉ số. Tỷ lệ pixel xanh là tỷ lệ trên vùng ảnh hợp lệ. Phép thử panorama dùng các vùng cắt chồng lấn từ cùng một ảnh nguồn.

## Mã nguồn

- [`runner.py`](src/cv_final/runner.py) chạy các bước xử lý và lưu kết quả. [`app.py`](app.py) gọi chung quy trình này từ giao diện Streamlit.
- [`preprocess.py`](src/cv_final/preprocess.py) chuẩn bị ảnh đầu vào. [`geometry.py`](src/cv_final/geometry.py) căn chỉnh và ghép ảnh.
- [`segmentation.py`](src/cv_final/segmentation.py) tạo mặt nạ HSV/ExG. [`evaluate.py`](src/cv_final/evaluate.py) tính điểm dựa trên nhãn tham chiếu.

Đường dẫn ảnh và nhãn CamVid nằm trong [`data/manifest.json`](data/manifest.json). Hai bài tập trước được dẫn trong [`THIRD_PARTY.md`](THIRD_PARTY.md). Mã nguồn và tài liệu có sử dụng AI hỗ trợ.
