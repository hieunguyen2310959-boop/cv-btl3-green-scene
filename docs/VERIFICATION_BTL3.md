# Verification snapshot — BTL3

Các số liệu dưới đây là kết quả thực thi local trên Windows, không phải kết quả CI được giả định.

## Thực nghiệm dữ liệu thật

- Đã tải 40 cặp ảnh/nhãn CamVid đúng URL/commit nguồn, lưu SHA256 trong manifest.
- Split theo video sequence: 20 dev (`0006R0`, `0016E5`) và 20 test (`0001TP`, `Seq05VD`), không giao sequence.
- Thử 29 cấu hình màu chỉ trên dev, khóa `configs/selected.json`, sau đó đánh giá cả 20 frame test cho từng HSV/ExG.
- Chạy 20 bộ crop cho từng ORB/SIFT. Cả hai ghép thành công 20/20 trong **controlled crop experiment**, không phải benchmark camera thực.
- Hai negative control ảnh không texture trả failure; không có identity giả thành công.
- Chạy lại toàn bộ thực nghiệm lần thứ hai: confusion counts, điểm segmentation và lỗi góc hình học **trùng chính xác**; thời gian khác nhau được đo lại, không sao chép.
- `reports/results/summary.json`, per-scene CSV/JSON và development trials là nguồn cho báo cáo.

## Chất lượng baseline (công bố cả điểm thấp)

- HSV macro IoU test: 0.1089237612; ExG: 0.1546786699.
- Lựa chọn demo vẫn là HSV vì được chọn theo dev trước khi đo test.
- Chưa có semantic segmentation model được train, chưa benchmark camera views độc lập, không đo diện tích địa lý.

## Cổng kiểm tra local đã chạy

- Bộ test unit/integration sau sửa: **53 passed, 110 subtests passed** trên bản export sạch với môi trường `.venv` mới từ lockfile.
- Thêm regression cho serialization LF; JSON được ghi UTF-8/LF trước khi hash, không phụ thuộc CRLF của Windows. SHA256 của cấu hình lấy trực tiếp từ Git index khớp summary và text PDF: `909874a620fb98e77f8d80391aba0ec3509c8d355877ad3d1d094bd0e822f0eb`.
- Streamlit AppTest: khởi động và nhấn nút chạy trên ảnh CamVid có dữ liệu, 0 exception, có đủ 3 metric.
- Đã sửa lỗi clean-checkout: manifest có sẵn nhưng raw ảnh chưa tải sẽ hiện hướng dẫn download, không crash. Đã kiểm chứng trên bản sạch thực sự chưa có `data/camvid/`, không chỉ bằng fault injection.
- Từ bản sạch: `setup.cmd` thành công → download lại 40 ảnh/nhãn → CLI một ảnh và panorama crop trả 0 → thực nghiệm toàn bộ → build PDF 10 trang.
- So với bản làm việc: confusion counts/scores và lỗi góc khớp chính xác; hash cấu hình khớp. Timing được đo riêng.
- CLI hai ảnh không texture trả đúng **exit code 2**, status failed, không có panorama.png.
- Report PDF được build từ JSON, 10 trang; đã kiểm tra text layer và xem trang bảng/hình/phân công để phát hiện font/tràn lề.
- Scan staged source không có token GitHub, private key, `eval`, shell injection hoặc unsafe pickle.
- Review độc lập đã kiểm tra protocol split/GT/canvas và tái lập metric trên staged export. Sau sửa hash LF và số lượng trial, vòng re-verification trả **PASS**, không còn security concern hoặc logic error chặn publish. Các đề xuất hardening upload/download chỉ áp dụng nếu mở demo ra public hosting.

## Tính toàn vẹn bàn giao

- Không push upstream BTL1/BTL2, dữ liệu raw, .venv, .env hoặc nguyên đề môn học.
- `.gitattributes` khóa LF cho JSON để hash cấu hình không bị đổi bởi line-ending của Git trên Windows.
- Tài liệu này ghi các cổng local đã chạy. Trạng thái remote/GitHub Actions phải đọc trực tiếp trên GitHub sau push; không suy ra CI xanh từ unit test local.
