# Nguồn tham khảo, dữ liệu và quyền sử dụng

## Mã nguồn BTL3

Các module `src/cv_final`, `app.py`, script BTL3 và tests được hiện thực mới có AI hỗ trợ. Không sao chép code từ hai repo BTL1/BTL2 chưa có LICENSE. Không gán quyền tác giả hoặc đóng góp đã hoàn thành cho từng thành viên; bảng phân công trong báo cáo là phần bàn giao để nhóm review/xác nhận.

## Dữ liệu CamVid

- Chủ nguồn: Cambridge-driving Labeled Video Database (CamVid), University of Cambridge.
- Trang nguồn: https://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
- Mirror 12 lớp được tải trực tiếp: https://github.com/alexgkendall/SegNet-Tutorial
- Commit cố định: `bb68b64e72cb004b3ed927fee0f5624dd8097498`.
- Mapping xác nhận từ `Scripts/test_segmentation_camvid.py`: index 5 = Tree; index 11 = Unlabelled.
- Không sử dụng SegNet code inference hoặc pretrained weights.
- `data/manifest.json` lưu URL/hash; ảnh và nhãn raw không đưa vào Git. `scripts/download_data.py` tải từ nguồn gốc để chạy học thuật.
- Không tuyên bố ảnh/nhãn có giấy phép MIT/CC0. Xem điều khoản của chủ dữ liệu trước khi tái phân phối hoặc dùng ngoài phạm vi học thuật. Hình chọn lọc trong báo cáo là minh họa phân tích có ghi nguồn.

## Các bài trước của nhóm (chỉ tham khảo khái niệm)

- BTL1: https://github.com/thanhkimnguyen/CO3057_NHOM_02 — commit `bb863828c1325210728e047197b74bfb6f0ea7d0`.
- BTL2: https://github.com/FRGTama/BTLCV2 — commit `4c5f7ca61f45ecbc78701841c2a8ab4316c30659`.
- Clone trong `upstream/` giữ nguyên, không đóng gói hoặc push lên repo BTL3.
- Nội suy biên và false-success fallback được kiểm thử lại trong implementation mới; không sửa upstream.

## Thư viện

NumPy, Pillow, OpenCV, Streamlit, ReportLab, PyMuPDF và pytest giữ giấy phép riêng của từng upstream; project chỉ phụ thuộc qua package manager, không vendoring. Tài liệu API chính: https://docs.opencv.org/ . ORB/SIFT/findHomography/warpPerspective dùng từ OpenCV; không tuyên bố đã tự hiện thực detector/RANSAC từ đầu.

## Đề môn học

Đề `CV-project-tonghop (1).pdf` được đọc trên máy người dùng để đối chiếu yêu cầu, không tự công bố nguyên đề hoặc báo cáo BTL cũ trên repo công khai.
