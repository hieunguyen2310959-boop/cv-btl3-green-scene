# Demo và bảo vệ BTL3 (5–7 phút)

## Kịch bản chạy

1. `cmd /c setup.cmd`, tải dữ liệu bằng `scripts/download_data.py` nếu chưa có.
2. `python -m streamlit run app.py --server.address 127.0.0.1` bằng Python của `.venv`.
3. Chọn một ảnh CamVid: chỉ ra ảnh thật, biên Canny, Hough lines/corners, mask và overlay.
4. Đổi HSV ↔ ExG: giải thích ExG thường nhận nhiều pixel hơn nhưng dễ nhầm nền; đối chiếu GT Tree.
5. Chuyển panorama crop có kiểm soát: ORB ↔ SIFT; chỉ ra đây là crop của cùng một ảnh, không phải camera views độc lập.
6. Mở `reports/results/summary.json`: split, confusion counts, macro/micro IoU và lỗi góc biết ground truth.
7. Nêu hai lỗi từng thấy ở bài cũ đã được chặn bằng test: mất biên identity warp; dùng identity khi matching thất bại.

## Những câu cần trả lời được

**Nhóm có tự hiện thực ORB/SIFT/RANSAC không?** Không. BTL3 dùng API OpenCV; đóng góp là thiết kế pipeline, hợp đồng dữ liệu, kiểm tra an toàn, valid mask, GT alignment, protocol đánh giá, runner và demo. Không nhận phần detector của thư viện là thuật toán tự viết.

**Tại sao chọn HSV cho demo dù test ExG có IoU cao hơn?** Lựa chọn được khóa trên dev, trước khi đo test. Đổi lựa chọn theo test sẽ làm test tham gia chọn mô hình. Báo cáo giữ nguyên lựa chọn và công bố cả hai kết quả.

**Tại sao IoU thấp?** Tree là nhãn semantic, trong khi baseline chỉ nhìn màu; lá/bóng râm/vật thể xanh không ánh xạ một-một. Đây là kết quả âm có ý nghĩa: không được dùng baseline này để đo cây xanh chính xác. Không che kết quả bằng accuracy nền cao hoặc chỉ chọn ảnh đẹp.

**Tại sao không gọi tỷ lệ pixel là diện tích mảng xanh?** Pixel phụ thuộc phối cảnh, khoảng cách và camera. Chưa có calibration, depth hoặc scale thực địa.

**Vùng đen có phải padding không?** Không. Warp mặt nạ hỗ trợ riêng; pixel đen thật vẫn valid. Tỷ lệ tính trên valid mask, còn metric bỏ thêm nhãn Unlabelled.

**Inlier RMSE có chứng minh hình học đúng không?** Không, đó là lỗi fit nội bộ. Crop benchmark có translation thật biết trước, nên đo thêm sai số góc độc lập. Dù vậy crop vẫn dễ hơn camera thực có parallax/chuyển động.

**Dataset có leakage không?** Dev và test tách video sequence. Frame cùng sequence còn tương quan; chỉ có hai sequence test, không coi 20 frame là 20 môi trường độc lập. Không chỉnh tham số theo test.

**Có thể dùng ảnh tự chụp không?** Demo/CLI nhận ảnh riêng; cần texture/overlap và gần giả định homography. Chưa có benchmark camera độc lập trong số liệu bàn giao, không được trình bày crop như ảnh tự chụp.

## Trước khi nộp

- Nhóm tự chạy lại và hiểu phần mình nhận; xác nhận đóng góp thực tế thay cho bảng phân công đề xuất.
- Kiểm tra quy định AI và định dạng báo cáo của môn học. Tài liệu/code có AI hỗ trợ, không che nguồn.
- Repo public đúng yêu cầu đề; không push dữ liệu raw hoặc nguyên đề môn học.
- Không thay số metric bằng mục tiêu kỳ vọng. Nếu bổ sung thuật toán/cảnh mới, khóa lại protocol và tái sinh bảng/PDF.
