"""Build the Vietnamese report from measured JSON, not hand-entered scores."""
from pathlib import Path
from html import escape
import json
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, KeepTogether, Preformatted)
from reportlab.lib.pagesizes import A4

ROOT = Path(__file__).resolve().parents[1]


def main():
    results = ROOT / 'reports/results'
    summary = json.loads((results / 'summary.json').read_text(encoding='utf-8'))
    rows = json.loads((results / 'segmentation_per_scene.json').read_text(encoding='utf-8'))
    geometry = json.loads((results / 'geometry_per_scene.json').read_text(encoding='utf-8'))
    trials = json.loads((results / 'development_trials.json').read_text(encoding='utf-8'))
    config = json.loads((ROOT / 'configs/selected.json').read_text(encoding='utf-8'))
    fonts = Path('C:/Windows/Fonts')
    if not (fonts / 'arial.ttf').exists():
        import os
        fonts = Path(os.environ.get('CV_REPORT_FONT_DIR', '/usr/share/fonts/truetype/dejavu'))
        regular, bold = fonts / 'DejaVuSans.ttf', fonts / 'DejaVuSans-Bold.ttf'
    else:
        regular, bold = fonts / 'arial.ttf', fonts / 'arialbd.ttf'
    pdfmetrics.registerFont(TTFont('Body', str(regular)))
    pdfmetrics.registerFont(TTFont('BodyBold', str(bold)))
    pdfmetrics.registerFontFamily('Body', normal='Body', bold='BodyBold', italic='Body', boldItalic='BodyBold')
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='VN', fontName='Body', fontSize=10.5, leading=15.5, spaceAfter=9))
    styles.add(ParagraphStyle(name='SmallVN', fontName='Body', fontSize=8.3, leading=12, spaceAfter=5))
    styles.add(ParagraphStyle(name='TitleVN', fontName='BodyBold', fontSize=23, leading=31,
                             alignment=TA_CENTER, textColor=colors.HexColor('#123b35'), spaceAfter=16))
    styles.add(ParagraphStyle(name='H1VN', fontName='BodyBold', fontSize=16, leading=22,
                             spaceAfter=13, textColor=colors.HexColor('#123b35')))
    styles.add(ParagraphStyle(name='H2VN', fontName='BodyBold', fontSize=12, leading=17, spaceAfter=9))
    story, markdown = [], []
    def p(text, small=False):
        story.append(Paragraph(text, styles['SmallVN' if small else 'VN']))
        markdown.append(text.replace('<b>', '**').replace('</b>', '**').replace('<br/>', '\n'))
    def heading(text):
        story.append(Paragraph(text, styles['H1VN']))
        markdown.append('\n## ' + text)
    def sub(text):
        story.append(Paragraph(text, styles['H2VN']))
        markdown.append('\n### ' + text)
    def table(data, widths=None):
        cells = [[Paragraph(escape(str(x)), styles['SmallVN']) for x in row] for row in data]
        t = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e2eee9')),
            ('VALIGN', (0,0), (-1,-1), 'TOP'), ('BOX',(0,0),(-1,-1),.4,colors.HexColor('#b1c5bc')),
            ('INNERGRID',(0,0),(-1,-1),.25,colors.HexColor('#cfdbd5')),
            ('LEFTPADDING',(0,0),(-1,-1),6), ('RIGHTPADDING',(0,0),(-1,-1),6),
            ('TOPPADDING',(0,0),(-1,-1),5), ('BOTTOMPADDING',(0,0),(-1,-1),5),
        ]))
        story.extend([t, Spacer(1,12)])
        markdown.extend([' | '.join(map(str,row)) for row in data])
    def figure(name, caption, max_height=180):
        path = ROOT / 'reports/figures' / name
        if not path.exists():
            return
        from PIL import Image as PILImage
        w,h = PILImage.open(path).size
        scale = min(495/w, max_height/h)
        story.append(Image(str(path), width=w*scale, height=h*scale, hAlign='CENTER'))
        p(caption, small=True)
        markdown.append(f'![{caption}](figures/{name})')
    def page():
        story.append(PageBreak())
    fmt = lambda x: '—' if x is None else f'{x:.4f}'

    story.append(Spacer(1,30))
    p('TRƯỜNG ĐẠI HỌC BÁCH KHOA — ĐHQG TP.HCM<br/>KHOA KHOA HỌC VÀ KỸ THUẬT MÁY TÍNH')
    story.append(Spacer(1,35))
    story.append(Paragraph('BÀI TẬP LỚN 3<br/>PHÂN TÍCH CẢNH &amp;<br/>VÙNG CÂY XANH', styles['TitleVN']))
    p('CO3057 — Xử lý ảnh số và Thị giác máy tính<br/>Nhóm 02 · Lớp L01<br/>Giảng viên: Thầy Võ Thanh Hùng')
    story.append(Spacer(1,22))
    table([['MSSV','Thành viên'],['2313026','Hứa Thanh Tâm'],['2212612','Hoàng Ngô Thiên Phúc'],
           ['2310959','Nguyễn Minh Hiếu'],['2433204','Nguyễn Thị Kim Thanh']], [100,395])
    p('<b>Tóm tắt.</b> Xây dựng pipeline Python chạy local, kết hợp tiền xử lý RGB, phân tích hình học '
      '(Canny, Hough, corners và panorama ORB/SIFT) với hai baseline phân vùng theo màu HSV và Excess Green. '
      'Dữ liệu là ảnh đường phố thực có nhãn Tree trong CamVid. Cấu hình được chọn trên tập phát triển, '
      'sau đó đo trên các video sequence tách biệt. Số liệu và hình trong báo cáo được sinh từ lần chạy thật.')
    p('<b>Giới hạn diễn giải:</b> panorama được đánh giá bằng các crop chồng lấn từ cùng một ảnh thực; '
      'không tuyên bố đã đánh giá các góc chụp camera độc lập. Tree là lớp tham chiếu, không phải mọi loại thực vật.')
    p('Mã nguồn: https://github.com/hieunguyen2310959-boop/cv-btl3-green-scene', small=True)
    p('Thời điểm thực nghiệm UTC: '+escape(summary['generated_at_utc']), small=True)
    page()

    heading('1. Bài toán và mức độ đáp ứng đề')
    p('Đầu vào là một ảnh đường phố hoặc danh sách ảnh cùng cảnh. Đầu ra gồm ảnh đã chuẩn hóa, '
      'biên/đường thẳng/điểm góc, panorama nếu ghép thành công, mask vùng cây xanh ứng viên, overlay và JSON kết quả. '
      'Ứng dụng hỗ trợ khảo sát thị giác ở mức prototype, không phục vụ quyết định quy hoạch thực tế.')
    table([['Yêu cầu đề tổng hợp','Hiện thực / bằng chứng'],
           ['3.1 Dữ liệu và tiền xử lý','CamVid có nguồn, SHA256; RGB, grayscale; resize giữ tỷ lệ'],
           ['3.2 Hình học / cấu trúc','Canny + Hough + Shi–Tomasi; panorama ORB/SIFT, RANSAC homography'],
           ['3.3 Nội dung / đối tượng','Phân vùng cơ bản bằng HSV và ExG; đối chiếu nhãn Tree'],
           ['3.4 Trực quan hóa','Inputs, edges, lines/corners, panorama, mask, overlay, demo Streamlit'],
           ['Mở rộng','So sánh 2 đặc trưng, 2 baseline; metric định lượng; test lỗi; demo local']], [145,350])
    sub('1.1 Kết nối BTL1 và BTL2')
    p('BTL1 cung cấp nền tảng màu/xám, lọc và biến đổi ảnh; BTL2 cung cấp luồng đặc trưng → matching → '
      'RANSAC → warp/blend. BTL3 hiện thực lại các module độc lập bằng NumPy/OpenCV, không chép mã nguồn '
      'chưa có giấy phép từ hai repo cũ. Hai lỗi từng tái hiện được đưa thành điều kiện kiểm thử: '
      'identity warp phải giữ hàng/cột cuối; ghép thất bại phải trả failure thay vì dùng identity giả thành công.')
    sub('1.2 Pipeline')
    p('<b>Ảnh nguồn → kiểm tra/resize → hình học → vùng hợp lệ → phân vùng → đánh giá → trực quan hóa.</b><br/>'
      'Nhánh một ảnh dùng edges/lines/corners. Nhánh nhiều ảnh thêm ORB/SIFT, matching, RANSAC và warp/blend. '
      'Mọi metric đều được ghi JSON; file lỗi không có ảnh thành công giả.')
    page()

    heading('2. Dữ liệu, nhãn và thiết kế thực nghiệm')
    ds = summary['dataset']
    table([['Tập','Số frame','Video sequence'], ['Phát triển',ds['dev_frames'],', '.join(ds['dev_sequences'])],
           ['Kiểm thử',ds['test_frames'],', '.join(ds['test_sequences'])]], [100,85,310])
    p('Nguồn ảnh/nhãn: CamVid, qua bản 12 lớp của SegNet-Tutorial [1–2]. Ảnh 480 × 360 pixel. '
      'Mỗi sequence lấy 10 frame cách đều theo danh sách nguồn, không chọn theo nhãn hoặc kết quả dự đoán. '
      'Split được xác định theo sequence trước khi chỉnh ngưỡng; không có sequence xuất hiện ở cả hai tập.')
    p('Nhãn dương: <b>Tree = 5</b>. Bỏ qua <b>Unlabelled = 11</b> khi tính metric. Cần phân biệt '
      'mask ảnh hợp lệ do warp với mask đánh giá: mask đánh giá còn loại pixel nhãn Void/ngoài GT. '
      'Dự đoán không đọc nhãn; chỉ hàm đánh giá được dùng nhãn tham chiếu.')
    p('Giữ manifest với đường dẫn, URL nguồn, commit upstream và SHA256 cho từng ảnh/nhãn. '
      'Không đưa toàn bộ dữ liệu gốc vào Git; script tải từ nguồn gốc và kiểm chứng bằng hash trước khi chạy.')
    sub('2.1 Phân biệt hai loại đánh giá')
    p('<b>Phân vùng ảnh thực:</b> đánh giá HSV/ExG trực tiếp trên toàn bộ 20 frame test, không phụ thuộc '
      'ghép panorama thành công. Nhờ vậy không loại âm thầm các cảnh khó ghép khỏi bảng chính.')
    p('<b>Hình học có kiểm soát:</b> tạo 3 crop rộng 65% ảnh tại các vị trí x biết trước, cùng chiều cao. '
      'Đây là biến đổi tổng hợp của ảnh thực, cho phép đo lỗi chiếu góc độc lập với inlier RANSAC. '
      'Không có parallax, thay đổi phối cảnh lớn hoặc vật di chuyển giữa các view; vì vậy không dùng '
      'kết quả này để khẳng định độ tin cậy panorama camera thực.')
    p('Các frame cùng video vẫn tương quan; 20 frame test không tương đương 20 môi trường độc lập. '
      'Quy mô này phù hợp prototype môn học, chưa phải benchmark tổng quát.')
    page()

    heading('3. Phân tích hình học và cấu trúc')
    p('Ảnh RGB được chuyển grayscale riêng cho đặc trưng; màu RGB vẫn được giữ cho phân vùng. '
      'Canny tìm biên với ngưỡng 60/150; HoughLinesP tìm đoạn thẳng với threshold 30, '
      'minLineLength 25 và maxLineGap 8. Shi–Tomasi cung cấp tối đa 150 corner để trực quan hóa.')
    figure('geometry.png', 'Hình 1. Ảnh thực, biên Canny và đường thẳng/điểm góc. Không xem các đường này là mặt phẳng 3D.', 160)
    sub('3.1 Ghép panorama')
    p('So sánh ORB (descriptor nhị phân, Hamming) và SIFT (descriptor float, L2). Matching 2-nearest-neighbor '
      'với ratio test; RANSAC ước lượng homography nguồn → ảnh trước. Chuỗi homography được đưa về ảnh đầu, '
      'sau đó cộng translation để canvas chứa toàn bộ các view. Không thay homography lỗi bằng identity.')
    p('Warp ảnh và warp mặt nạ hỗ trợ độc lập; blending theo trọng số trên vùng giao. Pixel đen thật vẫn '
      'thuộc vùng hợp lệ, không suy valid mask từ RGB khác 0. Kiểm tra tính hữu hạn, ma trận suy biến, '
      'projective pole và giới hạn canvas trước khi cấp phát.')
    p('RMSE tái chiếu trên inlier là chỉ số fit nội bộ, không phải kiểm chứng độc lập. '
      'Trong crop experiment, sai số góc so với translation biết trước mới là đối chiếu hình học độc lập.')
    page()

    heading('4. Phân vùng và metric')
    sub('4.1 HSV')
    p('Chuyển RGB sang HSV theo OpenCV (H trong 0–179, S/V trong 0–255). Pixel được nhận nếu hue nằm '
      'trong khoảng cho trước, saturation đủ cao và value vượt ngưỡng. Phương pháp dễ giải thích nhưng '
      'nhạy với ánh sáng, cân bằng trắng, bóng râm và đồ vật sơn xanh.')
    sub('4.2 Excess Green (ExG)')
    p('Chuẩn hóa màu theo tổng kênh: r = R/(R+G+B), g = G/(R+G+B), b = B/(R+G+B); '
      'pixel đen dùng mẫu số an toàn. Chỉ số <b>ExG = 2g − r − b</b>; so sánh với ngưỡng cố định. '
      'Đây là chỉ số ưu thế màu xanh, không phải xác suất có thực vật.')
    p('Cả hai baseline chỉ dự đoán trong valid mask. Mã có tùy chọn morphology và lọc component nhỏ; '
      'cấu hình đo chính không bật hậu xử lý để so sánh baseline màu rõ ràng.')
    sub('4.3 Đánh giá')
    p('TP/FP/FN/TN đếm trên vùng valid và không Void. IoU = TP/(TP+FP+FN); precision = TP/(TP+FP); '
      'recall = TP/(TP+FN); F1 = 2TP/(2TP+FP+FN). Khi cả prediction và GT rỗng, IoU/F1 được quy ước bằng 1; '
      'nếu không có pixel đánh giá thì báo lỗi, không tạo NaN.')
    p('Macro IoU là trung bình theo frame; micro IoU tính từ tổng confusion counts. '
      'Báo thêm sai số tuyệt đối của tỷ lệ pixel trên vùng có nhãn; tỷ lệ output thường dùng toàn valid mask, '
      'nên không được lẫn hai mẫu số này.')
    sub('4.4 Chọn tham số trên dev')
    p(f'Đã chạy {len(trials)} cấu hình trên {ds["dev_frames"]} frame dev. Mỗi baseline chọn macro IoU cao nhất; '
      'sau đó ghi configs/selected.json và hash trước khi mở vòng đánh giá test. Không chọn lại ngưỡng theo test.')
    for method in ['hsv','exg']:
        p(method.upper()+': '+escape(json.dumps(config['alternatives'][method]))+ '; dev macro IoU = '+fmt(summary['dev_best'][method]['macro_iou']), small=True)
    page()

    heading('5. Kết quả phân vùng trên ảnh thực')
    data = [['Baseline','Macro IoU','Micro IoU','Precision','Recall','F1']]
    for method in ['hsv','exg']:
        m=summary['segmentation'][method]
        data.append([method.upper(),fmt(m['macro_iou']),fmt(m['micro_iou']),fmt(m['macro_precision']),fmt(m['macro_recall']),fmt(m['macro_f1'])])
    table(data, [75,84,84,84,84,84])
    p('Toàn bộ frame test đều được tính trong bảng; không loại ảnh có ít cây hoặc lỗi màu. '
      'Baseline dùng cho demo/panorama được chọn theo dev, không chọn theo bảng test. '
      'HSV thắng trên dev nhưng ExG có IoU cao hơn trên test; vẫn giữ lựa chọn đã khóa để tránh chọn theo test. '
      'Cả hai IoU đều thấp: baseline màu chưa đủ để đo cây xanh đáng tin cậy.')
    for method in ['hsv','exg']:
        m=summary['segmentation'][method]
        p(f'{method.upper()}: tổng TP={m["tp"]:,}, FP={m["fp"]:,}, FN={m["fn"]:,}; '
          f'sai số tuyệt đối tỷ lệ pixel trung bình = {m["mean_absolute_ratio_error"]*100:.2f} điểm phần trăm.', small=True)
    figure('worst.png', 'Hình 2. Ví dụ IoU thấp nhất của baseline được chọn theo dev; lấy sau đánh giá để minh họa lỗi, không loại khỏi tổng hợp.', 145)
    figure('best.png', 'Hình 3. Ví dụ IoU cao nhất; nhãn Tree tham chiếu và hai prediction. Chênh lệch màu/ánh sáng gây bỏ sót hoặc nhận nhầm.', 145)
    p('Các overlay tô vùng được nhận, không xác nhận mọi vùng xanh là cây. Cần quan sát đồng thời GT và '
      'prediction; không kết luận chất lượng chỉ từ ảnh overlay dễ nhìn.')
    page()

    heading('6. Kết quả hình học có kiểm soát')
    data=[['Đặc trưng','Thành công / thử','Lỗi góc TB (px)','Thời gian hình học TB (s)']]
    for method in ['orb','sift']:
        m=summary['geometry_controlled_crops'][method]
        data.append([method.upper(),f'{m["succeeded"]} / {m["attempted"]}',fmt(m['mean_corner_error_pixels']),fmt(m['mean_geometry_seconds'])])
    table(data,[85,120,130,160])
    p('Một lượt đo trên cùng máy, OpenCV một thread; thời gian gồm phát hiện đặc trưng, matching, '
      'ước lượng, warp và blend. Không coi số đo này là benchmark tốc độ ổn định hay cam kết FPS. '
      'Sai số góc là trung bình tại 4 góc của mỗi view so với translation đã biết, bao gồm view neo.')
    figure('panorama_orb.png', 'Hình 4. Panorama ORB từ crop của một ảnh thực. Chỉ là controlled registration experiment.', 150)
    figure('panorama_sift.png', 'Hình 5. Cùng đầu vào với SIFT; segmentation dùng cấu hình đã khóa từ dev.', 150)
    sub('6.1 Kiểm soát âm và thất bại')
    for c in summary['negative_controls']:
        p(f'{c["method"].upper()} — hai ảnh không texture: <b>{c["status"]}</b>; lý do: '+escape(str(c['reason'])), small=True)
    p('Các ca này phải thất bại rõ ràng, không sinh ảnh giả thành công. Bộ test còn kiểm tra identity warp, '
      'pixel đen hợp lệ, canvas quá lớn và ma trận projective nguy hiểm. Thành công trên crop không thay thế '
      'thử nghiệm camera với parallax, đổi phối cảnh, phơi sáng hoặc đối tượng chuyển động.')
    page()

    heading('7. Kết quả theo từng frame test')
    by_scene={}
    for row in rows:
        by_scene.setdefault(row['scene_id'],{})[row['method']]=row
    data=[['Frame','HSV IoU','HSV F1','ExG IoU','ExG F1']]
    for sid, values in sorted(by_scene.items()):
        data.append([sid,fmt(values['hsv']['iou']),fmt(values['hsv']['f1']),fmt(values['exg']['iou']),fmt(values['exg']['f1'])])
    table(data,[190,76,76,76,77])
    p('CSV/JSON đầy đủ: reports/results/segmentation_per_scene.*. '
      'Chi tiết từng cặp match/inlier, thời gian và lỗi góc: reports/results/geometry_per_scene.json. '
      'Bảng này được tạo trực tiếp từ JSON cùng lần chạy; không sao chép số liệu từ báo cáo BTL2.', small=True)
    page()

    heading('8. Tái lập, hạn chế và kết luận')
    sub('8.1 Tái lập')
    env=summary['environment']
    p('Môi trường thực thi: '+escape(env['platform'])+'; Python '+escape(env['python'].split()[0])+
      '; OpenCV '+escape(env['opencv'])+'; NumPy '+escape(env['numpy'])+'. Seed = 2310959; OpenCV threads = 1.', small=True)
    p('Windows: setup.cmd → tải dữ liệu → test → chạy thực nghiệm → build báo cáo. '
      'README chứa lệnh đầy đủ, requirements-btl3.lock khóa phiên bản. Không cần Docker/GPU; '
      'không tải model hay gửi ảnh lên dịch vụ inference. Dữ liệu được tải lần đầu từ nguồn công khai.')
    p('Hash cấu hình khóa: '+summary['config_sha256'], small=True)
    sub('8.2 Hạn chế')
    for text in [
        'Hai video sequence kiểm thử và các frame tương quan; chưa chứng minh khả năng tổng quát qua mùa, quốc gia hoặc camera.',
        'HSV/ExG là baseline màu: bỏ sót lá vàng/bóng râm, nhầm đồ vật xanh; Tree không bao phủ đầy đủ cỏ và các loại thực vật.',
        'Panorama chỉ đánh giá crop translation; cần bổ sung ảnh chụp độc lập để đánh giá parallax và robustness camera thực.',
        'Không suy diện tích địa lý từ tỷ lệ pixel; không có calibration camera/depth/scale.',
    ]:
        p('• '+text, small=True)
    sub('8.3 Phân công bàn giao để nhóm xác nhận')
    table([['Thành viên','Phần cần review, nắm vững và trình bày'],
           ['Hứa Thanh Tâm','geometry.py; ORB/SIFT, matching, RANSAC, warp/blend'],
           ['Hoàng Ngô Thiên Phúc','segmentation.py; HSV/ExG và failure cases'],
           ['Nguyễn Minh Hiếu','dataset/split, evaluate.py, tests và thực nghiệm'],
           ['Nguyễn Thị Kim Thanh','preprocess/runner, demo và tích hợp']], [140,355])
    p('Đây là phân công bàn giao đề xuất, không phải xác nhận các thành viên đã tự viết các module. '
      'Code và tài liệu có AI hỗ trợ; nhóm cần review, chạy lại, xác nhận đóng góp thực tế và tuân thủ quy định môn học.', small=True)
    p('<b>Kết luận:</b> prototype kết nối đủ bốn nhóm yêu cầu cốt lõi bằng pipeline chạy được, '
      'có nguồn dữ liệu/nhãn, so sánh baseline và bằng chứng thực nghiệm. Giá trị chính là khả năng '
      'tái lập và diễn giải, không phải độ chính xác semantic hoặc khả năng panorama ngoài thực địa đã được chứng minh.')
    page()

    heading('Tài liệu tham khảo và nguồn gốc')
    refs=[
      '[1] Cambridge-driving Labeled Video Database (CamVid). University of Cambridge. https://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/',
      '[2] Alex Kendall et al. SegNet-Tutorial, dữ liệu CamVid 12 lớp và mapping Tree/Unlabelled. https://github.com/alexgkendall/SegNet-Tutorial ; commit bb68b64e72cb004b3ed927fee0f5624dd8097498. Không sử dụng trọng số SegNet.',
      '[3] OpenCV documentation: ORB, SIFT, feature matching, findHomography, warpPerspective, Canny, HoughLinesP và color conversions. https://docs.opencv.org/ ; phiên bản chạy ghi trong mục 8.',
      '[4] Báo cáo và repository BTL1 của nhóm: thanhkimnguyen/CO3057_NHOM_02, commit bb863828c1325210728e047197b74bfb6f0ea7d0. Tham khảo khái niệm, không đóng gói mã nguồn vào repo BTL3.',
      '[5] Báo cáo và repository BTL2 của nhóm: FRGTama/BTLCV2, commit 4c5f7ca61f45ecbc78701841c2a8ab4316c30659. Tham khảo pipeline và các ca lỗi đã tái hiện.',
      '[6] Đề Bài tập tổng hợp môn CO3057 do người dùng cung cấp: CV-project-tonghop (1).pdf, mục 3–5. Không đưa nguyên tài liệu môn học lên repo công khai.',
    ]
    for text in refs:
        p(escape(text), small=True)
    sub('Đối chiếu artifact')
    table([['Artifact','Vai trò'],['data/manifest.json','URL, split, nhãn, SHA256 nguồn'],
           ['configs/selected.json','Ngưỡng đã khóa trước đánh giá test'],
           ['reports/results/summary.json','Tổng hợp đo thật và môi trường'],
           ['reports/results/development_trials.json','Toàn bộ cấu hình thử trên dev'],
           ['reports/Bao-cao-BTL3-CV.pdf','Báo cáo được build từ kết quả'],
           ['docs/VERIFICATION_BTL3.md','Bằng chứng kiểm thử/chạy lại'],
           ['THIRD_PARTY.md','Nguồn tham khảo và phạm vi tái sử dụng']], [215,280])
    p('Dữ liệu gốc thuộc tác giả CamVid; script tải trực tiếp phục vụ học thuật. '
      'Không tự gán giấy phép MIT cho ảnh/nhãn. Hình trích trong báo cáo phục vụ phân tích thực nghiệm, có nguồn dẫn. '
      'Nếu cần phân phối lại dữ liệu ngoài phạm vi này, phải kiểm tra điều khoản từ chủ sở hữu.', small=True)

    out=ROOT/'reports/Bao-cao-BTL3-CV.pdf'
    def footer(canvas, doc):
        canvas.setFont('Body',8)
        canvas.setFillColor(colors.HexColor('#52635c'))
        canvas.drawString(50,28,'CO3057 · BTL3 · Green Scene · '+summary['generated_at_utc'][:10])
        canvas.drawRightString(A4[0]-50,28,str(doc.page))
    SimpleDocTemplate(str(out), pagesize=A4, rightMargin=50,leftMargin=50,topMargin=45,bottomMargin=48,
                      title='BTL3 CO3057 — Phân tích cảnh và vùng cây xanh', author='Nhóm 02 (AI-assisted prototype)').build(story, onFirstPage=footer,onLaterPages=footer)
    (ROOT/'reports/Bao-cao-BTL3-CV.md').write_text('\n\n'.join(markdown),encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
