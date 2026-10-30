"""Small local demo; no inference service or cloud upload required."""
from pathlib import Path
import hashlib
import json
import uuid
import streamlit as st
from cv_final.runner import run_scene

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Green Scene | CO3057', layout='wide')
st.title('Green Scene — Phân tích cảnh & vùng cây xanh')
st.caption('BTL3 · CO3057 · RGB → hình học → phân vùng → trực quan hóa')
st.info('Baseline màu không phải mô hình hiểu ngữ nghĩa. Tỷ lệ pixel xanh không phải diện tích thực địa. '
        'Panorama từ crop là kiểm thử có kiểm soát, không phải ảnh camera chụp độc lập.')
manifest_path = ROOT / 'data/manifest.json'
if not manifest_path.exists():
    st.warning('Chưa có dữ liệu. Chạy: python scripts/download_data.py')
else:
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    scenes = {e['scene_id']: e for e in manifest['scenes']}
    mode = st.sidebar.selectbox('Chế độ', ['Một ảnh thực: edges/lines + phân vùng',
                                          'Panorama từ crop có kiểm soát', 'Ảnh riêng: chọn theo thứ tự trái → phải'])
    method = st.sidebar.selectbox('Phân vùng', ['hsv', 'exg'])
    geo = st.sidebar.selectbox('Đặc trưng panorama', ['orb', 'sift'])
    config_path = ROOT / 'configs/selected.json'
    if not config_path.exists():
        config_path = ROOT / 'configs/default.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    config['segmentation'] = dict(config.get('alternatives', {}).get(method, {'method': method}))
    config['geometry']['method'] = geo
    entry = None
    if mode.startswith('Ảnh riêng'):
        uploaded = st.file_uploader('Chọn từ 1–8 ảnh, thứ tự danh sách là thứ tự ghép',
                                   type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
        if uploaded:
            if len(uploaded) > 8:
                st.error('Tối đa 8 ảnh để giới hạn bộ nhớ.')
            else:
                cache = ROOT / '.cache/uploads'
                cache.mkdir(parents=True, exist_ok=True)
                paths = []
                for item in uploaded:
                    payload = item.getvalue()
                    path = cache / (hashlib.sha256(payload).hexdigest() + '.png')
                    path.write_bytes(payload)
                    paths.append(str(path))
                st.write('Thứ tự:', [u.name for u in uploaded])
                entry = {'scene_id': 'uploaded', 'view_paths': paths}
    else:
        scene = st.selectbox('Ảnh CamVid có nhãn', list(scenes))
        if not (ROOT / scenes[scene]['image_path']).is_file():
            st.warning('Chưa có dữ liệu ảnh. Chạy: python scripts/download_data.py')
        else:
            entry = dict(scenes[scene])
            entry['controlled_crops'] = mode.startswith('Panorama')
            st.caption(f"Split: {entry['split']} · sequence: {entry['sequence']} · nhãn dương: Tree (5)")
            st.image(str(ROOT / entry['image_path']), caption='Ảnh thực đầu vào', width=600)
    if st.button('Chạy phân tích', type='primary', disabled=entry is None):
        output = ROOT / 'outputs/demo' / uuid.uuid4().hex
        with st.spinner('Đang chạy trên máy local…'):
            result = run_scene(entry, config, output, ROOT)
        st.session_state['latest'] = (str(output), result)
    if 'latest' in st.session_state:
        output_str, result = st.session_state['latest']
        output = Path(output_str)
        if result['status'] != 'success':
            st.error('Không thể phân tích: ' + str(result['reason']))
        else:
            a, b, c = st.columns(3)
            a.metric('Pixel được baseline nhận là xanh', f"{result['vegetation_pixel_ratio']:.1%}")
            b.metric('Pixel ảnh hợp lệ', f"{result['valid_pixels']:,}")
            c.metric('Thời gian gồm I/O', f"{result['timing_seconds']['total_including_io']:.3f} s")
            left, right = st.columns(2)
            for target, name, caption in [
                (left, 'panorama', 'Ảnh / panorama'), (right, 'overlay', 'Vùng dự đoán (overlay)'),
                (left, 'lines', 'Hough lines & Shi–Tomasi corners'), (right, 'vegetation_mask', 'Mask dự đoán'),
            ]:
                target.image(str(output / (name + '.png')), caption=caption, width='stretch')
            if (output / 'ground_truth.png').exists():
                st.image(str(output / 'ground_truth.png'), caption='Nhãn Tree tham chiếu (bỏ qua Void)', width=600)
                st.json(result['evaluation'])
            with st.expander('Trạng thái, homography và thời gian'):
                st.json(result)
            st.download_button('Tải status.json', (output / 'status.json').read_bytes(), 'status.json', 'application/json')
