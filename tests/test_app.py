from pathlib import Path
import shutil


def test_demo_without_raw_dataset_shows_download_instruction(tmp_path):
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[1]
    shutil.copy2(root / 'app.py', tmp_path / 'app.py')
    (tmp_path / 'data').mkdir()
    shutil.copy2(root / 'data/manifest.json', tmp_path / 'data/manifest.json')
    shutil.copytree(root / 'configs', tmp_path / 'configs')
    assert not (tmp_path / 'data/camvid').exists()

    app = AppTest.from_file(str(tmp_path / 'app.py'), default_timeout=30).run()
    assert not app.exception
    assert len(app.title) == 1
    assert any('python scripts/download_data.py' in item.value for item in app.warning)
    assert app.button[0].disabled

    app.sidebar.selectbox[0].select('Panorama từ crop có kiểm soát').run()
    assert not app.exception
    assert any('python scripts/download_data.py' in item.value for item in app.warning)
    assert app.button[0].disabled

    app.sidebar.selectbox[0].select('Ảnh riêng: chọn theo thứ tự trái → phải').run()
    assert not app.exception
    assert len(app.get('file_uploader')) == 1


def test_demo_opens_without_exceptions():
    assert Path('app.py').exists(), 'demo missing'
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path('app.py').resolve()), default_timeout=30).run()
    assert not app.exception
    assert len(app.title) == 1
