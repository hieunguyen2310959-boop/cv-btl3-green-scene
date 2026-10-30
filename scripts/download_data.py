"""Download a fixed CamVid subset; upstream remains the authority for usage terms.
Only downloads source assets, never manufactures annotations or metric values.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'bb68b64e72cb004b3ed927fee0f5624dd8097498'
BASE = f'https://raw.githubusercontent.com/alexgkendall/SegNet-Tutorial/{COMMIT}/CamVid/'


def fetch(url):
    import time
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except (OSError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def main():
    manifest = []
    for source_split, sequences, split in [
        ('train', ['0006R0', '0016E5'], 'dev'),
        ('test', ['0001TP', 'Seq05VD'], 'test'),
    ]:
        lines = fetch(BASE + source_split + '.txt').decode().splitlines()
        for sequence in sequences:
            names = [line.split()[0].split('/')[-1] for line in lines
                     if line.split()[0].split('/')[-1].startswith(sequence + '_')]
            # Fixed uniform sampling, without viewing labels/predictions.
            indices = [round(i * (len(names) - 1) / 9) for i in range(10)]
            for i in indices:
                name = names[i]
                manifest.append({
                    'scene_id': Path(name).stem, 'split': split, 'sequence': sequence,
                    'image_path': f'data/camvid/images/{name}',
                    'annotation_path': f'data/camvid/labels/{name}',
                    'image_url': BASE + source_split + '/' + name,
                    'annotation_url': BASE + source_split + 'annot/' + name,
                    'source_split': source_split, 'positive_class_ids': [5], 'ignore_class_ids': [11],
                    'annotation_semantics': 'CamVid 12-class Tree; not all vegetation species',
                })
    def download(entry):
        for kind in ['image', 'annotation']:
            path = ROOT / entry[kind + '_path']
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = path.read_bytes() if path.exists() else fetch(entry[kind + '_url'])
            path.write_bytes(payload)
            entry[kind + '_sha256'] = hashlib.sha256(payload).hexdigest()
        return entry
    manifest = list(ThreadPoolExecutor(max_workers=8).map(download, manifest))
    dev_seq = {x['sequence'] for x in manifest if x['split'] == 'dev'}
    test_seq = {x['sequence'] for x in manifest if x['split'] == 'test'}
    assert not (dev_seq & test_seq)
    obj = {'dataset': 'CamVid / SegNet-Tutorial subset', 'upstream_commit': COMMIT,
           'sampling': '10 uniformly-spaced frames per sequence, deterministic; no label selection',
           'split_policy': 'sequence-disjoint development and test; correlated frames within sequence',
           'scenes': manifest}
    (ROOT / 'data/manifest.json').write_text(json.dumps(obj, indent=2), encoding='utf-8')
    print(json.dumps({'downloaded_scenes': len(manifest), 'dev': 20, 'test': 20}))


if __name__ == '__main__':
    main()
