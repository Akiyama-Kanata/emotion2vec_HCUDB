import json

nb_path = r'c:\Users\RD004\Documents\lab\emotion2vec\vad_downstream\experiment.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# --- 修正1: c958a94a を置き換え ---
new_src_1 = [
    'import os, sys\n',
    'os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 二重ロード回避\n',
    '\n',
    '# ffmpeg が anaconda の Library/bin にあるので PATH に追加する\n',
    r'_ffmpeg_dir = r"C:\Users\RD004\anaconda3\Library\bin"' + '\n',
    'if _ffmpeg_dir not in os.environ.get("PATH", ""):\n',
    '    os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")\n',
    '    print(f"PATH に追加: {_ffmpeg_dir}")\n',
    '\n',
    '# 環境確認\n',
    'import torch, torchaudio, soundfile\n',
    'print(f"torch      : {torch.__version__}")\n',
    'print(f"torchaudio : {torchaudio.__version__}")\n',
    'print(f"soundfile  : {soundfile.__version__}")\n',
    'print(f"CUDA利用可能 : {torch.cuda.is_available()} (CPUモードで実行)")\n',
]

for cell in cells:
    if cell.get('id') == 'c958a94a':
        cell['source'] = new_src_1
        cell['outputs'] = []
        cell['execution_count'] = None
        print('修正1: c958a94a 置き換え完了')
        break

# --- 修正2: c958a94a の直後に新規セルを挿入（既存なら上書き） ---
patch_src = [
    '# torchaudio 2.9+ は torchcodec を必須とするが未インストール。\n',
    '# soundfile（インストール済み）で torchaudio.load を上書きして回避する。\n',
    'import soundfile as sf\n',
    'import torch\n',
    'import torchaudio as _torchaudio\n',
    '\n',
    'def _soundfile_load(uri, frame_offset=0, num_frames=-1, normalize=True,\n',
    '                    channels_first=True, format=None, buffer_size=4096, backend=None):\n',
    '    data, sr = sf.read(str(uri), dtype="float32", always_2d=True)\n',
    '    tensor = torch.from_numpy(data).T\n',
    '    if frame_offset > 0:\n',
    '        tensor = tensor[:, frame_offset:]\n',
    '    if num_frames > 0:\n',
    '        tensor = tensor[:, :num_frames]\n',
    '    if not channels_first:\n',
    '        tensor = tensor.T\n',
    '    return tensor, sr\n',
    '\n',
    '_torchaudio.load = _soundfile_load\n',
    'print("torchaudio.load → soundfile バックエンドにパッチ完了")\n',
]

existing_ids = [c.get('id') for c in cells]
if '5baf7ca4' in existing_ids:
    for cell in cells:
        if cell.get('id') == '5baf7ca4':
            cell['source'] = patch_src
            cell['outputs'] = []
            cell['execution_count'] = None
            print('修正2: 5baf7ca4 更新完了（既存セルを上書き）')
            break
else:
    new_cell = {
        'cell_type': 'code',
        'id': '5baf7ca4',
        'metadata': {},
        'outputs': [],
        'execution_count': None,
        'source': patch_src,
    }
    idx = next(i for i, c in enumerate(cells) if c.get('id') == 'c958a94a')
    cells.insert(idx + 1, new_cell)
    print(f'修正2: 5baf7ca4 を index {idx+1} に挿入完了')

# --- 修正3: 3a998d48 の先頭に安全チェックを追加（重複防止） ---
safety_lines = [
    'if "model" not in dir() or model is None:\n',
    '    raise RuntimeError(\n',
    "        \"model が未定義です。セル '5. モデル構築' から順番に実行してください。\"\n",
    '    )\n',
    '\n',
]

for cell in cells:
    if cell.get('id') == '3a998d48':
        existing = cell['source'] if isinstance(cell['source'], list) else [cell['source']]
        if 'if "model" not in dir()' not in ''.join(existing):
            cell['source'] = safety_lines + existing
            print('修正3: 3a998d48 安全チェック追加完了')
        else:
            print('修正3: 3a998d48 は既に安全チェックあり（スキップ）')
        cell['outputs'] = []
        cell['execution_count'] = None
        break

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('ノートブック書き込み完了')
