"""열화상 컬러맵(256색) 복원 → palette.npy. train·validation 이미지만 사용하고 라벨은 쓰지 않음.

이미지는 온도를 256색 팔레트로 칠한 false-color 이미지라서, 색을 다시 온도 순서(0=저온 → 255=고온)로
되돌리려면 팔레트 순서가 필요함. 등장하는 색을 모아 가장 어두운 색부터 RGB 거리가 가까운 색을 이어 붙이고,
큰 점프가 생긴 꼬리 부분은 경로 길이가 가장 적게 늘어나는 위치에 다시 끼워 넣음.

사용: python build_palette.py --data_dir <battery_data_light 경로>   (결과는 제출본 palette.npy와 같음)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--data_dir', required=True)
ap.add_argument('--out', default=str(Path(__file__).resolve().parent / 'palette.npy'))
args = ap.parse_args()
D = Path(args.data_dir)

cols = set()
for s in ('train', 'validation'):
    f = pd.read_csv(D / s / f'{s}.csv')
    for p in f.image_path.sample(min(len(f), 3000), random_state=0):
        a = np.asarray(Image.open(D / s / p).convert('RGB')).reshape(-1, 3).astype(np.int32)
        cols.update(np.unique(a[:, 0] << 16 | a[:, 1] << 8 | a[:, 2]).tolist())
cols = np.array(sorted(cols)); rgb = np.stack([cols >> 16, (cols >> 8) & 255, cols & 255], 1).astype(float)

# 1) 가장 어두운 색부터 최근접 이웃 체인
left = list(range(len(rgb))); cur = int(np.argmin(rgb.sum(1))); order = [cur]; left.remove(cur)
while left:
    d = np.linalg.norm(rgb[left] - rgb[cur], axis=1); cur = left.pop(int(np.argmin(d))); order.append(cur)
pal = rgb[order].astype(np.uint8).astype(float)

# 2) 가장 큰 점프 이후의 색을 경로 증가량이 최소인 위치에 재삽입
g = np.linalg.norm(np.diff(pal, axis=0), axis=1); cut = int(np.argmax(g)) + 1
chain = list(pal[:cut])
for c in pal[cut:]:
    P = np.array(chain)
    cost = np.linalg.norm(P[:-1] - c, axis=1) + np.linalg.norm(P[1:] - c, axis=1) - np.linalg.norm(P[1:] - P[:-1], axis=1)
    chain.insert(int(np.argmin(cost)) + 1, c)
pal = np.array(chain).astype(np.uint8)
np.save(args.out, pal)
print('색', len(pal), '| 저장', args.out)
