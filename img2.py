"""4단계 이미지 피처: 컬러맵(256색 팔레트) → 온도 지수(0~1) 역변환 후 통계.

샘플마다 이미지 한 장만 보고 계산하므로 다른 행 정보를 쓰지 않습니다.
팔레트는 데이터에서 복원한 고정 LUT(palette.npy)이며 라벨을 쓰지 않습니다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image

PALETTE = np.load(Path(__file__).with_name('palette.npy')).astype(np.int64)   # (256, 3), index 0=저온 → 255=고온
_KEYS = PALETTE[:, 0] << 16 | PALETTE[:, 1] << 8 | PALETTE[:, 2]
_LUT = np.full(1 << 24, -1, dtype=np.int16)
_LUT[_KEYS] = np.arange(256)

HOT = (0.5, 0.7, 0.85, 0.95)
COLS = (['t_mean', 't_std', 't_min', 't_p10', 't_p50', 't_p90', 't_p95', 't_p99', 't_max', 't_top1pct',
         'c_mean', 'c_p95', 'c_max', 'grad_mean', 'hot_cx', 'hot_cy', 'hot_spread', 'unmapped']
        + [f'hot{int(h * 100)}' for h in HOT])


def to_index(rgb: np.ndarray) -> np.ndarray:
    """RGB 배열 → 온도 지수(0~1). 팔레트에 없는 색은 가장 가까운 팔레트 색으로."""
    a = rgb.astype(np.int64)
    idx = _LUT[a[..., 0] << 16 | a[..., 1] << 8 | a[..., 2]].astype(np.int32)
    miss = idx < 0
    if miss.any():
        d = ((a[miss][:, None, :] - PALETTE[None]) ** 2).sum(-1)
        idx[miss] = d.argmin(1)
    return idx / 255.0, float(miss.mean())


def _stats(path) -> list[float]:
    with Image.open(path) as im:
        rgb = np.asarray(im.convert('RGB'))
    T, unmapped = to_index(rgb)
    h, w = T.shape
    flat = T.ravel()
    q = np.percentile(flat, [10, 50, 90, 95, 99])
    k = max(1, int(0.01 * flat.size))
    top = np.partition(flat, -k)[-k:].mean()
    c = T[h // 4: h - h // 4, w // 4: w - w // 4]
    gy, gx = np.gradient(T)
    hot = T >= 0.7
    if hot.any():
        ys, xs = np.nonzero(hot)
        cx, cy = xs.mean() / w, ys.mean() / h
        spread = np.sqrt(((xs / w - cx) ** 2 + (ys / h - cy) ** 2).mean())
    else:
        cx = cy = spread = np.nan
    return ([flat.mean(), flat.std(), flat.min(), *q, flat.max(), top,
             c.mean(), np.percentile(c, 95), c.max(), np.hypot(gx, gy).mean(), cx, cy, spread, unmapped]
            + [(flat >= t).mean() for t in HOT])


def extract(frame: pd.DataFrame, paths: pd.Series, n_jobs: int = -1) -> pd.DataFrame:
    rows = Parallel(n_jobs=n_jobs, batch_size=64)(delayed(_stats)(p) for p in paths)
    return pd.DataFrame(rows, columns=[f'img2_{c}' for c in COLS],
                        index=pd.Index(frame['sample_id'].values, name='sample_id'))
