"""feat2 센서·누적 피처(이미지 절대값 피처는 전부 제거) + '실험 내 상대' 이미지 피처.
상대 피처(같은 experiment_id, relative_time_s <= 현재인 과거만 사용):
  x_rise      = 현재 - 첫 프레임
  x_from_min  = 현재 - 지금까지 최솟값
  x_delta30   = 현재 - 30초 전 (없으면 NaN)
절대 수준 x 자체는 모델에 넣지 않음."""
import numpy as np, pandas as pd
import feat3 as F3

GRAY_SIG = ['img_mean', 'img_std', 'img_p95', 'img_p99', 'img_hot_ratio']
CMAP_SIG = ['img2_t_mean', 'img2_t_p99', 'img2_hot70', 'img2_c_mean', 'img2_c_p95']


def _relative(frame, base, sigs, w=30.0):
    t = pd.to_numeric(frame['relative_time_s']).to_numpy(float)
    out = {f'{s}_{k}': np.full(len(frame), np.nan) for s in sigs for k in ('rise', 'from_min', 'delta30')}
    for _, idx in pd.Series(np.arange(len(frame))).groupby(frame['experiment_id'].to_numpy(), sort=False).indices.items():
        idx = idx[np.argsort(t[idx], kind='mergesort')]; tt = t[idx]
        lag = np.searchsorted(tt, tt - w, side='right') - 1
        for s in sigs:
            x = base[s].to_numpy(float)[idx]
            out[f'{s}_rise'][idx] = x - x[0]
            out[f'{s}_from_min'][idx] = x - np.minimum.accumulate(x)
            out[f'{s}_delta30'][idx] = np.where(lag >= 0, x - x[np.maximum(lag, 0)], np.nan)
    return pd.DataFrame(out, index=base.index)


def build_features(frame, img_gray, img_cmap, image):
    X = F3.build_features(frame, img_gray, img_cmap, 'none')       # 이미지 피처 전부 제거된 센서 피처
    assert not any(c.startswith('img') for c in X.columns)
    parts = [X]
    if image in ('rel_gray', 'rel_both'):
        G = img_gray.reindex(frame['sample_id'].values); G.index = X.index
        parts.append(_relative(frame, G, GRAY_SIG))
    if image in ('rel_cmap', 'rel_both'):
        I = img_cmap.reindex(frame['sample_id'].values); I.index = X.index
        parts.append(_relative(frame, I, CMAP_SIG))
    return pd.concat(parts, axis=1)
