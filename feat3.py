"""feat2(센서·누적 피처) + 이미지 블록 선택: 'none' | 'gray'(베이스라인) | 'idx'(팔레트 역변환) | 'idx_ts'(+과거 시계열)."""
import numpy as np, pandas as pd
import feat2 as F2

IMG2_TS = ['img2_t_p99', 'img2_t_mean', 'img2_hot70', 'img2_c_mean', 'img2_c_p95', 'img2_top_dummy'][:5]


def _causal(frame, base, sigs, w=30.0):
    t = pd.to_numeric(frame['relative_time_s']).to_numpy(float)
    out = {f'{s}_{k}': np.full(len(frame), np.nan) for s in sigs for k in ('cummax', 'from_peak', 'mean30', 'delta30', 'rise')}
    for _, idx in pd.Series(np.arange(len(frame))).groupby(frame['experiment_id'].to_numpy(), sort=False).indices.items():
        idx = idx[np.argsort(t[idx], kind='mergesort')]; tt = t[idx]
        start = np.searchsorted(tt, tt - w, side='right'); lag = start - 1
        for s in sigs:
            x = base[s].to_numpy(float)[idx]
            cm = np.maximum.accumulate(x)
            cs = np.concatenate([[0], np.cumsum(x)]); n = np.arange(1, len(x) + 1) - start
            out[f'{s}_cummax'][idx] = cm
            out[f'{s}_from_peak'][idx] = x - cm
            out[f'{s}_mean30'][idx] = (cs[np.arange(1, len(x) + 1)] - cs[start]) / n
            out[f'{s}_delta30'][idx] = np.where(lag >= 0, x - x[np.maximum(lag, 0)], np.nan)
            out[f'{s}_rise'][idx] = x - x[0]
    return pd.DataFrame(out, index=base.index)


def build_features(frame, img_gray, img_idx, image='idx_ts'):
    X = F2.build_features(frame, img_gray)
    if image != 'gray':
        X = X.drop(columns=[c for c in X.columns if c.startswith('img_')])
    if image in ('idx', 'idx_ts'):
        I = img_idx.reindex(frame['sample_id'].values); I.index = X.index
        assert I.notna().any(axis=1).all()
        X = pd.concat([X, I], axis=1)
        if image == 'idx_ts':
            X = pd.concat([X, _causal(frame, I, IMG2_TS)], axis=1)
    return X
