"""2단계 피처: 베이스라인 피처 + 추가 컬럼 + 누적/사건/기울기 피처.

규정 준수:
- 모든 시계열 피처는 같은 experiment_id 안에서 relative_time_s <= 현재 시점인 관측만 사용합니다.
- 실험마다 따로 계산하므로 다른 실험이 입력에 있든 없든 결과가 같습니다.
- 윈도우는 행 수가 아니라 초 단위입니다(test에 10s/15s 간격 실험이 있음).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import base_v7 as M  # noqa: E402  (대회 베이스라인 v7의 model.py)

GAS = ['meas_ch4', 'meas_co', 'meas_co2', 'meas_hcn', 'meas_hcl', 'meas_hf',
       'meas_n2o', 'meas_no', 'meas_no2', 'meas_so2']
EXTRA_RAW = GAS + ['thermal_t', 'atmp_t', 'indu_t']
META = ['bat_class', 'bat_cc', 'soc_status', 'tr_method']
CUMMAX_SIG = ['meas_t1', 'meas_t2', 'thermal_t', 'meas_p', 'img_max', 'img_p99', 'gas_sum']
SLOPE_SIG = ['meas_t1', 'meas_t2', 'thermal_t', 'meas_v', 'meas_p']
SLOPE_WINDOWS = [15.0, 60.0]


def _lag_index(t: np.ndarray, w: float) -> np.ndarray:
    """각 행에 대해 t - w 이하인 마지막 관측 위치(-1이면 없음)."""
    return np.searchsorted(t, t - w, side='right') - 1


def _since(t: np.ndarray, cond: np.ndarray) -> np.ndarray:
    """cond가 처음 참이 된 시점부터 지난 시간(초). 아직 참이 아니면 NaN. 과거만 사용."""
    out = np.full(len(t), np.nan)
    hit = np.flatnonzero(cond)
    if len(hit):
        k = hit[0]
        out[k:] = t[k:] - t[k]
    return out


def _one_experiment(t: np.ndarray, cols: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    f: dict[str, np.ndarray] = {}
    # 누적 최댓값 / 현재값의 최고점 대비 하락
    for s in CUMMAX_SIG:
        cm = np.fmax.accumulate(np.nan_to_num(cols[s], nan=-np.inf))
        cm[np.isinf(cm)] = np.nan
        f[f'{s}_cummax'] = cm
        f[f'{s}_from_peak'] = cols[s] - cm
    v = cols['meas_v']
    cmin = np.fmin.accumulate(np.nan_to_num(v, nan=np.inf))
    cmin[np.isinf(cmin)] = np.nan
    f['meas_v_cummin'] = cmin
    f['meas_v_drop_from_start'] = v[0] - v
    f['meas_t1_rise_from_start'] = cols['meas_t1'] - cols['meas_t1'][0]
    f['thermal_t_rise_from_start'] = cols['thermal_t'] - cols['thermal_t'][0]

    # 사건 이후 경과 시간
    t1cm = f['meas_t1_cummax']
    for thr in (45, 60, 80, 100, 150, 300):
        f[f'since_t1_{thr}'] = _since(t, t1cm >= thr)
    f['since_v_below_1'] = _since(t, cmin < 1.0)
    f['since_v_below_0.1'] = _since(t, cmin < 0.1)

    # 초 단위 기울기와 그 누적 최댓값
    for w in SLOPE_WINDOWS:
        lag = _lag_index(t, w)
        ok = lag >= 0
        li = np.where(ok, lag, 0)
        dt = np.where(ok, t - t[li], np.nan)
        for s in SLOPE_SIG:
            x = cols[s]
            sl = np.where(ok, (x - x[li]) / dt, np.nan)
            f[f'{s}_slope{int(w)}'] = sl
        for s in ('meas_t1', 'thermal_t'):
            sl = f[f'{s}_slope{int(w)}']
            cm = np.fmax.accumulate(np.nan_to_num(sl, nan=-np.inf))
            cm[np.isinf(cm)] = np.nan
            f[f'{s}_slope{int(w)}_cummax'] = cm
    return f


def build_features(frame: pd.DataFrame, image_features: pd.DataFrame, groups=('base', 'raw', 'cum')) -> pd.DataFrame:
    X = M.build_features(frame, image_features)          # 베이스라인 33개
    X = X.drop(columns=['meas_a'] + [c for c in X.columns if c.startswith('meas_a_')])
    parts = [X]
    if 'raw' in groups:
        raw = M.numeric(frame, EXTRA_RAW + META)
        raw.index = X.index
        parts.append(raw)
    if 'cum' in groups:
        base = M.numeric(frame, ['relative_time_s'] + M.SENSORS + EXTRA_RAW)
        base['gas_sum'] = base[GAS].sum(axis=1)
        img = image_features.reindex(frame['sample_id'].values)
        for c in ('img_max', 'img_p99'):
            base[c] = img[c].values
        base.index = X.index
        times = base['relative_time_s'].to_numpy(float)
        exp = frame['experiment_id'].to_numpy()
        out = None
        for _, idx in pd.Series(np.arange(len(frame))).groupby(exp, sort=False).indices.items():
            idx = idx[np.argsort(times[idx], kind='mergesort')]
            cols = {c: base[c].to_numpy(float)[idx] for c in base.columns}
            f = _one_experiment(times[idx], cols)
            if out is None:
                out = {k: np.full(len(frame), np.nan) for k in f}
            for k, v in f.items():
                out[k][idx] = v
        parts.append(pd.DataFrame(out, index=X.index))
    return pd.concat(parts, axis=1)
