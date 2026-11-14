"""추가 센서 피처: 용량·SOC 보정, 압력. 전부 같은 실험의 과거(relative_time_s <= 현재)만 사용."""
import numpy as np, pandas as pd
from feat2 import GAS


def per_exp(frame, fn, names):
    t = pd.to_numeric(frame.relative_time_s).to_numpy(float); out = {k: np.full(len(frame), np.nan) for k in names}
    for _, ix in pd.Series(np.arange(len(frame))).groupby(frame.experiment_id.to_numpy(), sort=False).indices.items():
        ix = ix[np.argsort(t[ix], kind='mergesort')]
        for k, v in fn(t[ix], ix).items(): out[k][ix] = v
    return pd.DataFrame(out, index=frame.index)


def capacity_soc(frame, base):
    """용량(bat_cc)으로 나눈 가스·온도 상승량, SOC와의 곱. 셀 크기가 다르면 같은 단계에서도 절대량이 다름."""
    cc = pd.to_numeric(frame.bat_cc, errors='coerce').replace(0, np.nan).values; soc = pd.to_numeric(frame.soc_status, errors='coerce').values
    gas = frame[GAS].apply(pd.to_numeric, errors='coerce').sum(axis=1).values
    return pd.DataFrame({'gas_sum_per_cc': gas / cc, 'gas_sum_peak_per_cc': base.gas_sum_cummax.values / cc,
                         't1_rise_per_cc': base.meas_t1_rise_from_start.values / cc,
                         'thermal_rise_per_cc': base.thermal_t_rise_from_start.values / cc,
                         't1_rise_x_soc': base.meas_t1_rise_from_start.values * soc,
                         'thermal_rise_x_soc': base.thermal_t_rise_from_start.values * soc}, index=base.index)


def pressure(frame):
    """압력: 시작 대비 상승, 최솟값 대비, 행간 최대 점프의 누적 최댓값, 상승량 1/2/5 도달 후 경과 초."""
    frame = frame.reset_index(drop=True); P = pd.to_numeric(frame.meas_p, errors='coerce').to_numpy(float)
    def fn(t, ix):
        p = P[ix]; rise = p - p[0]
        o = {'p_rise_start': rise, 'p_from_min': p - np.minimum.accumulate(p), 'p_jump_cummax': np.maximum.accumulate(np.r_[0.0, np.diff(p)])}
        pk = np.maximum.accumulate(rise)
        for th in (1, 2, 5):
            v = np.full(len(t), np.nan); hit = np.flatnonzero(pk >= th)
            if len(hit): v[hit[0]:] = t[hit[0]:] - t[hit[0]]
            o[f'since_p_rise_{th}'] = v
        return o
    return per_exp(frame, fn, ['p_rise_start', 'p_from_min', 'p_jump_cummax', 'since_p_rise_1', 'since_p_rise_2', 'since_p_rise_5'])
