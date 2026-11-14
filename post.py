"""3단계 후처리: 같은 실험 안에서 과거 예측만 사용하는 단조 디코딩."""
import numpy as np
import pandas as pd

L = np.arange(1, 7)


def _per_exp(frame, fn):
    """frame: experiment_id, relative_time_s, t1_cummax 를 포함. fn(idx_sorted) -> labels"""
    out = np.zeros(len(frame), dtype=int)
    t = frame['relative_time_s'].to_numpy(float)
    for _, idx in pd.Series(np.arange(len(frame))).groupby(frame['experiment_id'].to_numpy(), sort=False).indices.items():
        idx = idx[np.argsort(t[idx], kind='mergesort')]
        out[idx] = fn(idx)
    return out


def apply_rule(P, t1cm):
    """45℃ 규칙: t1 누적 최댓값 < 45 → 1단계 확정, 이상 → 1단계 확률 0."""
    P = P.copy()
    below = t1cm < 45
    P[below] = 0; P[below, 0] = 1
    P[~below, 0] = 0
    return P / P.sum(1, keepdims=True)


def decode(frame, P, t1cm, method='raw', k=2, eps=1e-3):
    if method == 'raw':
        return L[P.argmax(1)]
    P = apply_rule(P, t1cm)
    if method == 'rule':
        return L[P.argmax(1)]
    if method == 'cummax':
        return _per_exp(frame, lambda idx: np.maximum.accumulate(L[P[idx].argmax(1)]))
    if method == 'confirm':
        def f(idx):
            raw = L[P[idx].argmax(1)]; cur = raw[0]; out = []; run = 0; cand = None
            for r in raw:
                if r > cur:
                    run = run + 1 if cand is not None and r >= cand else 1
                    cand = r if run == 1 else min(cand, r)
                    if run >= k:
                        cur = cand; run = 0; cand = None
                else:
                    run = 0; cand = None
                out.append(cur)
            return np.array(out)
        return _per_exp(frame, f)
    if method == 'hmm':
        T = np.triu(np.full((6, 6), eps), 1)          # 위 단계로만 이동
        np.fill_diagonal(T, 1 - T.sum(1))
        def f(idx):
            b = np.zeros(6); b[0] = 1.0; out = []
            for p in P[idx]:
                b = (b @ T) * np.clip(p, 1e-6, None)
                b /= b.sum(); out.append(L[b.argmax()])
            return np.array(out)
        return _per_exp(frame, f)
    raise ValueError(method)
