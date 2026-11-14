"""피처 생성과 예측 로직 (train.py, test.py 공용).

구성
 1) 단계 모델: 센서·누적·사건·기울기 피처 + 실험 내 상대 이미지 피처(흑백 통계의 변화량)
    + 용량·SOC 보정 + 압력 피처. LightGBM 3개(seed 42/43/44) 확률 평균
    → 45℃ 규칙 → 누적확률 디코딩 → 실험 내 단조 증가.
 2) 2→3 전용 모델: 센서 피처 + 2단계 진입 후 경과 초 + 열화상 공간 핫스팟(I2) + 프레임 간 변화(I3).
    라벨 2·3 행만 학습한 이진 모델 3개(seed 42/43/44) 평균.
    CV에서 2→3 오차가 줄어든 용량 그룹(280Ah, 6Ah)에만 적용.
 3) 73Ah·가열 방식 2 그룹: 2→3 = 2단계 진입 시각 + train 그룹의 2단계 길이 중앙값.

규정 준수
 - 모든 시계열 피처는 같은 experiment_id 안에서 relative_time_s <= 현재인 관측만 사용 (윈도우는 초 단위).
 - 실험마다 따로 계산하므로 입력에 다른 실험이 있든 없든, 이후 시점 행이 있든 없든 결과가 같음.
 - 이미지 피처는 샘플마다 그 이미지 한 장만 보고 계산.
 - test.py는 저장된 가중치로 예측만 하며 어떤 것도 다시 fit 하지 않음.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import lightgbm as lgb
from joblib import Parallel, delayed
from PIL import Image

import base_v7 as M
import feat2
import feat6
import feats_extra
import frame_feats as FF
import img2
import post
import spatial_feats as SP

SEEDS = (42, 43, 44)
GATE_CC = (280.0, 6.0)      # 2→3 전용 모델을 적용하는 용량 그룹(Ah)
G73 = (2, 73.0, 2)          # (bat_class, bat_cc, tr_method): 지속 시간 규칙 그룹
TAU5_TR2 = 0.9
HP = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8,
          subsample_freq=1, colsample_bytree=0.8, class_weight='balanced')


def make_lgbm(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(**HP, num_threads=8, random_state=seed, verbose=-1)


# ---------------------------------------------------------------------------
# 1. 이미지 → 피처 (샘플마다 이미지 한 장만 사용)
# ---------------------------------------------------------------------------
def read_frame(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={'sample_id': 'string', 'experiment_id': 'string'})


def image_paths(frame: pd.DataFrame, image_root) -> pd.Series:
    root = Path(image_root)
    return M.resolve_image_paths(frame, [root, root.parent])


def gray_stats(frame: pd.DataFrame, paths: pd.Series, n_jobs: int = 8) -> pd.DataFrame:
    """베이스라인과 같은 흑백 통계 6개 (sample_id 인덱스)."""
    return M.extract_image_features(frame, paths, n_jobs=n_jobs)


def _map24(path) -> np.ndarray:
    """컬러맵 → 온도 지수(팔레트 역변환) → 24×32로 축소."""
    with Image.open(path) as im:
        rgb = np.asarray(im.convert('RGB'))
    T, _ = img2.to_index(rgb)
    small = Image.fromarray((T * 255).astype(np.uint8)).resize((32, 24), Image.BOX)
    return np.asarray(small, dtype=np.float32) / 255.0


def image_maps(frame: pd.DataFrame, paths: pd.Series, n_jobs: int = 8):
    """(sample_id → 위치, maps[N,24,32]). float16로 저장·사용하던 학습 때와 같은 정밀도로 맞춤."""
    maps = np.stack(Parallel(n_jobs=n_jobs, batch_size=64)(delayed(_map24)(p) for p in paths))
    maps = maps.astype(np.float16).astype(np.float32)
    pos = pd.Series(np.arange(len(frame)), index=frame['sample_id'].values.astype(str))
    return pos, maps


# ---------------------------------------------------------------------------
# 2. 표 피처
# ---------------------------------------------------------------------------
def stage_features(frame: pd.DataFrame, gray: pd.DataFrame) -> pd.DataFrame:
    """단계 모델 입력."""
    b = feat6.build_features(frame, gray, None, 'rel_gray').reset_index(drop=True)
    b = pd.concat([b, feats_extra.capacity_soc(frame.reset_index(drop=True), b)], axis=1)
    b = pd.concat([b, feats_extra.pressure(frame)], axis=1)
    return b.drop(columns=['since_t1_100'])


def t2_info(frame: pd.DataFrame, t1c: np.ndarray):
    """2단계 진입(t1 누적 최댓값 >= 45℃) 시각과 그 이후 경과 초. 과거만 사용."""
    tt = frame.relative_time_s.values.astype(float)
    since = np.full(len(frame), np.nan)
    T2 = {}
    for e, ix in frame.groupby('experiment_id').indices.items():
        h = ix[t1c[ix] >= 45]
        if len(h):
            T2[e] = tt[h].min()
            m = ix[tt[ix] >= T2[e]]
            since[m] = tt[m] - T2[e]
    return since, T2


def t3_features(frame: pd.DataFrame, Bx: pd.DataFrame, pos, maps) -> pd.DataFrame:
    """2→3 전용 모델 입력: 센서 피처 + 2단계 경과 초 + 공간 핫스팟 + 프레임 간 변화."""
    since, _ = t2_info(frame, Bx.meas_t1_cummax.values)
    sens = Bx[[c for c in Bx.columns if not c.startswith('img_')]]
    return pd.concat([sens, pd.DataFrame({'since_T2': since}), SP.build(frame, pos, maps),
                      FF.build(frame, pos, maps)], axis=1)


# ---------------------------------------------------------------------------
# 3. 예측
# ---------------------------------------------------------------------------
def predict(frame: pd.DataFrame, gray: pd.DataFrame, pos, maps, W: dict) -> pd.Series:
    """W: train.py가 저장한 가중치 묶음. 반환: sample_id 인덱스의 단계(1~6)."""
    f = frame.reset_index(drop=True)
    Bx = stage_features(f, gray)[W['stage_cols']]
    t1c = Bx.meas_t1_cummax.values
    tt = f.relative_time_s.values.astype(float)

    # 단계 모델: 확률 평균 → 45℃ 규칙 → 누적확률 디코딩 → 단조
    P = np.mean([m.predict_proba(Bx) for m in W['stage_models']], 0)
    Q = post.apply_rule(P, t1c)
    C = Q[:, ::-1].cumsum(1)[:, ::-1][:, 1:]          # C[:, k] = P(단계 >= k+2)
    TH = np.full((len(P), 5), 0.5)
    TH[f.tr_method.values == 2, 3] = W['tau5_tr2']
    raw = 1 + (C > TH).sum(1)
    p = post._per_exp(f, lambda idx: np.maximum.accumulate(raw[idx]))

    # 2→3 경계 재조정
    p3 = np.mean([m.predict_proba(t3_features(f, Bx, pos, maps)[W['t3_cols']])[:, 1] for m in W['t3_models']], 0)
    _, T2 = t2_info(f, t1c)
    gate = set(W['gate_cc'])
    for e, ix in f.groupby('experiment_id').indices.items():
        if e not in T2:
            continue
        k = (f.bat_class.values[ix[0]], f.bat_cc.values[ix[0]], f.tr_method.values[ix[0]])
        if k == tuple(W['g73']):
            T3 = T2[e] + W['dur73']
        elif float(k[1]) not in gate:
            continue
        else:
            ok = ix[(p3[ix] >= 0.5) & (tt[ix] >= T2[e])]
            T3 = tt[ok].min() if len(ok) else np.inf
        lo = (p[ix] >= 2) & (p[ix] <= 3)
        q = p[ix].copy()
        q[lo & (tt[ix] >= T3)] = 3
        q[lo & (tt[ix] < T3)] = 2
        p[ix] = q
    return pd.Series(p.astype(int), index=f.sample_id.values)
