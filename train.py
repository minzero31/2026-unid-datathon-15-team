"""학습: Train split만 사용해 단계 모델·2→3 전용 모델을 학습하고 weights/model.joblib에 저장.

사용: python train.py --data_dir <battery_data_light 경로>
"""
import argparse
from pathlib import Path

import joblib
import numpy as np

import model as MD

ap = argparse.ArgumentParser()
ap.add_argument('--data_dir', required=True)
ap.add_argument('--out', default=str(Path(__file__).resolve().parent / 'weights' / 'model.joblib'))
args = ap.parse_args()

D = Path(args.data_dir)
fr = MD.read_frame(D / 'train' / 'train.csv').reset_index(drop=True)
y = fr.label.values.astype(int)
paths = MD.image_paths(fr, D / 'train')
print('이미지 피처 계산:', len(fr), '행', flush=True)
gray = MD.gray_stats(fr, paths)
pos, maps = MD.image_maps(fr, paths)

# 1) 단계 모델
Bx = MD.stage_features(fr, gray)
stage_models = [MD.make_lgbm(s).fit(Bx, y) for s in MD.SEEDS]

# 2) 2→3 전용 모델: 라벨 2·3 행만, 목표 = 3단계 이상 여부
Mx = MD.t3_features(fr, Bx, pos, maps)
r23 = np.isin(y, [2, 3])
t3_models = [MD.make_lgbm(s).fit(Mx[r23], (y[r23] >= 3).astype(int)) for s in MD.SEEDS]

# 3) 73Ah·가열 방식 2 그룹의 2단계 길이 중앙값(초): 2단계 진입(t1 누적 최댓값 >= 45℃) → 3단계 시작
c = Bx.meas_t1_cummax.values
d = []
g = fr.assign(c=c)
g = g[(g.bat_class == MD.G73[0]) & (g.bat_cc == MD.G73[1]) & (g.tr_method == MD.G73[2])]
for _, gp in g.groupby('experiment_id'):
    a = gp.relative_time_s.values[gp.c.values >= 45]; b = gp.relative_time_s.values[gp.label.values >= 3]
    if len(a) and len(b):
        d.append(b.min() - a.min())
dur73 = float(np.median(d))

W = dict(stage_models=stage_models, stage_cols=list(Bx.columns), t3_models=t3_models, t3_cols=list(Mx.columns),
         gate_cc=list(MD.GATE_CC), g73=list(MD.G73), dur73=dur73, tau5_tr2=MD.TAU5_TR2)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
joblib.dump(W, args.out, compress=3)     # 무손실 압축 (GitHub 웹 업로드 한도 25MB)
print(f'저장 {args.out} | 단계 모델 피처 {Bx.shape[1]}개, 2→3 전용 {Mx.shape[1]}개 | 73Ah 그룹 실험 {len(d)}개, 2단계 길이 중앙값 {dur73}초')
