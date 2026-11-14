"""추론: 저장된 가중치로 예측만 수행 (재학습·재fit 없음). 입력 CSV에 있는 행만 사용.

사용: python test.py --input <CSV> --image_root <이미지 폴더> --output <저장 경로>
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

import model as MD

ap = argparse.ArgumentParser()
ap.add_argument('--input', required=True)
ap.add_argument('--image_root', required=True)
ap.add_argument('--output', required=True)
args = ap.parse_args()

frame = MD.read_frame(args.input).reset_index(drop=True)
W = joblib.load(Path(__file__).resolve().parent / 'weights' / 'model.joblib')
paths = MD.image_paths(frame, args.image_root)
gray = MD.gray_stats(frame, paths)
pos, maps = MD.image_maps(frame, paths)
pred = MD.predict(frame, gray, pos, maps, W)

out = pd.DataFrame({'sample_id': frame['sample_id'].values, 'label': pred.reindex(frame['sample_id'].values).values.astype(int)})
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(args.output, index=False)
print('저장', args.output, '|', len(out), '행 | 분포', out.label.value_counts().sort_index().to_dict())
