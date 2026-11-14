"""프레임 간 변화 피처 (24×32 온도 지수 지도, model.image_maps로 원본 이미지에서 계산). 같은 실험의 현재·과거 프레임만 사용, 윈도우는 초 단위.
컬러맵이 프레임마다 자동 스케일되므로 스케일에 덜 민감한 상관·패턴 변화 위주:
  corr_first: 첫 프레임과의 공간 상관 / corr_lagW: W초 전 프레임과의 상관 / mad_lagW: 평균 절대 변화 / chg_lagW: 0.1 넘게 변한 픽셀 비율
  위 값들의 누적 최소·최대(사건이 한 번이라도 있었는지), hot 영역(상위 10% 픽셀)의 무게중심 이동·면적 변화, 공간 표준편차의 변화."""
import numpy as np, pandas as pd
LAGS=(15.0,30.0,60.0)
def _corr(a,b):
    a=a-a.mean(1,keepdims=True); b=b-b.mean(1,keepdims=True); d=np.sqrt((a*a).sum(1)*(b*b).sum(1)); return np.where(d>1e-9,(a*b).sum(1)/np.maximum(d,1e-9),1.0)
def build(frame,pos,maps):
    frame=frame.reset_index(drop=True); t_all=pd.to_numeric(frame.relative_time_s).to_numpy(float); M=maps[pos.reindex(frame.sample_id.values.astype(str)).values].reshape(len(frame),-1)
    names=['corr_first','sstd','sstd_rise','hot_cx_shift','hot_cy_shift','top10_mean_minus_mean']+[f'{k}_lag{int(w)}' for w in LAGS for k in ('corr','mad','chg')]
    names+=[n+'_cummin' for n in names if n.startswith('corr')]+[n+'_cummax' for n in names if n.startswith(('mad','chg','hot_c'))]+['since_scene_change']
    out={n:np.full(len(frame),np.nan) for n in names}; H,W=24,32; yy,xx=np.mgrid[0:H,0:W]; yy=yy.ravel()/H; xx=xx.ravel()/W
    for _,ix in pd.Series(np.arange(len(frame))).groupby(frame.experiment_id.to_numpy(),sort=False).indices.items():
        ix=ix[np.argsort(t_all[ix],kind='mergesort')]; t=t_all[ix]; m=M[ix]; n=len(ix); o={}
        o['corr_first']=_corr(m,np.repeat(m[:1],n,0)); o['sstd']=m.std(1); o['sstd_rise']=o['sstd']-o['sstd'][0]
        thr=np.quantile(m,0.9,axis=1,keepdims=True); hot=(m>=thr).astype(np.float32); hs=np.maximum(hot.sum(1),1)
        cx=(hot*xx).sum(1)/hs; cy=(hot*yy).sum(1)/hs; o['hot_cx_shift']=np.abs(cx-cx[0]); o['hot_cy_shift']=np.abs(cy-cy[0]); o['top10_mean_minus_mean']=(m*hot).sum(1)/hs-m.mean(1)
        for w in LAGS:
            lag=np.searchsorted(t,t-w,side='right')-1; ok=lag>=0; ref=m[np.maximum(lag,0)]; d=np.abs(m-ref)
            o[f'corr_lag{int(w)}']=np.where(ok,_corr(m,ref),np.nan); o[f'mad_lag{int(w)}']=np.where(ok,d.mean(1),np.nan); o[f'chg_lag{int(w)}']=np.where(ok,(d>0.1).mean(1),np.nan)
        for k in list(o):
            if k.startswith('corr'): o[k+'_cummin']=np.fmin.accumulate(np.nan_to_num(o[k],nan=1.0))
            if k.startswith(('mad','chg','hot_c')): o[k+'_cummax']=np.fmax.accumulate(np.nan_to_num(o[k],nan=0.0))
        sc=np.nan_to_num(o['corr_lag15'],nan=1.0)<0.5; s=np.full(n,np.nan); h=np.flatnonzero(sc)
        if len(h): s[h[0]:]=t[h[0]:]-t[h[0]]
        o['since_scene_change']=s
        for k,v in o.items(): out[k][ix]=v
    return pd.DataFrame(out).add_prefix('fr_')
