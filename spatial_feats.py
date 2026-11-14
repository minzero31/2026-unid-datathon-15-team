"""I2 공간 핫스팟 피처 (24×32 온도 지수 지도 → 해상도 정규화됨). 프레임마다 자기 범위로 정규화(자동 스케일에 둔감)한 뒤 계산.
프레임 단위: 상대 고온 영역 면적 비율(범위의 50/70/90% 이상), 고온 영역 중심(cx, cy)·퍼짐·세로/가로 퍼짐 비, 최고점 주변 집중도(반경 2 이내 평균 − 전체 평균),
  고온 경계 선명도(경계 픽셀의 기울기), 공간 엔트로피, 좌우·상하 비대칭.
시간 방향(같은 실험의 과거만, 초 단위): 30/120초 전 대비 변화, 첫 프레임 대비 변화, 누적 최댓값, 중심 이동 속도."""
import numpy as np, pandas as pd
H,W=24,32; YY,XX=np.mgrid[0:H,0:W]; YY=YY.ravel()/H; XX=XX.ravel()/W
BASE=['area50','area70','area90','cx','cy','spread','aspect','peak_contrast','edge_sharp','entropy','asym_lr','asym_ud','range']
def frame_stats(M):                       # M: (n, 768)
    lo=M.min(1,keepdims=True); hi=M.max(1,keepdims=True); rg=np.maximum(hi-lo,1e-6); Z=(M-lo)/rg; out={}
    for q in (50,70,90): out[f'area{q}']=(Z>=q/100).mean(1)
    hot=(Z>=0.7).astype(np.float32); hs=np.maximum(hot.sum(1),1); cx=(hot*XX).sum(1)/hs; cy=(hot*YY).sum(1)/hs
    sx=np.sqrt((hot*(XX-cx[:,None])**2).sum(1)/hs); sy=np.sqrt((hot*(YY-cy[:,None])**2).sum(1)/hs)
    out['cx']=cx; out['cy']=cy; out['spread']=np.sqrt(sx**2+sy**2); out['aspect']=sy/np.maximum(sx,1e-3)
    Z3=Z.reshape(-1,H,W); pk=Z.argmax(1); py,px=pk//W,pk%W; d2=(np.arange(H)[None,:,None]-py[:,None,None])**2+(np.arange(W)[None,None,:]-px[:,None,None])**2
    near=(d2<=4); out['peak_contrast']=(Z3*near).sum((1,2))/near.sum((1,2))-Z.mean(1)
    gy,gx=np.gradient(Z3,axis=(1,2)); gm=np.hypot(gx,gy); hb=(Z3>=0.6)&(Z3<=0.8); out['edge_sharp']=(gm*hb).sum((1,2))/np.maximum(hb.sum((1,2)),1)
    p=Z/np.maximum(Z.sum(1,keepdims=True),1e-6); out['entropy']=-(p*np.log(p+1e-9)).sum(1)/np.log(H*W)
    out['asym_lr']=Z3[:,:,:W//2].mean((1,2))-Z3[:,:,W//2:].mean((1,2)); out['asym_ud']=Z3[:,:H//2].mean((1,2))-Z3[:,H//2:].mean((1,2)); out['range']=(hi-lo).ravel()
    return out
def build(frame,pos,maps):
    frame=frame.reset_index(drop=True); t_all=pd.to_numeric(frame.relative_time_s).to_numpy(float); M=maps[pos.reindex(frame.sample_id.values.astype(str)).values].reshape(len(frame),-1).astype(np.float32)
    S=frame_stats(M); out={f'sp_{k}':v.astype(float) for k,v in S.items()}; dyn=['area50','area70','area90','spread','peak_contrast','edge_sharp','entropy','cx','cy']
    ex={f'sp_{k}_{nm}':np.full(len(frame),np.nan) for k in dyn for nm in ('d30','d120','rise','cummax')}; ex['sp_centroid_speed30']=np.full(len(frame),np.nan); ex['sp_centroid_shift']=np.full(len(frame),np.nan)
    for _,ix in pd.Series(np.arange(len(frame))).groupby(frame.experiment_id.to_numpy(),sort=False).indices.items():
        ix=ix[np.argsort(t_all[ix],kind='mergesort')]; t=t_all[ix]; l30=np.searchsorted(t,t-30,side='right')-1; l120=np.searchsorted(t,t-120,side='right')-1
        for k in dyn:
            x=S[k][ix]; ex[f'sp_{k}_d30'][ix]=np.where(l30>=0,x-x[np.maximum(l30,0)],np.nan); ex[f'sp_{k}_d120'][ix]=np.where(l120>=0,x-x[np.maximum(l120,0)],np.nan); ex[f'sp_{k}_rise'][ix]=x-x[0]; ex[f'sp_{k}_cummax'][ix]=np.maximum.accumulate(x)
        cx,cy=S['cx'][ix],S['cy'][ix]; ex['sp_centroid_speed30'][ix]=np.where(l30>=0,np.hypot(cx-cx[np.maximum(l30,0)],cy-cy[np.maximum(l30,0)]),np.nan); ex['sp_centroid_shift'][ix]=np.hypot(cx-cx[0],cy-cy[0])
    out.update(ex); return pd.DataFrame(out)
