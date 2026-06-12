"""
inference.py — R²_int / R²_ext / RMSE 评估
"""
import os, sys, numpy as np, torch
from config import DEVICE, MODEL_DIR, CKPT_NAME, TRAIN_Q
from data_loader import build_datasets, denormalize_y
from model import make_model


def load_model(ckpt_path=None):
    if ckpt_path is None: ckpt_path = os.path.join(MODEL_DIR, CKPT_NAME)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = make_model().to(DEVICE)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    return model, ckpt.get("norm", {}), ckpt


def compute_r2(y_true, y_pred):
    ss = ((y_true-y_pred)**2).sum(axis=0)
    st = ((y_true-y_true.mean(axis=0))**2).sum(axis=0)
    return float((1-ss/(st+1e-20)).mean())


@torch.no_grad()
def evaluate_model(model, X, Y, wd, norm, bs=8192):
    preds, n = [], X.shape[0]
    for i in range(0, n, bs):
        preds.append(model(X[i:i+bs]).cpu().numpy())
    Yp_n = np.concatenate(preds); Yt_n = Y.cpu().numpy()
    Yt = denormalize_y(Yt_n, norm); Yp = denormalize_y(Yp_n, norm)
    m = wd.cpu().numpy() > 0; Yt,Yp = Yt[m],Yp[m]
    r2 = compute_r2(Yt, Yp)
    rmse = float(np.sqrt(((Yt-Yp)**2).mean(axis=0)).mean())
    fields = {}
    for j,nm in enumerate(['u','v','w','p']):
        fields[nm] = compute_r2(Yt[:,j:j+1], Yp[:,j:j+1])
    return r2, rmse, fields, Yt, Yp, m


def run_inference(ckpt_path=None):
    model, norm, ckpt = load_model(ckpt_path)
    data = build_datasets()
    print("="*60+"\nINFERENCE\n"+"="*60)

    # Interp
    print(f"\n--- Interpolation (spatial holdout from TRAIN_Q) ---")
    r2i,_,rfi,_,_,_ = evaluate_model(model, data['X_interp'], data['Y_interp'], data['wd_interp'], norm)
    print(f"  R2={r2i:.4f} u={rfi['u']:.4f} v={rfi['v']:.4f} w={rfi['w']:.4f} p={rfi['p']:.4f}")

    # Extrap
    if data['X_extrap'].shape[0] > 0:
        print(f"\n--- Extrapolation ---")
        r2e,_,rfe,_,_,_ = evaluate_model(model, data['X_extrap'], data['Y_extrap'], data['wd_extrap'], norm)
        print(f"  R2={r2e:.4f} u={rfe['u']:.4f} v={rfe['v']:.4f} w={rfe['w']:.4f} p={rfe['p']:.4f}")
    else:
        r2e, rfe = None, {}

    # Per Q
    print(f"\n--- Per Q ---")
    Xt, Yt, wdt = data['X_test'], data['Y_test'], data['wd_test']
    Qn = Xt[:,3].cpu().numpy(); Qp = (Qn+1)/2*norm['Q_max']
    for q in sorted(set(np.round(Qp))):
        mq = np.isclose(Qp, q)
        idx = np.where(mq)[0]
        if len(idx)==0: continue
        r2,_,rf,_,_,_ = evaluate_model(model, Xt[idx], Yt[idx], wdt[idx], norm)
        tag = "interp" if q in TRAIN_Q else "extrap"
        print(f"  Q={int(q):>4d} [{tag:6s}]: R2={r2:.4f} u={rf['u']:.4f} v={rf['v']:.4f} w={rf['w']:.4f} p={rf['p']:.4f}")

    print(f"\nSummary: R2_int={r2i:.4f}  R2_ext={r2e:.4f}" if r2e else "")
    print("="*60)
    return locals()


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv)>1 else None
    run_inference(ckpt)
