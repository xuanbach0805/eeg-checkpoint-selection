"""E-A3 — KGCC/PR-PL voi CHON CHECKPOINT HOP LE (source-validation).

VAN DE E-A3 GIAI QUYET
----------------------
Trong E-A, evaluator cua tac gia giu `best_acc` qua cac epoch tren CHINH subject test
-> chon checkpoint bang tap test. Chenh acc_best - acc_last = 10.94 diem (15/15 subject).
Reviewer se noi dung: "acc_last cung khong cong bang, khong ai trien khai epoch cuoi mu quang."
E-A3 tra loi bang cach chon epoch theo mot tap validation HOP LE: hai subject NGUON duoc giu lai,
co nhan, KHONG dung de train va KHONG phai target.

THIET KE
--------
Voi moi fold (target subject t, theo thu tu subject 1..15, index 0..14):
  - target        = subject t                        (khong nhan, dung transductive nhu ban goc)
  - source-val    = 2 subject KE TIEP theo vong tron: (t+1) mod 15, (t+2) mod 15
                    -> quy tac CO DINH, khong chon theo ket qua, khong cherry-pick
  - source-train  = 12 subject con lai                (co nhan)
Moi epoch: do accuracy tren source-val VA tren target.
  acc_at_srcval  = accuracy tren TARGET tai epoch co source-val accuracy cao nhat
                   (tie -> epoch nho nhat, deterministic)
  acc_last       = accuracy tren target o epoch cuoi
  acc_best_oracle= max accuracy tren target qua cac epoch  (= giao thuc cua tac gia)

LUU Y KHI VIET BAI (quan trong)
-------------------------------
E-A3 chi con 12 subject nguon (E-A co 14) nen MUC accuracy thap hon E-A mot chut.
=> So sanh chinh phai la NOI BO E-A3: acc_at_srcval vs acc_last vs acc_best_oracle,
   ba con so tu CUNG mot lan chay. KHONG so acc_at_srcval cua E-A3 voi acc_last cua E-A.

Moi thu khac giu y nguyen ban goc: model, loss, hyperparams (90 epoch, batch 96,
RMSprop 1e-3/0.9/1e-5, cosine -> 1e-4, boost linear, cluster_weight 2).
Toi uu (da verify cho ket qua giong het tung chu so): du lieu len GPU mot lan;
`cluster_label_update` chi o iteration cuoi moi epoch; RNG state luu trong checkpoint.

CHAY
----
    chay_kgcc_ea3.bat            (double-click)
hoac
    python kgcc_ea3.py --cache "D:\\30.Dataset\\1.EEG_Datasets\\_rivf2026_cache" ^
                       --out   "...\\BSPC2026\\results\\kgcc_ea3" ^
                       --grid  "...\\BSPC2026\\results\\kgcc\\grid_s1.npy"
Resume duoc o muc epoch. Ctrl+C bat cu luc nao, chay lai la di tiep.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from model_PR_PL import Domain_adaption_model, discriminator, weight_init
from loss_function import CustomLoss, DomainAdversarialLoss

SEED_CHANNEL_LIST = [
    'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5',
    'FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ','C2','C4','C6','T8',
    'TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','P7','P5','P3','P1','PZ','P2','P4',
    'P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8','CB1','O1','OZ','O2','CB2']
SEED_LOCATION_LIST = [
    ['-','-','-','FP1','FPZ','FP2','-','-','-'],
    ['-','-','-','AF3','-','AF4','-','-','-'],
    ['F7','F5','F3','F1','FZ','F2','F4','F6','F8'],
    ['FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8'],
    ['T7','C5','C3','C1','CZ','C2','C4','C6','T8'],
    ['TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8'],
    ['P7','P5','P3','P1','PZ','P2','P4','P6','P8'],
    ['-','PO7','PO5','PO3','POZ','PO4','PO6','PO8','-'],
    ['-','-','CB1','O1','OZ','O2','CB2','-','-']]


def build_grid(cache, out_dir, grid_path=None):
    if grid_path and os.path.exists(grid_path):
        mp = os.path.join(os.path.dirname(grid_path), "meta_s1.npz")
        if os.path.exists(mp):
            print(f"[prep] dung lai luoi co san: {grid_path}", flush=True)
            return grid_path, mp
    gp, mp = os.path.join(out_dir, "grid_s1.npy"), os.path.join(out_dir, "meta_s1.npz")
    if os.path.exists(gp) and os.path.exists(mp):
        return gp, mp
    info = json.load(open(os.path.join(cache, "info.json")))
    assert info["channels"] == SEED_CHANNEL_LIST, "thu tu kenh khong khop SEED_CHANNEL_LIST"
    meta = np.load(os.path.join(cache, "meta.npz"))
    subject, session, label = meta["subject"], meta["session"], meta["label"]
    sel = np.where(session == 1)[0]
    X = np.load(os.path.join(cache, "X_de_LDS.npy"), mmap_mode="r")
    Xs = np.asarray(X[sel], dtype=np.float32).reshape(-1, 62, 5)
    subj, lab = subject[sel].astype(np.int16), label[sel].astype(np.int8)
    pos = {ch: (r, c) for r, row in enumerate(SEED_LOCATION_LIST)
           for c, ch in enumerate(row) if ch != '-'}
    grid = np.zeros((Xs.shape[0], 5, 9, 9), dtype=np.float32)
    for i, ch in enumerate(SEED_CHANNEL_LIST):
        r, c = pos[ch]
        grid[:, :, r, c] = Xs[:, i, :]
    del Xs
    for s in np.unique(subj):
        m = subj == s
        d = grid[m]; lo = d.min(axis=0, keepdims=True); hi = d.max(axis=0, keepdims=True)
        grid[m] = (d - lo) / (hi - lo + 1e-6)
    np.save(gp, grid); np.savez(mp, subject=subj, label=lab)
    print(f"[prep] luoi {grid.shape} -> {gp}", flush=True)
    return gp, mp


def get_generated_targets(model, x_s, x_t, y_s):
    with torch.no_grad():
        model.eval()
        _, _, _, _, dist = model(x_s, x_t, y_s)
        return model.get_cos_similarity_distance(y_s), model.get_cos_similarity_by_threshold(dist)


def evaluate(model, X, Y, dev, bs=1024):
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            feat = model.fea_extrator_f(X[i:i + bs])
            logit = torch.matmul(torch.matmul(model.U, feat.T).T, model.stored_mat.to(dev))
            preds.append(torch.softmax(logit, 1).argmax(1).cpu().numpy())
    cl = np.concatenate(preds); mapped = np.zeros_like(cl)
    for i in range(len(model.cluster_label)):
        mapped[cl == i] = model.cluster_label[i]
    return float((mapped == Y.argmax(1).cpu().numpy()).mean())


def run_fold(fold, grid, meta, a, dev):
    res_path = os.path.join(a.out, f"fold{fold:02d}.json")
    ck_path = os.path.join(a.out, f"fold{fold:02d}_ckpt.pt")
    if os.path.exists(res_path):
        print(f"[fold {fold}] da xong, bo qua", flush=True); return
    t0 = time.time()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if dev == "cpu":
        torch.set_num_threads(a.threads)

    subj, lab = meta["subject"], meta["label"]
    subjects = np.unique(subj)                                  # 1..15
    t_sub = int(subjects[fold])
    v_subs = [int(subjects[(fold + 1) % 15]), int(subjects[(fold + 2) % 15])]  # quy tac co dinh
    m_tgt = subj == t_sub
    m_val = np.isin(subj, v_subs)
    m_src = ~(m_tgt | m_val)

    Xs = torch.from_numpy(np.asarray(grid[m_src])).to(dev)
    Xt = torch.from_numpy(np.asarray(grid[m_tgt])).to(dev)
    Xv = torch.from_numpy(np.asarray(grid[m_val])).to(dev)
    Ys = F.one_hot(torch.from_numpy(lab[m_src].astype(np.int64) + 1), 3).float().to(dev)
    Yt = F.one_hot(torch.from_numpy(lab[m_tgt].astype(np.int64) + 1), 3).float().to(dev)
    Yv = F.one_hot(torch.from_numpy(lab[m_val].astype(np.int64) + 1), 3).float().to(dev)
    if dev == "cuda":
        torch.backends.cudnn.benchmark = True

    model = Domain_adaption_model(32, 32, 32, 32, 3, 32, a.epochs, 0.9, 0.5, dev).to(dev)
    model.apply(weight_init)
    disc = discriminator(32).to(dev); disc.apply(weight_init)
    dann = DomainAdversarialLoss(disc, max_iter=a.epochs).to(dev)
    task_loss = CustomLoss(dann, hidden_4=32, device=dev)
    opt = torch.optim.RMSprop(list(model.parameters()) + list(disc.parameters()),
                              lr=1e-3, momentum=0.9, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-4)

    start_ep, h_tgt, h_val = 1, [], []
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); disc.load_state_dict(ck["disc"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        model.stored_mat = ck["stored_mat"].to(dev); model.cluster_label = ck["cluster_label"]
        model.upper_threshold, model.lower_threshold = ck["upper"], ck["lower"]
        model.threshold = (model.upper_threshold + model.lower_threshold) / 2
        dann.grl.iter_num = ck["grl_iter"]
        h_tgt, h_val = ck["h_tgt"], ck["h_val"]; start_ep = ck["epoch"] + 1
        if ck.get("rng_cpu") is not None:
            torch.set_rng_state(ck["rng_cpu"])
            if dev == "cuda" and ck.get("rng_cuda") is not None:
                torch.cuda.set_rng_state(ck["rng_cuda"])
            np.random.set_state(ck["rng_np"])
        print(f"[fold {fold}] resume tu epoch {start_ep}", flush=True)

    n_src, n_tgt = len(Xs), len(Xt)
    iters = n_src // a.batch_size
    print(f"[fold {fold}] target subj {t_sub} | val subj {v_subs} | "
          f"source {n_src} ({12} subj) | val {len(Xv)} | target {n_tgt} | {iters} iter/epoch",
          flush=True)

    for ep in range(start_ep, a.epochs + 1):
        boost = 2.0 * (ep / a.epochs)
        ps = torch.randperm(n_src).to(dev)
        pt = torch.randperm(n_tgt).to(dev); tp = 0
        model.train()
        for it in range(iters):
            si = ps[it * a.batch_size:(it + 1) * a.batch_size]
            if tp + a.batch_size > n_tgt:
                pt = torch.randperm(n_tgt).to(dev); tp = 0
            ti = pt[tp:tp + a.batch_size]; tp += a.batch_size
            x_s, y_s, x_t = Xs[si], Ys[si], Xt[ti]
            est, est_t = get_generated_targets(model, x_s, x_t, y_s)
            model.train(); opt.zero_grad()
            _, f_s, f_t, sim, sim_t = model(x_s, x_t, y_s)
            total, _, _ = task_loss(model, boost, f_s, f_t, sim, sim_t, est, est_t, a.batch_size)
            total.backward(); opt.step()
            if it == iters - 1:
                model.eval(); model.cluster_label_update(x_s, y_s)
        sched.step(); model.update_threshold(ep)
        h_val.append(evaluate(model, Xv, Yv, dev))
        h_tgt.append(evaluate(model, Xt, Yt, dev))
        if ep % a.ckpt_every == 0 or ep == a.epochs:
            torch.save({"model": model.state_dict(), "disc": disc.state_dict(),
                        "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "stored_mat": model.stored_mat.cpu(), "cluster_label": model.cluster_label,
                        "upper": model.upper_threshold, "lower": model.lower_threshold,
                        "grl_iter": dann.grl.iter_num, "h_tgt": h_tgt, "h_val": h_val,
                        "epoch": ep, "rng_cpu": torch.get_rng_state(),
                        "rng_cuda": torch.cuda.get_rng_state() if dev == "cuda" else None,
                        "rng_np": np.random.get_state()}, ck_path)
        if ep % 5 == 0 or ep == 1:
            k = int(np.argmax(h_val))
            print(f"[fold {fold}] ep {ep:3d}/{a.epochs} val {h_val[-1]:.4f} tgt {h_tgt[-1]:.4f} "
                  f"| chon ep{k+1}: tgt {h_tgt[k]:.4f} | oracle {max(h_tgt):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    k = int(np.argmax(h_val))                                   # tie -> epoch nho nhat
    res = dict(fold=fold, target_subject=t_sub, val_subjects=v_subs, epochs=a.epochs,
               batch_size=a.batch_size, seed=a.seed, device=dev,
               n_source=int(n_src), n_val=int(len(Xv)), n_target=int(n_tgt),
               srcval_best_epoch=k + 1, srcval_best_acc=h_val[k],
               acc_at_srcval=h_tgt[k],
               acc_last=h_tgt[-1],
               acc_best_oracle=max(h_tgt),
               acc_mean_last10=float(np.mean(h_tgt[-10:])),
               history_target=h_tgt, history_srcval=h_val,
               seconds=time.time() - t0)
    json.dump(res, open(res_path, "w"), indent=1)
    torch.save({"model": model.state_dict(), "stored_mat": model.stored_mat.cpu(),
                "cluster_label": model.cluster_label}, os.path.join(a.out, f"fold{fold:02d}_model.pt"))
    if os.path.exists(ck_path):
        os.remove(ck_path)
    print(f"[fold {fold}] XONG  chon ep{k+1} -> tgt {h_tgt[k]:.4f} | last {h_tgt[-1]:.4f} | "
          f"oracle {max(h_tgt):.4f}  ({res['seconds']:.0f}s)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=r"D:\30.Dataset\1.EEG_Datasets\_rivf2026_cache")
    p.add_argument("--out", default="./kgcc_ea3_results")
    p.add_argument("--grid", default=None, help="duong dan grid_s1.npy da dung o E-A (tai su dung)")
    p.add_argument("--folds", default="all")
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--ckpt-every", type=int, default=5)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.threads == 0:
        a.threads = max(1, (os.cpu_count() or 2))
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    print(f"device={dev} threads={a.threads} torch={torch.__version__}", flush=True)
    if dev == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("*" * 74)
        print("* CANH BAO: dang chay tren CPU, se mat rat nhieu gio.")
        print("* Neu may co GPU NVIDIA: pip install torch --index-url "
              "https://download.pytorch.org/whl/cu126")
        print("* Tien do da chay khong mat (co checkpoint).")
        print("*" * 74, flush=True)

    gp, mp = build_grid(a.cache, a.out, a.grid)
    grid = np.load(gp, mmap_mode="r"); meta = np.load(mp)
    folds = list(range(15)) if a.folds == "all" else [int(x) for x in a.folds.split(",")]
    for f in folds:
        run_fold(f, grid, meta, a, dev)

    done = [json.load(open(os.path.join(a.out, f"fold{f:02d}.json")))
            for f in range(15) if os.path.exists(os.path.join(a.out, f"fold{f:02d}.json"))]
    if done:
        sv = np.array([d["acc_at_srcval"] for d in done])
        la = np.array([d["acc_last"] for d in done])
        orc = np.array([d["acc_best_oracle"] for d in done])
        sd = (lambda v: v.std(ddof=1) if len(v) > 1 else 0.0)
        print(f"\n=== E-A3: {len(done)}/15 fold (source-train 12 subject) ===")
        print(f"acc_at_srcval   {100*sv.mean():6.2f} +- {100*sd(sv):5.2f}   "
              f"<- CHON EPOCH HOP LE (2 subject nguon giu lai)")
        print(f"acc_last        {100*la.mean():6.2f} +- {100*sd(la):5.2f}")
        print(f"acc_best_oracle {100*orc.mean():6.2f} +- {100*sd(orc):5.2f}   "
              f"<- giao thuc cua tac gia (chon tren tap test)")
        print(f"\nthoi phong do chon tren tap test = {100*(orc.mean()-sv.mean()):.2f} diem")
        print(f"loi cua chon hop le so voi epoch cuoi = {100*(sv.mean()-la.mean()):+.2f} diem")
        print("\nLUU Y: E-A3 chi co 12 subject nguon (E-A co 14) nen muc accuracy thap hon E-A.")
        print("So sanh chinh phai la NOI BO E-A3, khong tron voi so cua E-A.")


if __name__ == "__main__":
    main()
