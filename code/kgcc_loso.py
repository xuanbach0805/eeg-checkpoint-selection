"""KGCC / PR-PL (Zhong, Shi & Wang 2025) — LOSO tren SEED, chay doc lap tren may Dennis.

CHAY THE NAO
------------
    pip install torch numpy
    python kgcc_loso.py --cache "D:\\30.Dataset\\1.EEG_Datasets\\_rivf2026_cache" ^
                        --out   "D:\\2.GoogleDrive\\10.phD\\1.EEG\\2026\\Journal2026_pilot\\BSPC2026\\results\\kgcc"

Chay lai bao nhieu lan cung duoc: script RESUME duoc o muc EPOCH (checkpoint moi 5 epoch)
va bo qua fold da xong. Ctrl+C bat cu luc nao, chay lai la di tiep.

Neu may co GPU NVIDIA:  them  --device cuda   (nhanh hon ~20-50 lan)
Muon chay it fold mot:         --folds 0,1,2

TRUNG THANH VOI BAN GOC
-----------------------
model      KGCC_src/models/model_PR_PL.py            (import tu file copy nguyen van)
loss       KGCC_src/utils/loss_function.py           (import tu file copy nguyen van)
train step KGCC_src/utils/supervised_driver.py:create_trainer_engine
hyperparam KGCC_src/utils/click_options.py defaults:
             max_epochs 90 · batch_size 96 · RMSprop lr 1e-3 momentum 0.9 wd 1e-5
             · cosine -> 1e-4 · boost_type 'linear' · cluster_weight 2
du lieu    KGCC_src/utils/qat_utils.py:get_dataloaders_and_model
             de_LDS · ToGrid(SEED 9x9) · after_session min-max · CHI session 1 · LeaveOneSubjectOut
Chi thay phan ignite/click bang vong lap thuong. Toan bo phan toan hoc giu nguyen.

LUU Y GIAO THUC (se ghi trong bai): evaluator cua tac gia giu best_acc qua cac epoch tren
chinh subject test -> chon checkpoint bang tap test. Script nay ghi CA HAI: acc_best va acc_last.
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


def build_grid(cache, out_dir):
    """de_LDS -> luoi (n,5,9,9), session 1, chuan hoa min-max theo session. Cache lai ra dia."""
    gp, mp = os.path.join(out_dir, "grid_s1.npy"), os.path.join(out_dir, "meta_s1.npz")
    if os.path.exists(gp) and os.path.exists(mp):
        return gp, mp
    info = json.load(open(os.path.join(cache, "info.json")))
    assert info["channels"] == SEED_CHANNEL_LIST, "thu tu kenh khong khop torcheeg SEED_CHANNEL_LIST"
    meta = np.load(os.path.join(cache, "meta.npz"))
    subject, session, label = meta["subject"], meta["session"], meta["label"]
    sel = np.where(session == 1)[0]
    X = np.load(os.path.join(cache, "X_de_LDS.npy"), mmap_mode="r")
    Xs = np.asarray(X[sel], dtype=np.float32).reshape(-1, 62, 5)     # channel-major
    subj, lab = subject[sel].astype(np.int16), label[sel].astype(np.int8)
    pos = {ch: (r, c) for r, row in enumerate(SEED_LOCATION_LIST)
           for c, ch in enumerate(row) if ch != '-'}
    assert len(pos) == 62
    grid = np.zeros((Xs.shape[0], 5, 9, 9), dtype=np.float32)
    for i, ch in enumerate(SEED_CHANNEL_LIST):
        r, c = pos[ch]
        grid[:, :, r, c] = Xs[:, i, :]
    del Xs
    for s in np.unique(subj):                                        # after_hook_normalize
        m = subj == s
        d = grid[m]; lo = d.min(axis=0, keepdims=True); hi = d.max(axis=0, keepdims=True)
        grid[m] = (d - lo) / (hi - lo + 1e-6)
    np.save(gp, grid); np.savez(mp, subject=subj, label=lab)
    print(f"[prep] luoi {grid.shape} da luu vao {gp}", flush=True)
    return gp, mp


def get_generated_targets(model, x_s, x_t, y_s):
    with torch.no_grad():
        model.eval()
        _, _, _, _, dist = model(x_s, x_t, y_s)
        return model.get_cos_similarity_distance(y_s), model.get_cos_similarity_by_threshold(dist)


def full_eval(model, Xt, Yt, dev, bs=512):
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(Xt), bs):
            feat = model.fea_extrator_f(Xt[i:i + bs])
            logit = torch.matmul(torch.matmul(model.U, feat.T).T, model.stored_mat.to(dev))
            preds.append(torch.softmax(logit, 1).argmax(1).cpu().numpy())
    cl = np.concatenate(preds); mapped = np.zeros_like(cl)
    for i in range(len(model.cluster_label)):
        mapped[cl == i] = model.cluster_label[i]
    return float((mapped == Yt.argmax(1).cpu().numpy()).mean())


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
    tsub = int(np.unique(subj)[fold]); src, tgt = subj != tsub, subj == tsub
    Xs = torch.from_numpy(np.asarray(grid[src])).to(dev)
    Xt = torch.from_numpy(np.asarray(grid[tgt])).to(dev)
    Ys = F.one_hot(torch.from_numpy(lab[src].astype(np.int64) + 1), 3).float().to(dev)
    Yt = F.one_hot(torch.from_numpy(lab[tgt].astype(np.int64) + 1), 3).float().to(dev)
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

    start_ep, hist = 1, []
    if os.path.exists(ck_path):                                       # RESUME
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); disc.load_state_dict(ck["disc"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        model.stored_mat = ck["stored_mat"].to(dev); model.cluster_label = ck["cluster_label"]
        model.upper_threshold, model.lower_threshold = ck["upper"], ck["lower"]
        model.threshold = (model.upper_threshold + model.lower_threshold) / 2
        dann.grl.iter_num = ck["grl_iter"]; hist = ck["hist"]; start_ep = ck["epoch"] + 1
        if ck.get("rng_cpu") is not None:                              # tai lap chinh xac
            torch.set_rng_state(ck["rng_cpu"])
            if dev == "cuda" and ck.get("rng_cuda") is not None:
                torch.cuda.set_rng_state(ck["rng_cuda"])
            np.random.set_state(ck["rng_np"])
        else:
            print(f"[fold {fold}] checkpoint cu khong co trang thai RNG - thu tu batch cua cac "
                  f"epoch con lai se khac ban chay lien mach (khong anh huong tinh hop le)",
                  flush=True)
        print(f"[fold {fold}] resume tu epoch {start_ep}", flush=True)

    n_src, n_tgt = len(Xs), len(Xt)
    iters = n_src // a.batch_size
    print(f"[fold {fold}] subject {tsub} | source {n_src} | target {n_tgt} | {iters} iter/epoch",
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
        hist.append(full_eval(model, Xt, Yt, dev))
        if ep % a.ckpt_every == 0 or ep == a.epochs:
            torch.save({"model": model.state_dict(), "disc": disc.state_dict(),
                        "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "stored_mat": model.stored_mat.cpu(), "cluster_label": model.cluster_label,
                        "upper": model.upper_threshold, "lower": model.lower_threshold,
                        "grl_iter": dann.grl.iter_num, "hist": hist, "epoch": ep,
                        "rng_cpu": torch.get_rng_state(),
                        "rng_cuda": torch.cuda.get_rng_state() if dev == "cuda" else None,
                        "rng_np": np.random.get_state()}, ck_path)
        if ep % 5 == 0 or ep == 1:
            print(f"[fold {fold}] ep {ep:3d}/{a.epochs} acc {hist[-1]:.4f} "
                  f"best {max(hist):.4f} ({time.time()-t0:.0f}s)", flush=True)

    json.dump(dict(fold=fold, target_subject=tsub, epochs=a.epochs, batch_size=a.batch_size,
                   seed=a.seed, acc_last=hist[-1], acc_best=max(hist),
                   acc_best_epoch=int(np.argmax(hist) + 1),
                   acc_mean_last10=float(np.mean(hist[-10:])), history=hist,
                   n_source=int(n_src), n_target=int(n_tgt), device=dev),
              open(res_path, "w"), indent=1)
    torch.save({"model": model.state_dict(), "stored_mat": model.stored_mat.cpu(),
                "cluster_label": model.cluster_label}, os.path.join(a.out, f"fold{fold:02d}_model.pt"))
    if os.path.exists(ck_path):
        os.remove(ck_path)
    print(f"[fold {fold}] XONG  last {hist[-1]:.4f}  best {max(hist):.4f}  "
          f"({time.time()-t0:.0f}s)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=r"D:\30.Dataset\1.EEG_Datasets\_rivf2026_cache")
    p.add_argument("--out", default="./kgcc_results")
    p.add_argument("--folds", default="all")
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=0, help="0 = de torch tu quyet")
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
        print("* CANH BAO: dang chay tren CPU, se mat nhieu gio.")
        print("* Neu may co GPU NVIDIA thi torch dang cai la ban CPU-only (ban PyPI mac dinh).")
        print("* Cai lai ban CUDA, lay lenh dung tai https://pytorch.org/get-started/locally/")
        print("*   vi du:  pip uninstall -y torch")
        print("*           pip install torch --index-url https://download.pytorch.org/whl/cu126")
        print("* Roi chay lai file .bat nay. Tien do da chay khong mat (co checkpoint).")
        print("*" * 74, flush=True)

    gp, mp = build_grid(a.cache, a.out)
    grid = np.load(gp, mmap_mode="r"); meta = np.load(mp)
    folds = list(range(15)) if a.folds == "all" else [int(x) for x in a.folds.split(",")]
    for f in folds:
        run_fold(f, grid, meta, a, dev)

    done = [json.load(open(os.path.join(a.out, f"fold{f:02d}.json")))
            for f in range(15) if os.path.exists(os.path.join(a.out, f"fold{f:02d}.json"))]
    if done:
        bl = np.array([d["acc_last"] for d in done]); bb = np.array([d["acc_best"] for d in done])
        print(f"\n=== {len(done)}/15 fold ===")
        print(f"acc_last  {100*bl.mean():.2f} +- {100*bl.std(ddof=1) if len(bl)>1 else 0:.2f}")
        print(f"acc_best  {100*bb.mean():.2f} +- {100*bb.std(ddof=1) if len(bb)>1 else 0:.2f}  "
              f"(giao thuc cua tac gia: chon epoch tot nhat tren tap test)")


if __name__ == "__main__":
    main()
