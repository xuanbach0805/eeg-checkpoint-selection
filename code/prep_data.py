"""Build SEED session-1 grid tensor for KGCC/PR-PL, matching the authors' torcheeg pipeline.

Authors' pipeline (KGCC_src/utils/qat_utils.py):
  SEEDFeatureDataset(feature=['de_LDS'],
                     offline_transform=ToGrid(SEED_CHANNEL_LOCATION_DICT),
                     after_session=after_hook_normalize)
  Subcategory('session_id') -> take FIRST session group
  LeaveOneSubjectOut over that session

Reproduced here:
  - de_LDS, channel-major flattening (T, 62*5) -> (T, 62, 5)   [prepare_seed.py line 106]
  - ToGrid with SEED_LOCATION_LIST 9x9 (read from installed torcheeg source, not memory)
  - after_hook_normalize = per-session min-max over samples, per (band,row,col), eps=1e-6
"""
import json, numpy as np, os

CACHE = "/mnt/user-data/uploads/1.EEG_Datasets/_rivf2026_cache"
OUT = "/home/claude/kgcc/data"
os.makedirs(OUT, exist_ok=True)

SEED_CHANNEL_LIST = [
    'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4',
    'F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8',
    'T7','C5','C3','C1','CZ','C2','C4','C6','T8','TP7','CP5','CP3',
    'CP1','CPZ','CP2','CP4','CP6','TP8','P7','P5','P3','P1','PZ',
    'P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8',
    'CB1','O1','OZ','O2','CB2']

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


def main():
    info = json.load(open(f"{CACHE}/info.json"))
    chans = info["channels"]
    assert chans == SEED_CHANNEL_LIST, "channel order mismatch with torcheeg SEED_CHANNEL_LIST"

    meta = np.load(f"{CACHE}/meta.npz")
    subject, session, label = meta["subject"], meta["session"], meta["label"]

    # authors take the FIRST session group
    sel = np.where(session == 1)[0]
    print(f"session-1 windows: {len(sel)} of {len(session)}")

    X = np.load(f"{CACHE}/X_de_LDS.npy", mmap_mode="r")
    assert X.shape[1] == 310
    Xs = np.asarray(X[sel], dtype=np.float32).reshape(-1, 62, 5)   # channel-major
    subj = subject[sel].astype(np.int16)
    lab = label[sel].astype(np.int8)                               # -1/0/1

    # ToGrid: (n, 62, 5) -> (n, 5, 9, 9)
    pos = {ch: (r, c) for r, row in enumerate(SEED_LOCATION_LIST)
           for c, ch in enumerate(row) if ch != '-'}
    assert len(pos) == 62, len(pos)
    grid = np.zeros((Xs.shape[0], 5, 9, 9), dtype=np.float32)
    for i, ch in enumerate(SEED_CHANNEL_LIST):
        r, c = pos[ch]
        grid[:, :, r, c] = Xs[:, i, :]
    del Xs

    # after_hook_normalize per subject-session (here: one session), min-max over samples
    eps = 1e-6
    for s in np.unique(subj):
        m = subj == s
        d = grid[m]
        lo = d.min(axis=0, keepdims=True)
        hi = d.max(axis=0, keepdims=True)
        grid[m] = (d - lo) / (hi - lo + eps)

    np.save(f"{OUT}/grid_s1.npy", grid)
    np.savez(f"{OUT}/meta_s1.npz", subject=subj, label=lab)
    print("grid:", grid.shape, grid.dtype, "range", float(grid.min()), float(grid.max()))
    print("subjects:", np.unique(subj), "labels:", np.unique(lab, return_counts=True))


if __name__ == "__main__":
    main()
