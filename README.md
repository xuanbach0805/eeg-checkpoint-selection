# Checkpoint selection in cross-subject EEG emotion recognition

Code, hardware description and per-fold results for:

> Nguyen-Duy X-B and Dinh-Duc A-V, "Checkpoint selection rather than architecture accounts for
> most of the reported cross-subject gap in EEG emotion recognition, and what remains fits in
> 90 bytes". Submitted to *Journal of Neural Engineering*.

The paper reports that, in the released source of six cross-subject SEED methods, the accuracy
recorded for a held-out participant is a maximum over training checkpoints evaluated on that
participant's own labels; that removing this inflation costs 10.94 accuracy points on all 15
participants; that no target-label-free criterion recovers the loss; and that what survives the
correction synthesises to 262 logic elements on a 65 nm FPGA.

## Layout

| Path | Contents |
|---|---|
| `code/` | analysis and training scripts |
| `ed/` | fixed-point data generator, synthesisable Verilog, Quartus synthesis scripts |
| `ed/rtl_data/` | generated weights, biases, stimuli and golden vectors for five configurations |
| `results/` | per-fold JSON for every experiment |
| `results/kgcc/`, `results/kgcc_ea3/` | 90-epoch accuracy trajectories for all 15 folds |
| `figs/make_figures.py` | regenerates all four figures from `results/` |

**Every number and every figure in the paper can be checked from `results/` alone, without
re-running any training.** The training scripts are provided so the results themselves can be
regenerated from the raw corpus.

## Data

This repository contains no EEG recordings and no features derived from them.

- **SEED** must be requested from the Shanghai Jiao Tong University BCMI laboratory under its own
  data-use agreement.
- **DREAMER** and **HBUED** must be obtained from their respective providers.

`code/prep_data.py` builds the feature cache once SEED is available locally.

## Third-party code — not included here

Two files are needed to run `code/kgcc_loso.py` and `code/kgcc_ea3.py` and are **not** part of this
repository, because they are not ours:

- `model_PR_PL.py` — the reference model. Obtain from https://github.com/KAZABANA/PR-PL
  (MIT licence) and place it in `code/`.
- `loss_function.py` — the loss modules used by that model, as distributed with the
  quantisation-aware-training release of Zhong, Shi and Wang (2025),
  *J. Vis. Commun. Image Represent.* **108** 104415. Place it in `code/`.

We reimplemented only the training loop; the model and losses are the original authors' work, used
verbatim. All hyper-parameters follow the released configuration and are documented at the top of
`code/kgcc_loso.py`.

## Reproducing the paper

The experiments were run with PyTorch 2.6.0 (CUDA 12.4 build), scikit-learn 1.9.0,
NumPy 2.4.4 and SciPy 1.17.1 on Windows; `requirements.txt` records the pinned versions
and how to obtain the CUDA build. Matplotlib is needed only by `figs/make_figures.py`
and is not part of the experiment environment.

```
pip install -r requirements.txt

python code/prep_data.py --seed-root <path to SEED>
python code/gta_recompute.py        # table 2, figure 2
python code/linear_matched.py       # tables 3 and 6, figure 3
python code/kgcc_loso.py            # table 6, figure 1(a)     [GPU, ~8 h]
python code/kgcc_ea3.py             # table 8, figure 1(b)     [GPU, ~6 h]
python code/ea2_ptq_kgcc.py         # table 7
python code/eb_mlp.py               # section 4.2
python code/analysis_gtb.py         # statistics
python code/final_analysis.py       # resource budget, table 4

python ed/gen_rtl_data.py           # fixed-point data + overflow and tie checks
quartus_sh -t ed/rtl/synth_all.tcl  # table 5, figure 4        [needs Intel Quartus]

python figs/make_figures.py         # all four figures
```

Both training scripts checkpoint every five epochs and resume from where they stopped, so
interrupting them is safe.

### Hardware note

`ed/rtl/synth_all.tcl` asks Quartus which device families are installed and picks a target
automatically; it builds outside any cloud-synced folder, because Quartus creates and deletes
`incremental_db` continuously and file-sync services interfere with that. Results in the paper are
for a Cyclone III EP3C16F484C6. No device was programmed and no power was measured.

## Licence

MIT for our code (see `LICENSE`). Third-party files, once you add them, keep their original
licences.
