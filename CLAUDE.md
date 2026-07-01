# Repo handout — Spectrogram Autoencoder for Deepfake Speech Detection

A reference for understanding this codebase. It describes what the repo does and how the
pieces fit together, then lists known issues as verifiable, code-grounded observations.
It does not prescribe fixes or argue for any particular research direction.

---

## 1. Main idea

Train an autoencoder (AE) **only on real (bonafide) speech**. At test time, the per-utterance
**reconstruction error** is used as the detection score: real speech is expected to reconstruct
well (low score), synthetic/deepfake speech less well (higher score). No fake samples are used in
training — the detector is unsupervised from the spoof side.

Detection is performed on **mel spectrograms**. Before the spectrogram enters the encoder, a binary
**mask** is applied to it (multiplicatively). The reconstruction **target is the full mel**, so the
network must reproduce the masked-out content as well.

Labels throughout the code: **`bonafide = 1`, `spoof = 0`**. A higher reconstruction score is
treated as "more likely spoof." AUROC is computed as `roc_auc_score(labels, -scores)` so that
bonafide (label 1) is the positive class.

---

## 2. End-to-end pipeline

```
audio file
  → load: mono, resample to 16 kHz, repeat-pad/truncate to 5 s        (dataset.py _load_audio)
  → [train only] RawBoost waveform augmentation, prob 0.7             (train.py apply_rawboost)
  → log-mel spectrogram: 80 mels, n_fft=1024, hop=160, win=400,
    f_max=8000, AmplitudeToDB(power, top_db=80), fixed to [80, 500]   (features.py compute_mel)
  → binary mask (contour or energy)                                   (features.py compute_mask)
  → masked_input = mel * mask                                         (train.py / evaluate.py)
  → AE encoder → latent (dim 256) → decoder → reconstruction [80,500] (models/*)
  → train: MSE(reconstruction, full mel)                             (train.py)
  → eval:  score = mean|mel - reconstruction|  (or feature-space L2)  (evaluate.py)
  → metrics: EER, AUROC, score distribution, error-map plots         (utils.py)
```

Important: in training, `masked_input` and the reconstruction `target` are **both derived from the
same mel** (`train.py:108-109`). When RawBoost is on, that mel is the augmented one, so input and
target share the same augmentation; the **only** difference between input and target is the mask.
At evaluation there is no RawBoost.

---

## 3. File map

| File | Role |
|---|---|
| `config.py` | `Config` dataclass: all hyperparameters (audio, mel, mask, model, training, eval). |
| `features.py` | `compute_mel`, `compute_contour_mask`, `compute_energy_mask`, `compute_mask` dispatch. |
| `dataset.py` | Dataset classes (ASVspoof2019/5, In-The-Wild, FakeXpose, MLAAD, FamousFigures, AT-ADD), `_load_audio`, `collate_fn`. |
| `RawBoost.py` | Convolutive / impulsive / stationary noise functions used for waveform augmentation. |
| `models/cnn_ae.py` | CNN autoencoder (4 conv stages, FC bottleneck). Exposes `intermediate_features`. |
| `models/convnext_ae.py` | ConvNeXt-based AE. Exposes `intermediate_features`. |
| `models/vit_ae.py` | ViT-Tiny encoder + transformer decoder AE. |
| `models/vit_small_ae.py` | ViT-Small variant. |
| `models/__init__.py` | `build_model(config)` selects encoder by `config.encoder`. |
| `train.py` | Training loop, RawBoost, dataset assembly, optimizer/scheduler, checkpointing, resume. |
| `evaluate.py` | Scoring, EER/AUROC, score-distribution and error-map plots, paired plots. |
| `utils.py` | `compute_eer`, plotting helpers, `set_seed`. |
| `*.sbatch` | SLURM launch scripts (`train.sbatch`, `train_atadd.sbatch`, `eval.sbatch`). |
| `preprocessing_explorer.ipynb` | Exploratory preprocessing / visualization notebook. |

---

## 4. Configuration (`config.py`)

- **Audio:** `sample_rate=16000`, `max_duration_seconds=5`.
- **Mel:** `n_mels=80`, `n_fft=1024`, `hop_length=160`, `win_length=400`, `f_min=0`, `f_max=8000`.
- **Mask:** `mask_type="contour"` (or `"energy"`); `contour_threshold_db=40.0` (kept where normalized
  Sobel gradient `> threshold/100 = 0.40`); `energy_threshold_db=40.0` (kept where value is within
  40 dB of the spectrogram's global max).
- **Model:** `encoder="cnn"` (`cnn`/`convnext`/`vit_tiny`/`vit_small`), `latent_dim=256`.
- **Training:** `batch_size=32`, `lr=1e-3`, `epochs=50`, `patience=10`, `rawboost_prob=0.7`,
  `num_workers=4`, `device="cuda"`, `save_dir="checkpoints"`.
- **Eval:** `eval_threshold` (optional).

`train.py` lets any `Config` field be overridden from the command line (`--<field>`).

---

## 5. Masking strategies (`features.py`)

- **contour** (`compute_contour_mask`): Sobel gradient magnitude, min-max normalized to [0,1], mask =
  `(normalized > threshold_db/100)`. Keeps high-gradient cells (edges/transitions), sets the rest to 0.
- **energy** (`compute_energy_mask`): mask = `(mel_db >= global_max - threshold_db)`. Keeps the loudest
  cells, sets the rest to 0.

The mask is applied as `mel * mask`: unkept cells become 0; the reconstruction target keeps the full mel.

---

## 6. Models (`models/`)

- **CNN AE** (~21M params): encoder = 4 conv blocks (1→32→64→128→256, each Conv-BN-GELU-MaxPool),
  flatten → FC to `latent_dim`; decoder = FC → transposed-conv mirror. `intermediate_features` returns
  the 4 block feature maps.
- **ConvNeXt AE:** ConvNeXt-style encoder + decoder; also exposes `intermediate_features`.
- **ViT-Tiny AE:** timm `vit_tiny_patch16_224` (in_chans=1) encoder; patch tokens are **mean-pooled to a
  single latent vector**, projected to `latent_dim`; a 4-layer transformer decoder reconstructs patch
  tokens, which are unpatchified to `[80,500]`.
- **ViT-Small AE:** larger ViT variant of the same scheme.

`forward(x)` returns `(reconstruction, latent)` for all models, with output cropped/padded to `[B,1,80,500]`.

---

## 7. Training (`train.py`)

- Primary train set: bonafide subset of `--train_dataset` (`asv19` or `atadd`).
- Validation: either `--val_frac > 0` (split a fraction of bonafide train as val) or a separate
  `--dev_root`/`--dev_protocol`.
- Optional extra bonafide training sources (concatenated): ASVspoof5, MLAAD, FamousFigures
  (`--asv5_train_root`, `--mlaad_train_root`, `--famousfigures_train_root` + protocols).
- Loss: `F.mse_loss(recon, target)` over the full mel.
- Optimizer/schedule: AdamW + OneCycleLR (`pct_start=0.1`). Early stopping on **bonafide validation
  reconstruction loss** with `patience`. Best checkpoint saved as `best.pt`; periodic `epoch_N.pt`.
- `--resume` restores model/optimizer/scheduler/epoch/best_val.

Example:
```bash
python train.py --train_protocol <p> --train_root <r> --val_frac 0.1 --encoder cnn
```

---

## 8. Evaluation (`evaluate.py`)

- `--checkpoint` loads the model + its saved `Config`.
- `--dataset` one of `asv19, asv5, itw, fakexpose, mlaad, famousfigures, atadd, all`.
- Scoring (`--score_type`):
  - `pixel` (default): `score = mean(|mel - recon|)`.
  - `feature`: weighted L2 distance between encoder feature maps of `mel` vs `recon`
    (`--feature_weights "1.0,0.8,0.4,0.2"`, optional per-channel L2 norm). **CNN/ConvNeXt only**;
    for ViT encoders it falls back to pixel scoring with a warning.
- Per dataset it writes a scores CSV, prints **EER / AUROC / threshold / n**, saves a score-distribution
  plot and per-sample error-map plots (reservoir sample of 10). For `fakexpose` and `famousfigures` it
  also makes a paired real/fake comparison plot.

Example:
```bash
python evaluate.py --checkpoint checkpoints/best.pt --dataset itw --root <r> --protocol <p>
```

---

## 9. Datasets (`dataset.py`)

All datasets return `(waveform[T], label, audio_name)` and support `subset ∈ {all, bonafide, spoof}`.

| Class | Source | Protocol format |
|---|---|---|
| `ASVspoof2019Dataset` | ASVspoof 2019 LA | whitespace; filename = `parts[0]` basename, label = `parts[2]`. |
| `ASVspoof5Dataset` | ASVspoof 5 | 6-col train/dev or 10-col eval TSV; label column auto-detected. |
| `InTheWildDataset` | In-The-Wild | headerless CSV `file,speaker,label`. |
| `FakeXposeDataset` | FakeXpose | directory-based: `Original/`→bonafide, `ElevenLabs/`(11labs)→spoof. |
| `MLAADDataset` | MLAAD (multilingual) | 5-col; `rel_path lang corpus src label`. |
| `FamousFiguresDataset` | FamousFigures | TSV header `AudioName Speaker Source Label AudioPath`. |
| `ATADDDataset` | AT-ADD Track 1 | CSV header `name,label,generator`; `real`→1, `fake`→0. |

`_load_audio` converts to mono, resamples to 16 kHz, then **repeat-pads (tiles) or truncates** to 5 s.

---

## 10. Known issues / things to verify (observations only)

These are factual, code-level observations. They are listed without proposed solutions.

1. **Decision threshold uses test labels.** `utils.compute_eer` selects the threshold at the EER
   operating point from the **labeled evaluation scores**, and `evaluate.py` recomputes it **per
   dataset** (`evaluate.py:214`). No threshold is derived from train/dev data alone. EER/AUROC are
   reported per dataset; there is no single shared, label-free operating threshold.

2. **Mask is applied by zeroing, and loss/score are full-map.** `masked_input = mel * mask` sets
   unkept cells to 0 rather than removing them; both the training loss (`train.py:112`) and the eval
   score (`evaluate.py:112`) are computed over the **entire** `[80,500]` map, not over the masked cells
   specifically.

3. **Contour mask keeps high-gradient cells and zeros low-gradient (flat) cells** (`features.py
   compute_contour_mask`). The project's stated research interest concerns flat/over-smoothed regions;
   the interaction between this mask and that interest has not been empirically characterized in-repo.

4. **Fixed-length handling creates flat regions.** Short clips are repeat-padded (tiled) to 5 s
   (`dataset.py:30-34`) and mels are padded to 500 frames with `mel_db.min()` (`features.py:46-48`).
   These padded regions are constant/flat and their extent depends on the original clip length.

5. **Dynamic-range and normalization choices.** `AmplitudeToDB(top_db=80)` clamps the mel to an 80 dB
   window. The contour mask normalizes the Sobel magnitude by the **global** min/max of each utterance
   (`features.py:92-94`), where the min can be the padded floor.

6. **Feature-space scoring is encoder-specific.** Only `cnn` and `convnext` implement
   `intermediate_features`; `vit_tiny`/`vit_small` silently fall back to pixel scoring
   (`evaluate.py:311`).

7. **ViT autoencoders bottleneck through a single pooled vector.** Patch tokens are mean-pooled to one
   latent and the same vector is broadcast to every decoder query (`vit_ae.py:102,111-113`), so the
   decoder input carries no per-position information.

8. **RawBoost augments both input and target.** Augmentation is applied to the waveform before the mel
   is computed, and that single augmented mel is used for **both** `masked_input` and `target`
   (`train.py:96-109`). The objective is therefore reconstruction of the (augmented) mel from its masked
   version; it is not a clean-target denoising objective.

9. **Concatenated bonafide sources are unweighted.** When extra training sets are added they are joined
   with `ConcatDataset` and sampled uniformly by size (`train.py`), i.e. no per-source balancing.

10. **Model selection metric.** Early stopping uses bonafide validation **reconstruction loss**
    (`train.py`), which measures reconstruction quality on real speech, not real-vs-spoof separation
    (no spoof data is seen in training).

---

## 11. Quick orientation for a new reader

- Start with `README.md` (the idea) → `features.py` (mel + mask) → `train.py` (objective + data flow)
  → `evaluate.py` (scoring + metrics) → `models/cnn_ae.py` (the default model).
- The default configuration is CNN encoder, contour mask at 0.40, pixel scoring, RawBoost p=0.7.
- To reproduce any reported number, check which checkpoint's saved `Config` was used (it travels inside
  the checkpoint and is restored by `evaluate.load_checkpoint`).
