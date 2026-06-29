# AMS: Adaptive Modality Scheduling for Industrial Multi-Scenario CTR Prediction

Official implementation of the KDD'26 paper
**"AMS: Adaptive Modality Scheduling for Industrial Multi-Scenario CTR Prediction"**.

> With the increasing diversity of recommendation scenarios and the widespread
> adoption of multi-modal information, jointly modeling multi-scenario and
> multi-modal data has become a critical trend in industrial CTR prediction.
> AMS is a lightweight framework that introduces an **adaptive modality mask
> generation** network, performing sample-level, on-demand modality scheduling
> to explicitly model scenario-specific users' modality preferences.

- Paper: <https://doi.org/10.1145/3770855.3818319>
- Code: <https://github.com/hdu-yzl/AMS>

## Method overview

Every sample carries an **ID** feature (user/item embeddings), a **text**
feature and an **image** feature, drawn from a specific *scenario* (domain).

1. **Modality feature initialization & alignment** — an embedding layer encodes
   ID features; pretrained encoders extract text/image features, which are then
   projected into a scenario-shared latent space.
2. **Cross-modal attention** — the ID representation acts as the query to
   enhance the text and image representations.
3. **Modality Mask Generation (MMG)** — a hypernetwork conditioned on the
   scenario and modality information produces per-modality activation
   probabilities, discretized into a binary mask with a *learnable threshold*
   and a *straight-through estimator (STE)*. Only the **text** and **image**
   modalities are masked; the ID modality is always kept as a stable anchor.
4. **Modality Attention Fusion (MAF)** — only the activated modalities are fused
   by a lightweight, pluggable attention module.
5. **MSR backbone** — the fused representation is fed to a decoupled
   multi-scenario backbone (an MLP by default) for CTR prediction.

Training warms up the mask network first (modalities disabled), then anneals a
temperature so the soft gates sharpen into hard 0/1 selections, regularized by a
KL term that balances modality activation.

The shipped fusion variants (`--model`) correspond to the fusion methods studied
in the paper:

| variant | fusion mechanism |
|---------|------------------|
| `dnn1`  | concatenation    |
| `dnn2`  | attention (MAF, default) |
| `dnn3`  | gate             |
| `dnn4`  | Transformer      |
| `dnn5`, `dnn6` | scenario-only soft-gate ablations (AntM2C) |

## Repository layout

```
.
├── train_amazon.py            # training entry point for Amazon (3 scenarios)
├── train_antm2c.py            # training entry point for AntM2C (5 scenarios)
├── modules/
│   ├── layers.py              # reusable building blocks (MLP, FM, cross net, ...)
│   ├── mask_models_amazon.py  # AMS mask-based fusion models (3 scenarios)
│   ├── mask_models_antm2c.py  # AMS mask-based fusion models (5 scenarios)
│   └── baseline_models.py     # FM / DNN / DeepFM / DCN / IPNN baselines
├── dataloader/
│   └── loader.py              # unified TFRecord loaders (AmazonLoader / Antm2cLoader)
├── utils/
│   └── train_utils.py         # device, logging and factory helpers
├── data_processing/           # raw data -> TFRecords (see its own README)
│   ├── build_tfrecords.py
│   ├── amazon/
│   └── antm2c/
└── pretrained_models/         # feature-extraction encoders (download here)
```

## Datasets

| dataset | scenarios | source |
|---------|-----------|--------|
| AntM2C  | S1–S5 (5) | <https://www.atecup.cn/dataSetDetailOpen/1> |
| Amazon  | S1–S3 (3): Sports & Outdoors, Movies & TV, Books | <http://jmcauley.ucsd.edu/data/amazon/links.html> |

Statistics of the processed datasets (paper Table 6):

| dataset | metric | S1 | S2 | S3 | S4 | S5 |
|---------|--------|----|----|----|----|----|
| AntM2C  | users     | 34,569 | 46,717 | 67,133 | 28,407 | 14,930 |
| AntM2C  | items     | 24,973 | 11,663 | 10,549 | 17,497 | 20,655 |
| AntM2C  | instances | 653,257 | 2,757,994 | 5,698,072 | 1,269,308 | 812,354 |
| Amazon  | users     | 143,638 | 143,638 | 143,638 | – | – |
| Amazon  | items     | 699,060 | 147,608 | 104,037 | – | – |
| Amazon  | instances | 1,944,082 | 427,259 | 762,448 | – | – |

**Splitting protocol** (paper Appendix A.1):

- **AntM2C** — chronological split by interaction timestamp: the last 4 days are
  the test set, the preceding 2 days the validation set, and the rest training.
- **Amazon** — keep users with ≥ 5 interactions and only those shared across the
  three categories; binary labels use a rating threshold of 3; per user the most
  recent 10% of items are test, the preceding 10% validation, the rest training.

## Feature extraction encoders

Multi-modal features are produced by the pretrained encoders described in the
paper (Appendix A.1):

| dataset | modality | encoder | dim |
|---------|----------|---------|-----|
| AntM2C  | text     | Multilingual BERT | 768 (× 6 fields) |
| AntM2C  | image    | Chinese-CLIP (ViT-B/16) | 512 |
| Amazon  | text     | Sentence-Transformers MiniLM | 384 |
| Amazon  | image    | published 4096-d visual features (McAuley) | 4096 |

Download the encoders into [`pretrained_models/`](pretrained_models/README.md)
(or set `BERT_PATH` / `CLIP_PATH` / `TEXT_ENCODER_PATH` to a HuggingFace hub id).

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

The data loaders read `TFRecord` files via TensorFlow and feed PyTorch tensors
to the models, so both frameworks are required.

## Data preparation

See [`data_processing/README.md`](data_processing/README.md) for the full,
step-by-step pipeline of each dataset. After running it you will have, per
dataset directory:

```
train_shuffle_*.tfrecord   val[0-N].tfrecord   test[0-N].tfrecord
image_feature.npy          text_feature(s).npy
```

## Training

```bash
# Amazon (3 scenarios)
python train_amazon.py --model dnn2 --data_dir /path/to/Amazon --cuda 0

# AntM2C (5 scenarios)
python train_antm2c.py --model dnn2 --data_dir /path/to/AntM2C --cuda 0
```

### Default hyper-parameters (paper Appendix A.2)

| hyper-parameter | value |
|-----------------|-------|
| embedding dimension | 128 |
| batch size | 512 |
| optimizer | Adam |
| MLP backbone | [1024, 512, 256], BatchNorm, Xavier init |
| learning rate | search in {1e-3, 1e-4, 1e-5} |
| L2 regularization | search in {1e-5, 1e-6, 1e-7} |
| KL constraint weight λ | search in {0.001, 0.005, 0.01, 0.05, 0.1} |

See `--help` on each entry point for the full argument list (`--lr`, `--m_lr`,
`--thre`, `--lambda1`, `--warmup_rate`, `--final_temp`, `--cuda`, ...).

Training reports per-scenario and overall AUC / Weighted-AUC on the test split
and writes the per-scenario text/image activation rates to a `*_mask_log_*.txt`
file.

## Reproducibility

All entry points fix the random seed (`2025`) for `torch`, `numpy` and the
TensorFlow shuffle buffer.

## Citation

```bibtex
@inproceedings{hao2026ams,
  title     = {AMS: Adaptive Modality Scheduling for Industrial Multi-Scenario CTR Prediction},
  author    = {Hao, Kailiang and Yang, Chaohua and Zeng, Wei and Su, Jianan and
               Shen, Kaixin and Wu, Jingtong and Liu, Dugang and Tang, Xing and
               Bin, Jingyang and He, Xiuqiang and Ming, Zhong},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '26)},
  year      = {2026},
  doi       = {10.1145/3770855.3818319}
}
```
