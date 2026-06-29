# Data Processing

This directory turns the raw Amazon / AntM2C data into sequence-aware
`TFRecord` files consumed by the training scripts. Both pipelines share the
final stage, [`build_tfrecords.py`](build_tfrecords.py).

Every TFRecord example contains:

| field            | dtype       | description                                  |
|------------------|-------------|----------------------------------------------|
| `text_features`  | float[T]    | item text embedding(s)                       |
| `image_features` | float[I]    | item image embedding                         |
| `id_feature`     | int64[2]    | `(user_id, item_id)`                         |
| `label`          | float[1]    | click label (0/1)                            |
| `domain`         | float[1]    | domain / scene id                            |
| `user_seq`       | int64[5]    | last 5 clicked item ids (left-padded with 0) |

`T = 384` (Amazon) / `768*6` (AntM2C); `I = 4096` (Amazon) / `512` (AntM2C).

## Amazon pipeline (`amazon/`)

| step | script | purpose |
|------|--------|---------|
| 1 | `step1_merge_and_label.py`            | merge 3 categories into domains, build binary labels |
| 2 | `step2_remap_and_split.py`            | remap ids, per-user chronological train/val/test split |
| 3 | `step3_extract_item_meta.py`          | join raw `meta_*.json.gz` metadata by item id |
| 4 | `step4_extract_text_image_features.py`| Sentence-BERT (MiniLM) text + published 4096-d image features |
| 5 | `../build_tfrecords.py --dataset amazon` | user sequences + per-domain TFRecords + shuffled train shards |

Feature encoders (paper Appendix A.1): **text** = Sentence-Transformers MiniLM
(384-d); **image** = the 4096-d visual features published with the McAuley
Amazon dataset (no image model needed, just the `image_features_*.b` files).

```bash
cd amazon
python step1_merge_and_label.py
python step2_remap_and_split.py
python step3_extract_item_meta.py
TEXT_ENCODER_PATH=../pretrained_models/all-MiniLM-L6-v2 python step4_extract_text_image_features.py
cd ..
python build_tfrecords.py --dataset amazon --csv-dir amazon/split_data --feature-dir amazon/data --output-dir amazon/data
```

## AntM2C pipeline (`antm2c/`)

| step | script | purpose |
|------|--------|---------|
| 1 | `step1_split_and_remap.py`         | parse raw logs, remap ids, split by date |
| 2 | `step2_extract_text_features.py`   | Multilingual BERT embeddings for the 6 textual fields |
| 3 | `step3_extract_image_features.py`  | Chinese-CLIP 512-d image features |
| 4 | `../build_tfrecords.py --dataset antm2c` | user sequences + per-domain TFRecords + shuffled train shards |

Feature encoders (paper Appendix A.1): **text** = Multilingual BERT (768-d);
**image** = Chinese-CLIP ViT-B/16 (512-d).

```bash
cd antm2c
python step1_split_and_remap.py
BERT_PATH=../pretrained_models/bert-base-multilingual-cased python step2_extract_text_features.py
CLIP_PATH=../pretrained_models/chinese-clip-vit-base-patch16 python step3_extract_image_features.py
cd ..
python build_tfrecords.py --dataset antm2c --csv-dir antm2c --feature-dir antm2c --output-dir antm2c
```

> The feature-extraction checkpoints are configured through the environment
> variables shown above. Download them into `pretrained_models/` (see
> [`pretrained_models/README.md`](../pretrained_models/README.md)) or point the
> variables directly at a HuggingFace hub id.
