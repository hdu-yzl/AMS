# Pretrained Feature-Extraction Models

The data-processing pipeline turns raw text/images into dense embeddings using
the pretrained encoders described in the AMS paper (Appendix A.1). Download the
models into the corresponding sub-directories below (or set the matching
environment variable to a HuggingFace hub id).

| dataset | modality | model | dim | folder | env var |
|---------|----------|-------|-----|--------|---------|
| AntM2C  | text     | Multilingual BERT          | 768  | `bert-base-multilingual-cased/`   | `BERT_PATH`         |
| AntM2C  | image    | Chinese-CLIP (ViT-B/16)    | 512  | `chinese-clip-vit-base-patch16/`  | `CLIP_PATH`         |
| Amazon  | text     | Sentence-Transformers MiniLM | 384 | `all-MiniLM-L6-v2/`              | `TEXT_ENCODER_PATH` |

> Amazon **image** features are NOT extracted here: the pipeline reuses the
> 4096-d visual features published with the McAuley Amazon dataset, so no image
> model is required for Amazon.

## How to download

Using the HuggingFace CLI (recommended):

```bash
pip install -U "huggingface_hub[cli]"

# AntM2C text encoder (Multilingual BERT)
huggingface-cli download bert-base-multilingual-cased \
    --local-dir pretrained_models/bert-base-multilingual-cased

# AntM2C image encoder (Chinese-CLIP)
huggingface-cli download OFA-Sys/chinese-clip-vit-base-patch16 \
    --local-dir pretrained_models/chinese-clip-vit-base-patch16

# Amazon text encoder (Sentence-BERT / MiniLM)
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
    --local-dir pretrained_models/all-MiniLM-L6-v2
```

Alternatively, skip the download and point the env vars directly at the hub ids,
e.g. `BERT_PATH=bert-base-multilingual-cased python step2_extract_text_features.py`.
