# Multimodal Aspect-Based Sentiment Classification via Contrastive Filtration and Cross-Modal Distributional Alignment

Official PyTorch implementation of CFDA for the multimodal aspect-based sentiment classification (MABSC) task.

## Abstract

In the multimodal aspect-based sentiment classification task, most existing methods assume that the paired image contains the object referred to by the aspect term, prioritizing cross-modal interaction while overlooking aspect-irrelevant visual regions and cross-modal distributional differences. We propose Contrastive Filtration and Cross-Modal Distributional Alignment (CFDA), a unified framework that filters out aspect-irrelevant visual content and aligns the two modalities at the distribution level. Firstly, a multi-grained visual representation is built from a global caption and local region features. Secondly, contrastive-guided filtration with an adaptive residual gate suppresses aspect-irrelevant visual responses. Finally, the filtered visual features are aligned with aspect-specific text features by minimizing an entropy-regularized Wasserstein distance, stabilized by bidirectional consistency regularization. Extensive experiments on Twitter-2015 and Twitter-2017 demonstrate that CFDA consistently outperforms representative text-only, visual-only, and multimodal baselines.

## Overview

CFDA consists of four main modules:

1. **Multi-grained Visual Representation Module**. The CATR image captioning model generates a global caption for each image, and the bottom-up-attention detector provides 36 local region features per image. The pre-trained language model encodes the text side.
2. **Contrastive-guided Filtration Module**. An adaptive residual gate suppresses aspect-irrelevant visual responses under the guidance of contrastive signals between text and vision.
3. **Cross-modal Distributional Alignment Module**. The filtered visual features are aligned with aspect-specific text features by minimizing an entropy-regularized Wasserstein distance, which is stabilized by bidirectional consistency regularization.
4. **Prediction Module**. The fused multimodal features are passed to a softmax layer for aspect-level sentiment prediction.

## Running Steps

### 1. Environment

```bash
conda create -n cfda python=3.10 -y
conda activate cfda
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

PyTorch is pinned to the CUDA 11.8 build. If your GPU or driver differs, install the PyTorch version that matches your CUDA from the [official PyTorch site](https://pytorch.org/get-started/locally/) instead.

### 2. Pre-trained Language Model

Download the pre-trained DeBERTa model from [Hugging Face](https://huggingface.co/) and place it under `./deberta`. You can also point `--model` to any local model path.

### 3. Datasets

Download the Twitter-2015 and Twitter-2017 MABSC datasets released by the [IJCAI 2019 proceedings](https://www.ijcai.org/proceedings/2019/) and place them under `./dataset`. Each dataset contains the images together with the `train.tsv`, `dev.tsv` and `test.tsv` files.

### 4. CATR Checkpoint

Download the CATR checkpoint and place it at `./catr/checkpoint.pth`. The checkpoint is the same one used by [TASG](https://github.com/zhenzhen313/TASG), which provides the download link in its repository.

### 5. Region Features

Prepare the bottom-up-attention region features and place them under `./region_box`:

- `region_box/15_twitter/_box` and `region_box/15_twitter/_att` for Twitter-2015
- `region_box/17_twitter/_box` and `region_box/17_twitter/_att` for Twitter-2017

Each image is represented by 36 region features of dimension 2048 together with the corresponding bounding boxes. The feature extraction code is available at the [bottom-up-attention repository](https://github.com/peteanderson80/bottom-up-attention).

### 6. Training

```bash
bash scirpts/train15.sh   # train on Twitter-2015
bash scirpts/train17.sh   # train on Twitter-2017
```

The trained weights are saved as `model_best.pth` according to the best macro F1 on the test set.

## Citation

If you find this work helpful, please cite:

```bibtex
@inproceedings{cfda,
  title     = {Multimodal Aspect-Based Sentiment Classification via Contrastive Filtration and Cross-Modal Distributional Alignment},
  author    = {Anonymous},
  booktitle = {Proceedings of COLING},
  year      = {2026},
  note      = {Citation will be updated after acceptance}
}
```
