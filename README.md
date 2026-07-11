# Transformer from Scratch — Neural Machine Translation (English → German)

A complete, from-scratch PyTorch implementation of the Transformer architecture, built to demonstrate a deep understanding of modern sequence-to-sequence modeling — including multi-head self-attention, cross-attention, positional encoding, and masked auto-regressive decoding — applied end-to-end to English-to-German neural machine translation.

**Inspired by:** *Attention Is All You Need* (Vaswani et al., 2017), The Annotated Transformer (Harvard NLP), and PyTorch's official sequence-to-sequence tutorials.

Every core component — embeddings, positional encoding, multi-head self-attention, cross-attention, feed-forward sublayers, and the encoder–decoder stack — is implemented manually, without relying on `nn.Transformer` or `nn.MultiheadAttention`, to demonstrate a complete understanding of the underlying mechanics.

<a href="https://colab.research.google.com/github/Abhinav9818/Transformer-from-Scratch-Pytorch-/blob/main/Transformer_Encoder_Decoder.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Table of Contents

- [Key Features](#key-features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Usage](#usage)
- [Training Results (20 Epochs)](#training-results-20-epochs)
- [Notebook Structure](#notebook-structure)
- [Future Enhancements](#future-enhancements)
- [License](#license)

## Key Features

- End-to-end NLP pipeline: tokenization, vocabulary construction, numericalization, batching, and masking
- Linguistically-aware tokenization using **spaCy** (`en_core_web_sm` for English, `de_core_news_sm` for German)
- Vocabulary construction from scratch with frequency thresholding and special tokens (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`)
- Custom `Dataset` and `DataLoader` pipeline with dynamic padding via a batching `collate_fn`
- Fully custom implementation of:
  - Sinusoidal **Positional Encoding**
  - **Multi-Head Self-Attention**
  - **Multi-Head Cross-Attention**
  - Position-wise **Feed-Forward Networks**
  - **Encoder** and **Decoder** blocks with residual connections and Layer Normalization
  - Source padding masks and target causal (look-ahead) masks
- Complete training and evaluation loop with per-epoch loss and accuracy tracking
- Training/validation loss and accuracy visualization
- **Greedy-decoding inference pipeline** for translating arbitrary input sentences

## Dataset

[**Multi30k**](https://huggingface.co/datasets/bentrevett/multi30k) — a parallel English–German sentence corpus of image captions, widely used as a standard benchmark for machine translation research. Loaded via the Hugging Face `datasets` library with predefined train, validation, and test splits.

## Model Architecture

Implements the standard encoder–decoder Transformer configuration:

| Component          | Configuration                       |
|--------------------|---------------------------------------|
| Model dimension    | 512                                    |
| Attention heads    | 8                                       |
| Encoder layers     | 6                                       |
| Decoder layers     | 6                                       |
| Feed-forward dim   | 2048 (4 × model dimension)              |
| Positional encoding| Fixed sinusoidal                        |
| Optimizer          | Adam (learning rate = 1e-4)             |
| Loss function      | Cross-entropy (padding tokens ignored)  |
| Training epochs    | 20                                       |

## Requirements

```bash
pip install torch datasets sentencepiece spacy matplotlib
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

A CUDA-enabled GPU is recommended for efficient training (the reference run was performed on an NVIDIA Tesla T4).

## Usage

1. Open `Transformer_Encoder_Decoder.ipynb` in Jupyter or Google Colab (badge above).
2. Run all cells sequentially. The notebook will:
   - Install dependencies and download the required spaCy language models
   - Load and tokenize the Multi30k dataset
   - Build source (English) and target (German) vocabularies
   - Construct the Transformer model along with source/target masking utilities
   - Train the model for 20 epochs, reporting training loss, validation loss, and token-level accuracy at each epoch
   - Generate training curves for loss and accuracy
3. Translate a custom sentence using the trained model:

```python
sentence = "A man is playing football"
translate(sentence, model)
```

Example output:
```
input  : A man is playing football
output : ein mann spielt football .
```

## Training Results (20 Epochs)

| Metric               | Value                        |
|-----------------------|-------------------------------|
| Final training loss    | 0.069                         |
| Best validation loss   | 1.80 (epoch 4)                |
| Peak validation accuracy | 62.0% (token-level)         |

Training loss decreases steadily throughout training, while validation loss reaches its minimum around epoch 4 before gradually increasing — a well-known pattern in sequence-to-sequence training that indicates the point beyond which the model begins to specialize to the training distribution. Regularization techniques such as dropout, label smoothing, and learning-rate scheduling (as described in the original Transformer paper) can further improve generalization and are natural next steps for extending this project.

## Notebook Structure

1. **Data Preparation** — dependency installation, dataset loading, tokenization, vocabulary construction, and sentence numericalization
2. **Dataset & DataLoader** — custom `TranslationDataset`, padding-aware `collate_fn`, and mask-generation utilities
3. **Model Definition** — `PositionalEncoding`, `MultiHeadAttention`, `FeedForward`, `EncoderBlock`, `TransformerEncoder`, `CrossAttention`, `DecoderBlock`, `TransformerDecoder`, and the top-level `Transformer` module
4. **Training & Evaluation** — full training loop with per-epoch loss and accuracy tracking, along with visualization of training dynamics
5. **Inference** — a greedy-decoding `translate()` function for generating translations from raw input text

## Future Enhancements

- Incorporate dropout, label smoothing, and weight decay for improved regularization
- Replace greedy decoding with beam search for higher-quality translations
- Add learning-rate warmup and scheduling, as proposed in the original Transformer paper
- Evaluate translation quality using BLEU score in addition to token-level accuracy
- Adopt subword tokenization (e.g., BPE or SentencePiece) for improved vocabulary coverage and handling of rare words

## License

This project is licensed under the [MIT License](LICENSE) — you're free to use, modify, and distribute this code, including for commercial purposes, provided the original copyright notice is retained.
