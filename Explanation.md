# Code Walkthrough & Architecture Flow

This document explains, section by section, how the Transformer-from-scratch notebook works — from raw text to a trained translation model — along with block diagrams for the data pipeline, model internals, training loop, and inference process.

---

## 1. High-Level System Flow

```
┌───────────────┐     ┌────────────────┐     ┌──────────────────┐     ┌───────────────┐
│  Raw Dataset  │ ──► │  Preprocessing │ ──► │   Model Training  │ ──► │   Inference   │
│  (Multi30k)   │     │ (tokenize/vocab)│     │ (Encoder/Decoder) │     │ (Translate())│
└───────────────┘     └────────────────┘     └──────────────────┘     └───────────────┘
```

Each stage is expanded in the sections below.

---

## 2. Data Preparation Pipeline

### 2.1 Flow Diagram

```
 ┌────────────────────┐
 │  load_dataset()     │   Hugging Face "bentrevett/multi30k"
 │  train/valid/test   │
 └─────────┬───────────┘
           │
           ▼
 ┌────────────────────┐
 │ spaCy tokenizers    │   tokenize_en(), tokenize_de()
 │ (lowercased tokens) │
 └─────────┬───────────┘
           │
           ▼
 ┌────────────────────┐
 │  build_vocab()      │   Counter over all tokens
 │  min_freq = 2       │   + special tokens:
 │                     │     <PAD>=0 <UNK>=1 <BOS>=2 <EOS>=3
 └─────────┬───────────┘
           │
           ▼
 ┌────────────────────┐
 │  numericalize()     │   sentence → [<BOS>, id1, id2, ..., <EOS>]
 └─────────┬───────────┘
           │
           ▼
 ┌────────────────────┐
 │ TranslationDataset  │   wraps HF dataset + returns tensor pairs
 └─────────┬───────────┘
           │
           ▼
 ┌────────────────────┐
 │  collate_fn()       │   pads variable-length sequences in a batch
 │  + DataLoader        │   using <PAD> token, batch_first=True
 └────────────────────┘
```

### 2.2 What each function does

| Function / Class      | Purpose |
|------------------------|---------|
| `tokenize_en` / `tokenize_de` | Splits raw sentences into lowercase word tokens using spaCy's tokenizer (no full NLP pipeline — only tokenization is used for speed). |
| `build_vocab`           | Builds a word→index mapping per language. Words appearing fewer than `min_freq` times are dropped and mapped to `<UNK>` at inference time. |
| `numericalize`          | Converts a tokenized sentence into a list of vocabulary indices, wrapped with `<BOS>` and `<EOS>` markers. |
| `TranslationDataset`    | A `torch.utils.data.Dataset` that numericalizes an (English, German) pair on the fly when indexed. |
| `collate_fn`            | Pads all sequences in a batch to the same length using `pad_sequence`, so they can be stacked into a single tensor per batch. |

### 2.3 Example transformation

```
"Two young, White males are outside near many bushes."
        │  tokenize_en()
        ▼
['two', 'young', ',', 'white', 'males', 'are', 'outside', 'near', 'many', 'bushes', '.']
        │  numericalize()
        ▼
[2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 3]
  ▲                                        ▲
 <BOS>                                   <EOS>
```

---

## 3. Masking Utilities

Two masks control what each token in the model is allowed to "see" during attention.

```
┌─────────────────────────────┐        ┌───────────────────────────────────┐
│         SOURCE MASK          │        │            TARGET MASK              │
│                              │        │                                     │
│  make_src_mask(src, PAD_IDX) │        │  make_tgt_mask(tgt, PAD_IDX)        │
│  → hides <PAD> tokens only   │        │  → hides <PAD> tokens               │
│                              │        │  → AND hides future tokens          │
│                              │        │     (causal / look-ahead mask)      │
└─────────────────────────────┘        └───────────────────────────────────┘
```

- **Source mask**: a boolean mask that blocks attention to padding tokens in the input sentence, so the encoder never attends to meaningless `<PAD>` positions.
- **Target mask**: combines a padding mask **and** a lower-triangular causal mask (`torch.tril`), so that when predicting token *t*, the decoder can only attend to tokens `1...t` — never future tokens. This is what makes the decoder auto-regressive during training.

```
Causal mask example (tgt_len = 4):
       token1 token2 token3 token4
token1   ✓      ✗      ✗      ✗
token2   ✓      ✓      ✗      ✗
token3   ✓      ✓      ✓      ✗
token4   ✓      ✓      ✓      ✓
```

---

## 4. Model Architecture

### 4.1 Overall Block Diagram

```
                       ┌──────────────────────────┐
    src (English) ───► │      TransformerEncoder   │
                       │  Embedding + PosEnc        │
                       │  ┌──────────────────────┐  │
                       │  │   EncoderBlock × 6    │  │
                       │  └──────────────────────┘  │
                       └────────────┬─────────────┘
                                    │ encoder_output
                                    ▼
                       ┌──────────────────────────┐
    tgt (German)  ───► │      TransformerDecoder   │
                       │  Embedding + PosEnc        │
                       │  ┌──────────────────────┐  │
                       │  │   DecoderBlock × 6    │◄─┼── encoder_output (cross-attn)
                       │  └──────────────────────┘  │
                       └────────────┬─────────────┘
                                    │
                                    ▼
                       ┌──────────────────────────┐
                       │  Linear (fc) → vocab_size │
                       └──────────────────────────┘
                                    │
                                    ▼
                          logits over German vocab
```

### 4.2 PositionalEncoding

Since attention has no built-in sense of token order, fixed sinusoidal position vectors are added to each token embedding.

```
PositionalEncoding(d_model, max_len)
    pe[pos, 2i]   = sin(pos / 10000^(2i/d_model))
    pe[pos, 2i+1] = cos(pos / 10000^(2i/d_model))

forward(x):  x + pe[:, :seq_len]
```

### 4.3 MultiHeadAttention (Self-Attention)

Used inside both the encoder (on the source) and the decoder (masked, on the target).

```
Input x (batch, seq_len, d_model)
        │
        ├──► Linear(w_q) ──► split into n_heads ──┐
        ├──► Linear(w_k) ──► split into n_heads ──┼──► scaled dot-product
        └──► Linear(w_v) ──► split into n_heads ──┘        attention
                                                       │
                                              softmax(QKᵀ/√d_k) · V
                                                       │
                                          concat heads → Linear(w_o)
                                                       │
                                                       ▼
                                            Output (batch, seq_len, d_model)
```

Masking is applied by setting masked-out attention scores to `-1e9` before the softmax, effectively giving them ~0 probability.

### 4.4 CrossAttention

Structurally identical to `MultiHeadAttention`, but Query comes from the **decoder**, while Key and Value come from the **encoder output**. This is how the decoder "looks at" the source sentence while generating the target sentence.

```
Query  ← decoder hidden state
Key    ← encoder_output
Value  ← encoder_output
      │
      ▼
scaled dot-product attention (masked by src_mask)
```

### 4.5 FeedForward

A simple two-layer MLP applied independently to each position:

```
Linear(d_model → 4×d_model) → ReLU → Linear(4×d_model → d_model)
```

### 4.6 EncoderBlock

```
        x
        │
        ├────────────────┐
        ▼                │
  MultiHeadAttention      │ (residual)
        │                │
        ▼◄───────────────┘
      Add
        │
        ▼
    LayerNorm (norm1)
        │
        ├────────────────┐
        ▼                │
   FeedForward            │ (residual)
        │                │
        ▼◄───────────────┘
      Add
        │
        ▼
    LayerNorm (norm2)
        │
        ▼
    block output
```

### 4.7 DecoderBlock

```
        x (target embeddings)
        │
        ├───────────────────┐
        ▼                   │
 Masked Self-Attention       │ (residual, uses tgt_mask)
        │                   │
        ▼◄──────────────────┘
      Add → LayerNorm (norm1)
        │
        ├───────────────────┐
        ▼                   │
  Cross-Attention             │ (residual, Q=decoder, K/V=encoder_output,
        │                   │  uses src_mask)
        ▼◄──────────────────┘
      Add → LayerNorm (norm2)
        │
        ├───────────────────┐
        ▼                   │
   FeedForward                │ (residual)
        │                   │
        ▼◄──────────────────┘
      Add → LayerNorm (norm3)
        │
        ▼
   block output
```

### 4.8 TransformerEncoder / TransformerDecoder

Both wrap an `nn.Embedding` layer, scale embeddings by `√d_model` (as in the original paper), add positional encodings, and then pass the result through a stack of `num_layers` blocks (`EncoderBlock` or `DecoderBlock`).

### 4.9 Full Transformer Module

```python
class Transformer(nn.Module):
    def forward(self, src, tgt, src_mask, tgt_mask):
        encoder_output = self.Encoder(src, src_mask)
        decoder_output = self.Decoder(tgt, encoder_output, src_mask, tgt_mask)
        return self.fc(decoder_output)   # logits over target vocabulary
```

This ties the encoder and decoder together and projects the final decoder representation to vocabulary-sized logits.

---

## 5. Training Loop Flow

```
for each epoch:
   ┌───────────────────────────────────────────────────────────┐
   │ TRAIN PHASE                                                │
   │  for each batch (src, tgt):                                │
   │     tgt_input  = tgt[:, :-1]      (teacher forcing input)  │
   │     tgt_output = tgt[:, 1:]       (shifted target labels)  │
   │     src_mask = make_src_mask(src)                          │
   │     tgt_mask = make_tgt_mask(tgt_input)                    │
   │     logits = model(src, tgt_input, src_mask, tgt_mask)     │
   │     loss = CrossEntropyLoss(logits, tgt_output)            │
   │     loss.backward() → optimizer.step()                     │
   └───────────────────────────────────────────────────────────┘
   ┌───────────────────────────────────────────────────────────┐
   │ VALIDATION PHASE (no gradients)                            │
   │  same forward pass as above, plus:                         │
   │     predictions = argmax(logits)                            │
   │     accuracy = correct non-pad predictions / total non-pad │
   └───────────────────────────────────────────────────────────┘
           │
           ▼
   record train_loss, val_loss, val_accuracy for this epoch
           │
           ▼
   after all epochs → plot loss curves and accuracy curve
```

**Key implementation detail — teacher forcing:** During training, the decoder is fed the *ground-truth* previous tokens (`tgt_input`) rather than its own previous predictions. The causal mask ensures it can't "cheat" by seeing future tokens, so it still has to learn to predict each next word from what came before.

---

## 6. Inference Flow (`translate()`)

Unlike training, inference has no ground-truth target — the decoder must generate tokens one at a time, feeding its own previous output back in (autoregressive / greedy decoding).

```
                     ┌─────────────────────────┐
 input sentence ───► │ numericalize + encode    │
                     │ encoder_output = Encoder(│
                     │      src, src_mask)      │
                     └────────────┬─────────────┘
                                  │
                     decoder_input = [<BOS>]
                                  │
                     ┌────────────▼─────────────┐
                     │  LOOP (max_len times):     │
                     │   1. tgt_mask = causal mask │
                     │      for current length     │
                     │   2. decoder_output =       │
                     │      Decoder(decoder_input, │
                     │        encoder_output, ...) │
                     │   3. logits = fc(decoder_   │
                     │      output)                │
                     │   4. next_token = argmax(   │
                     │      logits[:, -1])         │
                     │   5. append next_token to    │
                     │      decoder_input           │
                     │   6. stop if next_token ==   │
                     │      <EOS>                   │
                     └────────────┬─────────────┘
                                  │
                                  ▼
                  decode ids back to words → translated sentence
```

### Example

```
input  : "A man is playing football"
   │
   ▼ numericalize (English vocab)
[2, ...ids..., 3]
   │
   ▼ Encoder
encoder_output
   │
   ▼ greedy decode loop, one German token at a time
"ein" → "ein mann" → "ein mann spielt" → "ein mann spielt football" → "ein mann spielt football ."
   │
   ▼
output : "ein mann spielt football ."
```

---

## 7. End-to-End Summary Diagram

```
┌─────────────┐   tokenize    ┌─────────────┐   numericalize   ┌─────────────┐
│  Raw text   │ ────────────► │   Tokens    │ ───────────────► │   Tensor ids │
└─────────────┘               └─────────────┘                  └──────┬──────┘
                                                                        │ batch + pad
                                                                        ▼
                                                                 ┌──────────────┐
                                                                 │ src, tgt      │
                                                                 │ batches       │
                                                                 └──────┬───────┘
                                                                        │
                                              ┌─────────────────────────┼─────────────────────────┐
                                              ▼                         ▼                          │
                                     ┌─────────────────┐       ┌──────────────────┐                │
                                     │ make_src_mask()  │       │ make_tgt_mask()   │                │
                                     └────────┬────────┘       └─────────┬────────┘                │
                                              │                          │                          │
                                              ▼                          ▼                          │
                                     ┌───────────────────────────────────────────┐                  │
                                     │              Transformer model              │◄─────────────────┘
                                     │   Encoder(6 layers) → Decoder(6 layers)     │
                                     │              → Linear → logits              │
                                     └───────────────────────┬───────────────────┘
                                                              │
                                     ┌────────────────────────┼───────────────────────┐
                                     ▼                                                ▼
                          ┌───────────────────┐                          ┌────────────────────┐
                          │  Training: compute │                          │  Inference: greedy  │
                          │  loss, backprop,    │                          │  decode token by     │
                          │  update weights     │                          │  token until <EOS>   │
                          └───────────────────┘                          └────────────────────┘
```

---

## 8. Quick Reference — Class/Function Index

| Name                     | Role                                                     |
|---------------------------|-----------------------------------------------------------|
| `tokenize_en`, `tokenize_de` | spaCy-based tokenizers for English and German            |
| `build_vocab`              | Builds vocabulary dictionaries from token frequencies      |
| `numericalize`             | Converts tokens to ids with `<BOS>`/`<EOS>` markers        |
| `TranslationDataset`       | PyTorch `Dataset` wrapper for the parallel corpus           |
| `collate_fn`               | Pads batches to equal length                                |
| `make_src_mask`            | Builds the source padding mask                              |
| `make_tgt_mask`            | Builds the target padding + causal mask                     |
| `PositionalEncoding`       | Adds sinusoidal position information to embeddings           |
| `MultiHeadAttention`       | Self-attention used in encoder and (masked) decoder          |
| `CrossAttention`           | Decoder-to-encoder attention                                 |
| `FeedForward`              | Position-wise two-layer MLP                                  |
| `EncoderBlock`             | Self-attention + feed-forward + residuals + norms             |
| `DecoderBlock`             | Masked self-attention + cross-attention + feed-forward         |
| `TransformerEncoder`       | Embedding + positional encoding + stack of `EncoderBlock`s     |
| `TransformerDecoder`       | Embedding + positional encoding + stack of `DecoderBlock`s     |
| `Transformer`              | Combines encoder, decoder, and final output projection         |
| `translate`                | Greedy-decoding inference function                            |
