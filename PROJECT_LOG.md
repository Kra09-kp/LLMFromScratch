# Transformer From Scratch — Learning Journal & Mentor Guide

## Mentor Role & Pedagogical Contract
- **Role**: Personal Professor & Socratic Mentor.
- **Strict Rule**: **NO CODE WRITING**. Under no circumstances should the assistant write code for the student in notebooks, files, or chat.
- **Allowed Assistance**:
  - Conceptual deep dives and first-principles intuition.
  - Mathematical equations and step-by-step reasoning.
  - Tensor shapes, matrix dimensions, and visualization of operations.
  - Hints pointing to relevant PyTorch functions, module names, or design patterns.
  - Socratic questioning to debug, test edge cases, and verify understanding.

---

## Roadmap & Milestones

### Day 1: Tokenization from Scratch
- **Status**: Completed
- **Artifacts**: `notebooks/TokenizerFromScratch.ipynb`, `notebooks/mytokenizer.model`
- **Concepts Learned**: Token vocabulary, byte-pair encoding / subword tokenization mechanics, mapping tokens to IDs.

### Day 2: Embeddings & Single-Head Attention
- **Status**: Completed
- **Artifacts**: `notebooks/CasualHeadAttention.ipynb`
- **Concepts Learned**:
  - `ModelParameters` dataclass (vocab size, `d_model`, `max_seq_len`, etc.).
  - Token embedding lookup (`nn.Embedding`) & positional embeddings.
  - Implemented single-head causal self-attention with scaled dot-product and lower triangular causal masking.

### Day 3: Multi-Head Attention (Completed! 🎉)
- **Status**: Completed
- **Artifacts**: `notebooks/MultiHeadAttention.ipynb`
- **Accomplished**:
  - Clarified embedding layers as learnable 2D lookup tables rather than MLP layers.
  - Deep-dive into Scaled Dot-Product math: why scaling by $\sqrt{d_k}$ prevents softmax saturation and vanishing gradients.
  - Mastered 4D tensor surgery for Multi-Head Attention:
    - Splitting $d_{\text{model}} \rightarrow (\text{num\_heads}, \text{head\_dim})$.
    - Why Transpose #1 (`transpose(1, 2)`) enables tokens attending to tokens $(T \times T)$ per head.
    - Why Transpose #2 (`transpose(-2, -1)` on $K$) satisfies matrix multiplication $(T \times d_k) @ (d_k \times T) \rightarrow (T \times T)$.
    - Parallel causal mask broadcasting with `register_buffer`.
    - Concatenation via `.transpose(1, 2).contiguous().view(x.shape)` and final projection through $W_O$.
  - Encapsulated into clean, robust `MultiHeadAttention(nn.Module)` and verified with input `(2, 5, 32)` $\rightarrow$ output `(2, 5, 32)`.

### Day 4: Transformer Decoder Block & Normalization (Completed! 🎉)
- **Status**: Completed
- **Artifacts**: `notebooks/DecoderBlock.ipynb`
- **Accomplished**:
  - Consolidated architecture pipeline into `DecoderBlock.ipynb`.
  - Explored architectural evolution: Decoder-only vs. Encoder-Decoder (why cross-attention is removed).
  - Pre-LN vs. Post-LN: calculus of the residual shortcut ($1 + \frac{\partial F}{\partial x}$) and stabilizing deep models.
  - Implemented `FeedForward(nn.Module)`: $d_{\text{model}} \rightarrow 4 \times d_{\text{model}} \rightarrow d_{\text{model}}$ using GELU.
  - Implemented `DecoderBlock(nn.Module)` with Pre-LN residual streams:
    - $x \leftarrow x + \text{Attention}(\text{LayerNorm}_1(x))$
    - $x \leftarrow x + \text{FFN}(\text{LayerNorm}_2(x))$
  - Verified tensor shapes `(2, 5, 32)` $\rightarrow$ `(2, 5, 32)`.
  - Calculated exact theoretical parameter count ($12,576$) and confirmed with `p.numel()`.
  - Added architectural tree print and Mermaid flowchart.

### Day 5: Full Generative Transformer & Text Generation (Completed! 🎉)
- **Status**: Completed
- **Artifacts**: `notebooks/Transformer.ipynb`
- **Accomplished**:
  - Assembled the full GPT-style decoder architecture in `Transformer(nn.Module)`.
  - Stacked $N$ `DecoderBlock` layers using `nn.ModuleList`.
  - Added final LayerNorm before the language model projection head.
  - Implemented **Weight Tying** (`lm_head.weight = embedding_layer.token_embeddings.weight`):
    - Understood how reading and predicting share the same continuous semantic space.
    - Traced how PyTorch Autograd accumulates gradients ($\frac{\partial \mathcal{L}}{\partial W}_{\text{total}} = \text{grad}_{\text{lm}} + \text{grad}_{\text{emb}}$) to update the shared tensor.
  - Implemented forward pass with Cross-Entropy Loss computation:
    - Mastered why continuous probabilities via Softmax enable gradient flow, unlike discrete `argmax`.
    - Reshaped 3D logits $(B, T, V) \rightarrow (B \cdot T, V)$ and targets $(B, T) \rightarrow (B \cdot T)$ using `.view()`.
  - Investigated weight initialization:
    - Discovered why unscaled default embedding weights caused initial loss explosion ($\approx 21$).
    - Implemented `_init_weights` with $\mathcal{N}(0, 0.02)$ via `self.apply()`.
    - Verified initial loss matches theoretical prediction: $\ln(44) \approx 3.784 \leftrightarrow 3.82$.
  - Calculated exact theoretical parameter count ($26,784$) and verified with `p.numel()`.
  - Built the auto-regressive `.generate()` engine:
    - Context window cropping (`idx_cond = idx[:, -max_seq_len:]`) to respect causal length.
    - Softmax over last-token logits and creative sampling via `torch.multinomial`.
    - Appending generated tokens via `torch.cat`.

### Day 6: Training Pipeline, Dataset & Optimization (Completed! 🎉)
- **Status**: Completed
- **Artifacts**: `data/kirti.txt`, `notebooks/DatasetCreation.ipynb`, `notebooks/Training.ipynb`
- **Accomplished**:
  - **Custom Corpus Creation**:
    - Created personal biography & Q&A corpus (`data/kirti.txt`) containing 2,237 characters.
    - Extracted 50 unique characters ($V = 50$) and built deterministic `stoi`, `itos`, `encode`, and `decode`.
  - **Dataset Engineering & Boundary Mechanics**:
    - Avoided random token shuffling to preserve temporal causality and language continuity.
    - Discovered severed word boundary edge cases (severing "engineer") and snapped train/val split cleanly to paragraph boundaries (`rfind('\n\n')`).
    - Encoded into `torch.long` tensors for PyTorch embedding layer compatibility.
  - **The Shifted-Target Contract ($x \rightarrow y$)**:
    - Proved through Socratic first principles why loop-based prefix extraction is mathematically unified by the Causal Attention Mask.
    - Implemented clean batch generator `get_batch(split, batch_size, block_size)` with random slice indexing and `torch.stack()`.
  - **Autograd & Optimization Internal Architecture**:
    - Demystified how `loss.backward()` and `optimizer.step()` communicate via the shared parameter `p.grad` mailbox without explicit argument passing.
    - Verified starting cross-entropy loss against theoretical entropy: $\ln(50) \approx 3.912 \leftrightarrow 3.89$.
    - Implemented single-step verification showing immediate gradient descent ($3.89 \rightarrow 3.75$).
  - **Rigorous Evaluation & Training Loop**:
    - Designed `calculate_loss()` with `@torch.no_grad()` and `model.eval()` / `model.train()` toggling to disable dropout during evaluation.
    - Implemented complete iteration-based training loop with `torch.optim.AdamW(model.parameters(), lr=1e-3)`.
    - Trained for 2,500 iterations, driving train loss from **3.85 down to 0.89** and validation loss to **1.49**.
  - **Generative Intelligence Emergence**:
    - Successfully prompted the trained model to answer questions about Kirti Pogra.
    - Observed structural emergence: Q&A formatting (`Q:` $\rightarrow$ `A:`), name recognition ("Kirti Pogra"), degree ("MCA"), career ("AI/ML Engineer"), and interests ("space", "documentaries").

### Day 7: Generation Refinement, Sampling Controls & Checkpointing (Completed! 🎉)
- **Status**: Completed
- **Accomplished**:
  - **Temperature Sampling ($T$)**:
    - Understood entropy scaling: dividing logits before softmax preserves the probability simplex ($\sum P_i = 1$) while controlling distribution sharpness.
    - Handled zero temperature ($T = 0$) using deterministic `torch.argmax` greedy decoding without wasteful softmax/multinomial computation.
  - **Top-$k$ Sampling & Logit Masking**:
    - Mastered threshold cutoff mechanics: isolating top-$k$ scores via `torch.topk(last_token, k=top_k)`.
    - Applied negative infinity masking (`masked_fill(last_token < cutoff, float("-inf"))`) so non-top-$k$ tokens receive zero probability mass ($e^{-\infty} = 0$).
  - **Self-Contained Model Checkpointing**:
    - Analyzed architecture serialization: saving raw weights (`state_dict`) vs fragile pickled classes.
    - Demystified PyTorch 2.6+ `weights_only=True` security protocols and safe deserialization.
    - Bundled model parameters, architecture hyperparameters (`ModelConfig`), and tokenization mappings (`string_to_int`, `int_to_string`) into a unified `.pt` checkpoint file (`transformer_model.pt`).
    - Verified round-trip restoration and inference.

---

### Day 8: Scaling Up on Kaggle — Pre-training a Mini-LLM on GPU (Completed! 🏆)
- **Status**: Completed
- **Environment**: Kaggle NVIDIA Tesla T4 GPU (15 GB VRAM).
- **Dataset**: `TinyStories` pre-training corpus (4,466,061 train tokens, 194,559 validation tokens).
- **Tokenizer**: Pretrained GPT-2 BPE tokenizer via `tiktoken` ($V = 50,257$) with `<|endoftext|>` ($50,256$) story boundary delimiters.
- **Architecture Scaled**:
  - Parameters: $\sim 17.6\text{M}$
  - `vocab_size = 50257`, `d_model = 256`, `max_seq_len = 256`, `num_heads = 4`, `num_layers = 6`, `dropout = 0.1`
- **Accomplished**:
  - **GPU Acceleration & Mixed Precision**:
    - Dispatched batch streams and model weights directly to `cuda:0`.
    - Integrated `torch.amp.autocast('cuda')` and `torch.amp.GradScaler('cuda')` to exploit Tensor Cores in 16-bit precision without gradient underflow.
    - Optimized memory footprint with `optimizer.zero_grad(set_to_none=True)`.
  - **Training Dynamics & Loss Trajectory**:
    - Mathematical baseline matched theoretical entropy: $\ln(50257) \approx 10.82 \leftrightarrow 10.88$.
    - Rapid convergence from **10.48 down to 1.97 train loss** and **2.21 val loss** (Perplexity dropped from $50,000 \rightarrow 7.38$).
    - Zero severe overfitting; steady validation tracking.
  - **Full Emergent Narrative Intelligence**:
    - Generated syntactically flawless English stories with dialogue, quotation formatting, and character consistency (Lily, Mommy, Daddy).
    - Demonstrated boundary token synthesis: emitting `<|endoftext|>` upon story completion and transitioning cleanly to new story prefixes.
  - **Export**: Saved self-contained `tinystories_transformer_17m.pt` checkpoint to Kaggle working storage.




