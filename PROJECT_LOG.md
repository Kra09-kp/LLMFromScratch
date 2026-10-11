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

### Day 5: Full Generative Transformer & Text Generation (Next Up)
- Stacking $N$ `DecoderBlock` layers using `nn.ModuleList`.
- Final LayerNorm before the language model head.
- The LM Head (`Linear(d_model, vocab_size)`) & Weight Tying intuition.
- Cross-Entropy Loss computation: aligning logits with shifted targets.
- Auto-regressive text generation: prompt encoding, next-token sampling, and decoding.
