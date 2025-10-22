# 🧠 LLM and Neural Networks: Complete Guide

---

## 📌 **Four Core Steps of Model Training**

Neural network training follows a systematic 4-step cycle:

| Step | Name | What Happens |
|:----:|:-----|:-------------|
| 1️⃣ | **Forward Pass** | Make a prediction using current parameters |
| 2️⃣ | **Calculate Loss** | Measure how far off the prediction is from ground truth |
| 3️⃣ | **Backward Pass** | Compute gradients (how to wiggle parameters) |
| 4️⃣ | **Optimization** | Update parameters in the direction that reduces loss |

```
Input Data
   ↓
[1. Forward Pass] → Prediction
   ↓
[2. Loss Calculation] → Error Metric
   ↓
[3. Backward Pass] → Gradients
   ↓
[4. Optimization] → Update Weights
   ↓
Repeat...
```

---

## 🦙 **LlamaForCausalLM Architecture**

### **Complete Model Structure**

```
LlamaForCausalLM
├── LlamaModel (Transformer backbone)
│   ├── embed_tokens (Vocabulary → Dense vectors)
│   ├── 32 x LlamaDecoderLayers (Sequential processing)
│   ├── rotary_emb (Position encoding)
│   └── norm (Final normalization)
└── lm_head (Output projection to vocabulary)
```

---

## 🔑 **Key Components Breakdown**

### **1. Embedding Layer**
```
Embedding(128256, 4096)
```
- **Input:** Token IDs (0 to 128,255)
- **Output:** Dense vectors of 4,096 dimensions
- **Purpose:** Convert discrete tokens → continuous representations
- **Example:** Token ID #42 → [0.234, -0.891, 0.123, ..., -0.456] (4096 values)

| Property | Value |
|----------|-------|
| Vocab Size | 128,256 tokens |
| Embedding Dim | 4,096 |
| Total Parameters | ~524M |

---

### **2. Transformer Layers (×32)**

**Structure per layer:**

```
Input (4096-dim)
   ↓
┌─ [RMSNorm]
│     ↓
│  [Self-Attention] 
│     ↓ (residual connection)
├─ [RMSNorm]
│     ↓
│  [Feed-Forward MLP]
│     ↓ (residual connection)
↓
Output (4096-dim)
```

Each layer has:
- **Self-Attention mechanism** (what to focus on)
- **Feed-Forward Network** (deep processing)
- **Residual connections** (improved gradients)
- **Normalization layers** (training stability)

---

### **3. Self-Attention Mechanism** 🎯

**Purpose:** Allow each token to "see" and "attend to" all other tokens in sequence

**Components:**

| Component | Input | Output | Purpose |
|-----------|-------|--------|---------|
| **Query (q_proj)** | 4,096 | 4,096 | "What am I looking for?" |
| **Key (k_proj)** | 4,096 | **1,024** | "What does each token have?" |
| **Value (v_proj)** | 4,096 | **1,024** | "What information to extract?" |
| **Output (o_proj)** | 4,096 | 4,096 | "Combine all heads into one" |

**Key Insight:** K and V are **4× compressed** (4096 → 1024)
- ✅ Faster inference
- ✅ Reduced memory footprint
- ✅ Called **Multi-Query Attention (MQA)**

**Attention Flow:**
```
Queries × Keys^T → Scores
         ↓
     Softmax (get attention weights)
         ↓
Attention Weights × Values → Context
         ↓
Output Projection → Final output
```

---

### **4. Feed-Forward Network (MLP)** 🔗

**Purpose:** Add non-linear expressiveness and processing capacity

**Architecture:**

```
Input (4096)
    ↓
┌── [up_proj] ────────┐
│                     │
Input ──┤             ├─→ Element-wise Multiply
│        └── [gate_proj] ┘
│                     
├─→ (14336)
│      ↓
  [down_proj]
      ↓
   Output (4096)
```

| Layer | Input Dim | Output Dim | Ratio | Purpose |
|-------|-----------|-----------|-------|---------|
| up_proj | 4,096 | 14,336 | 3.5× | Expand (learn more) |
| gate_proj | 4,096 | 14,336 | 3.5× | Learn what matters |
| down_proj | 14,336 | 4,096 | 1/3.5 | Compress (output) |

**Gating Mechanism:**
```
gate_proj(x) ⊙ up_proj(x) → Element-wise multiplication
```
- Learns which information is important
- Similar to LSTM gates
- More selective than standard MLP

**Activation Function:**
- **SiLU (Sigmoid Linear Unit)** = `x × sigmoid(x)`
- Smoother than ReLU
- Better gradient flow during training

**Why 14,336?** Optimized for hardware efficiency (divisible by common GPU/TPU parallelization schemes)

---

### **5. Normalization Layers** ⚙️

**RMSNorm (Root Mean Square Normalization)**

```
Normalized = x / RMS(x) × scale
where RMS(x) = sqrt(mean(x²))
```

**Three Uses in Each Layer:**

1. **input_layernorm** - Before attention
   - Stabilizes attention computation
   - Reduces outliers
   
2. **post_attention_layernorm** - After attention
   - Before MLP processing
   - Ensures consistent scales

3. **Final norm** - After all 32 layers
   - Prepares for output head
   - Ensures bounded values

| Property | Value |
|----------|-------|
| Type | RMSNorm (not LayerNorm) |
| Epsilon | 1e-05 (prevents division by zero) |
| Efficiency | ~30% faster than LayerNorm |

---

### **6. Output Head** 📊

```
Linear(4096 → 128256)
```

**Process:**

```
Last Layer Hidden State (4096-dim)
         ↓
   [Linear projection]
         ↓
   Logits (128256 values)
         ↓
    [Softmax]
         ↓
Probabilities (sum to 1)
         ↓
[Sampling/Argmax] → Next Token
```

- Projects hidden state to vocabulary space
- No bias term (reduces parameters)
- Logits converted to probabilities via softmax
- Highest probability token is typically selected

---

## 🔄 **Complete Data Flow**

```
Sequence of Token IDs [101, 2054, 2003, ...]
         ↓
    [Embedding Layer]
         ↓
32-D Dense Embeddings (4096-dim)
         ↓
    FOR EACH of 32 LAYERS:
    ┌─────────────────────────────┐
    │  Layer i                    │
    ├─────────────────────────────┤
    │  ├─ Normalize (RMSNorm)     │
    │  ├─ Self-Attention          │
    │  │  ├─ Q, K, V projections  │
    │  │  ├─ Compute scores       │
    │  │  └─ Aggregate values      │
    │  ├─ Residual connection +   │
    │  │                          │
    │  ├─ Normalize (RMSNorm)     │
    │  ├─ Feed-Forward MLP        │
    │  │  ├─ up_proj (expand)     │
    │  │  ├─ gate_proj (gating)   │
    │  │  ├─ Multiply & activate  │
    │  │  └─ down_proj (compress) │
    │  └─ Residual connection +   │
    └─────────────────────────────┘
         ↓
    [Final RMSNorm]
         ↓
    [Output Head - Linear Layer]
         ↓
  Logits for All 128,256 Tokens
         ↓
    [Softmax]
         ↓
  Probability Distribution
         ↓
    Next Token Prediction ✨
```

---

## 💡 **Design Principles & Optimizations**

### **Why These Choices?**

| Feature | Why It Matters | Benefit |
|---------|----------------|---------|
| **4-bit Quantization** | Reduced memory footprint | Larger batches, faster training |
| **Multi-Query Attention** | K, V compressed 4× | Inference 3-4× faster |
| **3.5× MLP Expansion** | Increased capacity per layer | Better expressiveness |
| **RMSNorm** | Simpler normalization | 30% faster than LayerNorm |
| **Residual Connections** | Gradient highway | Prevents vanishing gradients |
| **Rotary Embeddings** | Position-aware attention | Better length extrapolation |
| **No Bias Terms** | Fewer parameters | ~3% memory reduction |

---

## 📈 **Model Statistics**

| Metric | Value |
|--------|-------|
| **Vocabulary Size** | 128,256 tokens |
| **Hidden Dimension** | 4,096 |
| **Number of Layers** | 32 |
| **Attention Heads** (implied) | 32 (via MQA) |
| **Head Dimension** | 128 |
| **Intermediate MLP Dim** | 14,336 |
| **Approximate Parameters** | ~70B (Llama 3 70B) |
| **Training FLOPs** | ~140 trillion per token |

---

## 🎓 **Key Concepts to Remember**

### ✅ **Core Insights**

1. **Attention is Pattern Matching**
   - Tokens vote on what's important
   - Multi-head = multiple patterns simultaneously

2. **MLP is Non-Linearity**
   - Linear combinations of linear transformations = still linear
   - MLP activation functions add "bent" to the function
   - Model can learn complex, non-linear patterns

3. **Residual Connections Enable Depth**
   - Without them: information gets diluted in deep networks
   - With them: gradients flow cleanly back to early layers
   - Allows 32 layers to work well together

4. **Normalization Stabilizes Training**
   - RMSNorm prevents exploding/vanishing gradients
   - Keeps all hidden states in reasonable ranges
   - Makes optimization landscape smoother

5. **Quantization ≠ Quality Loss**
   - 4-bit quantization: 4× memory savings
   - Minimal accuracy drop with proper calibration
   - Hardware-friendly for inference

---

## 🚀 **From Theory to Practice**

**During Inference (Generating Text):**
1. User provides prompt → Tokenized
2. Pass through all 32 layers sequentially
3. Each layer attends to previous tokens + refines representation
4. Output head predicts next token probability
5. Sample token from distribution
6. Append to sequence, repeat (autoregressive generation)

**During Training:**
1. Forward pass through entire architecture
2. Compute loss between prediction and true next token
3. Backpropagate gradients through all 32 layers
4. Update all weights using optimizer (e.g., AdamW)
5. Repeat until convergence

---

## 📚 **Quick Reference**

**Components | Count | Total Params**
- Embedding Layer | 1 | ~524M
- Attention Layers (×32) | 96 | ~40B
- MLP Layers (×32) | 96 | ~25B
- Normalization (×96) | 96 | ~0.4M
- Output Head | 1 | ~524M
- **TOTAL** | - | **~70B**

---

**Last Updated:** 2025 | Created for LLM Engineering Course