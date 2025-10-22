# ⚙️ Quantization: Making Models Smaller & Faster

## 🎯 What is Quantization?

Reduces model size and speeds up inference by converting high-precision numbers (32-bit floats) into lower-precision formats (8-bit integers).

```
FP32 (4 bytes): 3.141592653589793
INT8 (1 byte):  3
Memory saved:   4x | Speed gain: 2-4x | Accuracy: 99%+
```

---

## 📊 Precision Formats

| Format | Bytes | Best For | Accuracy | Size |
|--------|-------|----------|----------|------|
| **FP32** | 4 | Training | 100% | 100% |
| **FP16** | 2 | Training/Inference | 99.5% | 50% |
| **INT8** | 1 | Inference | 98-99% | 25% |
| **NF4** | 0.5 | Mobile/Edge | 96-99% | 12.5% |

---

## 🔄 How Quantization Works

**Example:** Convert `[0.34, 3.75, 5.64, 1.12, 2.7, -0.9, -4.7, 0.68, 1.43]` to INT8

1. Find min (-4.7) and max (5.64) → range = 10.34
2. Map to INT8 (0-255): `int_value = (float_value - min) / range × 255`
3. Result: `[123, 209, 217, 76, 119, 93, 0, 81, 99]`
4. Store scale (10.34/255) and offset (-4.7) for dequantizing

---

## 🎯 Quantization Methods

### **1. Post-Training Quantization (PTQ)** - Simplest
```python
from torch.quantization import quantize_dynamic

quantized_model = quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```
✅ No retraining | ❌ Slightly lower accuracy

### **2. Quantization-Aware Training (QAT)** - Best Quality
Train with quantization simulation for highest accuracy after quantization.
✅ Best accuracy | ❌ Requires retraining

### **3. QLoRA** - Best for Large Models ⭐
```python
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b",
    quantization_config=bnb_config
)
# 70B model: 280GB (FP32) → 48GB (QLoRA) ✅
```

---

## 🔧 BitsAndBytesConfig Deep Dive

### **Configuration Parameters**

```python
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Load in 4-bit (0.5 bytes/weight)
    bnb_4bit_use_double_quant=True,       # Quantize scale factors too (extra 15% saving)
    bnb_4bit_compute_dtype=torch.bfloat16,# Use bfloat16 for computations (avoid overflow)
    bnb_4bit_quant_type="nf4"             # NF4: information-theoretic bucketing
)
```

### **Inner Working with Math**

**Memory Savings Calculation:**
```
LLaMA-7B without quantization:
Parameters: 7 billion
FP32: 7B × 4 bytes = 28 GB

With 4-bit (load_in_4bit=True):
7B × 0.5 bytes = 3.5 GB (7.8x reduction!)

With double_quant (bnb_4bit_use_double_quant=True):
Scale factors: (7B / 64) × 1 byte = 109 MB saved
Total: 3.5 + overhead ≈ 4.5 GB
```

**NF4 vs INT4 (bnb_4bit_quant_type):**
```
INT4: 16 uniform buckets [-8, -7, ..., 6, 7]
      Same spacing everywhere (wastes buckets)

NF4:  16 adaptive buckets based on data distribution
      More precision where weights cluster
      Accuracy gain: ~2-3% vs INT4
```

**compute_dtype Importance:**
```
Why bfloat16 > float16?
- Float16 can overflow with extreme values
- bfloat16 maintains range (±3.4×10^38) while losing precision
- Perfect match for 4-bit weights (similar accuracy loss)

Process:
4-bit weights → dequantize to bfloat16 → compute → output
```

### **Configuration Strategies**

| Use Case | Config | double_quant | compute_dtype |
|----------|--------|--------------|---------------|
| **Inference (production)** | 4-bit NF4 | True | bfloat16 |
| **Fine-tuning with LoRA** | 4-bit NF4 | False | bfloat16 |
| **High accuracy needed** | 8-bit | True | bfloat16 |
| **Extreme edge devices** | 4-bit INT4 | True | float16 |

### **Complete Example: Fine-Tuning**

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 1. Quantization (skip double_quant for accuracy)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,  # Keep accuracy for training
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

# 2. Load model (7GB instead of 28GB)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=quant_config,
    device_map="auto"
)

# 3. Add LoRA (0.1% trainable parameters)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Total GPU memory: ~4 GB (fits 8GB GPU!) ✅
```

---

## 🏷️ NF4 (Natively Quantized Float-4)

Better than INT4 by adapting quantization levels to data distribution.

```python
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
# 8x smaller than FP32!
```

---

## 💡 When to Use Each

| Use Case | Choose | double_quant |
|----------|--------|--------------|
| Maximum accuracy | FP32 | - |
| Training faster | FP16 | - |
| Inference, balanced | INT8 | True |
| Mobile/Edge devices | INT4/NF4 | True |
| Fine-tune large models | 4-bit NF4 + LoRA | False |

---

## 🚀 Real-World Example: LLaMA-7B

```
Without Quantization:   28 GB needed ❌ (won't fit 8GB GPU)
INT8 Quantization:      7 GB needed ✅ (fits, 99% quality)
4-bit NF4:              3.5 GB needed ✅ (fits, 96-98% quality)
4-bit NF4+LoRA:         4-5 GB needed ✅ (fine-tune on 8GB GPU!)
```

---

## 🔧 Hugging Face Integration

**Simple Post-Training:**
```python
from torch.quantization import quantize_dynamic

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
quantized = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

**BitsAndBytes 4-bit:**
```python
from transformers import BitsAndBytesConfig

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=config
)
```

---

## 📈 Benchmarks

| Method | Size | Speed | Accuracy | GPU RAM | Use When |
|--------|------|-------|----------|---------|----------|
| FP32 | 100% | 1x | 100% | 28 GB | Research |
| FP16 | 50% | 1.5x | 99.5% | 14 GB | Training |
| INT8 | 25% | 3x | 98-99% | 7 GB | Balanced |
| NF4 | 12.5% | 3.5x | 96-99% | 3.5 GB | Production |
| NF4+LoRA | 12.5% | 3.2x | 97-99% | 4.5 GB | Fine-tuning |

---

## 🎯 Decision Tree

```
What are you doing?
├─ Training from scratch?
│  └─ Use FP32 or mixed precision (need accuracy)
├─ Fine-tuning?
│  ├─ Small model (< 3B)?
│  │  └─ Use FP16 or INT8
│  └─ Large model (7B+)?
│     └─ Use 4-bit NF4 + LoRA
└─ Inference only?
   ├─ Mobile/edge device?
   │  └─ Use 4-bit NF4
   └─ Server/cloud?
      └─ Use INT8 (better accuracy)
```

---

## 🌟 Key Takeaways

1. **PTQ simplest** - No retraining, works immediately
2. **QLoRA best** - Only way to fine-tune 7B+ on consumer GPUs
3. **NF4 > INT4** - Better accuracy with same size
4. **bfloat16 > float16** - Avoids overflow in 4-bit
5. **double_quant for inference** - Skip for training (maintain accuracy)
6. **Verify accuracy** - Always test quantized vs original on validation set

---

## ⚡ Pro Tips

✅ Use 4-bit NF4 for inference (smallest, fastest)
✅ Skip double_quant when fine-tuning (accuracy matters)
✅ Always use bfloat16 as compute_dtype (avoid overflow)
✅ Test accuracy drop before deploying
❌ Don't use float16 with 4-bit (can overflow)
❌ Don't mix quantization levels (use same for all layers)

