# 🤗 Hugging Face Library: Complete Guide

## 📦 Overview

Hugging Face is the **go-to ecosystem** for working with pre-trained models, datasets, and transformer-based NLP. It simplifies downloading, fine-tuning, and deploying state-of-the-art models.

---

## 🏗️ Core Hugging Face Libraries

### **1. 🔗 Hub**
**Purpose:** Connect to Hugging Face platform and download pre-trained models

```python
from huggingface_hub import hf_hub_download, login

# Login to your account
login(token="your_token_here")

# Download a model
model_path = hf_hub_download(
    repo_id="google-bert/bert-base-uncased",
    filename="pytorch_model.bin"
)
```

**When to use:**
- ✅ Downloading models from Hub
- ✅ Uploading your own models
- ✅ Managing model versions
- ✅ Private model access

---

### **2. 📊 Dataset**
**Purpose:** Load, process, and manage datasets efficiently

```python
from datasets import load_dataset

# Load from Hugging Face Hub
dataset = load_dataset("wikitext", "wikitext-2")

# Or from local files
dataset = load_dataset("csv", data_files="data.csv")

# Preprocessing
dataset = dataset.map(lambda x: {"text": x["text"].lower()})
dataset = dataset.filter(lambda x: len(x["text"]) > 10)
```

**Features:**
- ✅ 10,000+ datasets available
- ✅ Efficient streaming (no need to download entire dataset)
- ✅ Built-in preprocessing
- ✅ Multi-language support

---

### **3. 🔄 Transformers**
**Purpose:** Core library for working with transformer models

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Tokenize
inputs = tokenizer("Hello world!", return_tensors="pt")

# Forward pass
outputs = model(**inputs)
predictions = torch.softmax(outputs.logits, dim=-1)
```

**What it includes:**
- ✅ 1000+ pre-trained models
- ✅ Support for NLP, Vision, Speech, MultiModal
- ✅ Tokenizers
- ✅ Training utilities
- ✅ Fine-tuning helpers

---

### **4. ⚡ PEFT (Parameter Efficient Fine-Tuning)**
**Purpose:** Fine-tune large models with minimal parameters (LoRA, QLoRA, etc.)

```python
from peft import get_peft_model, LoraConfig, TaskType

# Traditional fine-tuning: Update ALL 7B parameters
# PEFT approach: Update only 0.1% parameters (much faster & cheaper!)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,  # Rank of LoRA
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]  # Only fine-tune these layers
)

model = get_peft_model(model, peft_config)
# Now train this model - 100x faster with same quality!
```

**Methods:**
- ✅ **LoRA** - Low Rank Adaptation
- ✅ **QLoRA** - Quantized LoRA (8-bit)
- ✅ **Prefix Tuning** - Optimize prefix
- ✅ **Prompt Tuning** - Soft prompts

**When to use:**
- ✅ Fine-tuning large models (LLMs, GPT)
- ✅ Limited GPU memory
- ✅ Quick adaptation to new tasks
- ✅ Cost-effective training

---

### **5. 🎯 TRL (Transformer Reinforcement Learning)**
**Purpose:** Train models with reinforcement learning (RLHF, DPO, etc.)

```python
from trl import SFTTrainer, DPOTrainer

# Supervised Fine-Tuning
sft_trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)
sft_trainer.train()

# Direct Preference Optimization
dpo_trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
```

**Available Methods:**
- ✅ **SFT** - Supervised Fine-Tuning
- ✅ **RLHF** - Reinforcement Learning from Human Feedback
- ✅ **DPO** - Direct Preference Optimization
- ✅ **PPO** - Proximal Policy Optimization

**Use cases:**
- ✅ Aligning models with human preferences
- ✅ Instruction following
- ✅ Reducing harmful outputs

---

### **6. 🚀 Accelerate**
**Purpose:** Distributed training made easy (multi-GPU, TPU, mixed precision)

```python
from accelerate import Accelerator

accelerator = Accelerator(
    mixed_precision="fp16",  # Use 16-bit precision for speed
    device_placement=True
)

# Automatically handles distribution
model, optimizer, train_loader = accelerator.prepare(
    model, optimizer, train_loader
)

for batch in train_loader:
    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
```

**Features:**
- ✅ **Multi-GPU training** - Use multiple GPUs automatically
- ✅ **Mixed Precision** - FP16 for speed (2x faster)
- ✅ **Gradient Accumulation** - Larger batch sizes on small GPUs
- ✅ **Zero Redundancy Optimizer** - Massive model training
- ✅ Works with DeepSpeed

---

## 🎯 Two Levels of Hugging Face APIs

### **Architecture Overview**
```
┌─────────────────────────────────────────┐
│     Your Application                    │
├─────────────────────────────────────────┤
│     High-Level API: PIPELINES 🚀       │
│  (Simple, fast, ready-to-use)          │
├─────────────────────────────────────────┤
│     Low-Level API: TRANSFORMERS ⚙️      │
│  (Raw power, full control)             │
├─────────────────────────────────────────┤
│ Tokenizers | Models | Configs           │
└─────────────────────────────────────────┘
```

---

### **Level 1️⃣: High-Level API - Pipelines**

**Best for:** Quick prototyping, beginners, production inference

```python
from transformers import pipeline

# 1 line to get started!
classifier = pipeline("sentiment-analysis")
result = classifier("I love this!")
# Output: [{"label": "POSITIVE", "score": 0.9995}]
```

**Advantages:**
- ✅ Extremely simple (1-2 lines)
- ✅ Auto-downloads model
- ✅ Handles tokenization automatically
- ✅ Best performance settings pre-configured
- ✅ Production-ready

**Disadvantages:**
- ❌ Less control
- ❌ Can't modify internal behavior
- ❌ Slower for batch processing

---

### **Level 2️⃣: Low-Level API - Tokenizers & Models**

**Best for:** Custom training, fine-tuning, advanced use cases

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Step 1: Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Step 2: Load model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Step 3: Tokenize
inputs = tokenizer("Hello world!", return_tensors="pt")
# Output: {"input_ids": [[101, 7592, ...]], "attention_mask": [[1, 1, ...]]}

# Step 4: Forward pass
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Step 5: Post-process
probabilities = torch.softmax(logits, dim=-1)
prediction = torch.argmax(probabilities, dim=-1)
```

**Advantages:**
- ✅ Full control over every step
- ✅ Custom modifications
- ✅ Fine-grained optimization
- ✅ Can access intermediate layers
- ✅ Batch processing friendly

**Disadvantages:**
- ❌ More code required
- ❌ Requires understanding transformers
- ❌ More error-prone

---

## 📋 Common Tasks Available in Pipelines

| Task | Code | Input | Output | Use Case |
|------|------|-------|--------|----------|
| **Sentiment Analysis** | `pipeline("sentiment-analysis")` | Text | Label + score | Social media |
| **NER** | `pipeline("ner")` | Text | Entities | Extract names |
| **QA** | `pipeline("question-answering")` | Q + context | Answer | Reading comp |
| **Summarization** | `pipeline("summarization")` | Long text | Short text | Condensing |
| **Translation** | `pipeline("translation_en_to_de")` | Text | Translated | Languages |
| **Text Generation** | `pipeline("text-generation")` | Prompt | Generated text | Autocomplete |
| **Zero-shot** | `pipeline("zero-shot-classification")` | Text + labels | Classified | Custom classes |
| **Fill Mask** | `pipeline("fill-mask")` | Text [MASK] | Predictions | MLM |

---

## 🔧 Pipeline Configuration

### **Basic Usage**
```python
# Simplest
pipeline("sentiment-analysis")

# With device
pipeline("sentiment-analysis", device="cuda")  # GPU
pipeline("sentiment-analysis", device="cpu")   # CPU

# With specific model
pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# With all options
pipeline(
    task="sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device="cuda",
    device_map="auto",  # Multi-GPU
    torch_dtype="float16",  # Memory efficient
    batch_size=32  # Process multiple at once
)
```

### **Reusing Pipeline (Recommended)**
```python
# ❌ DON'T: Create new pipeline each time (slow)
for text in texts:
    result = pipeline("sentiment-analysis")(text)

# ✅ DO: Create once, reuse
classifier = pipeline("sentiment-analysis", device="cuda")
for text in texts:
    result = classifier(text)  # 10x faster!
```

---

## 💡 When to Use What

### **Use Pipelines When:**
```python
✅ Learning transformers
✅ Quick prototyping
✅ Simple inference
✅ Production serving (simple use cases)
✅ Want minimum code

my_pipeline = pipeline("sentiment-analysis")
result = my_pipeline("I love this!")
```

### **Use Tokenizers + Models When:**
```python
✅ Fine-tuning models
✅ Complex workflows
✅ Need custom preprocessing
✅ Batch processing large datasets
✅ Building production systems

from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
```

---

## 🚀 Complete Example: Sentiment Analysis

### **Method 1: Using Pipelines (Simple)**
```python
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device="cuda"
)

texts = [
    "I love this movie!",
    "This is terrible",
    "It's okay, nothing special"
]

for text in texts:
    result = classifier(text)
    print(f"{text} → {result}")
```

### **Method 2: Using Low-Level API (Control)**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
).to("cuda")

def classify(text):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    label = torch.argmax(probs).item()
    score = probs[0][label].item()
    return {"label": ["NEGATIVE", "POSITIVE"][label], "score": score}

texts = ["I love this!", "This is terrible", "It's okay"]
for text in texts:
    print(f"{text} → {classify(text)}")
```

---

## 📊 Library Selection Guide

```
What do you want to do?
│
├─ Download pre-trained models
│  └─ Use: Hub
│
├─ Load datasets
│  └─ Use: datasets
│
├─ Simple inference
│  └─ Use: Transformers (Pipelines)
│
├─ Fine-tune models
│  ├─ Simple fine-tuning
│  │  └─ Use: Transformers + Accelerate
│  ├─ Large models (7B+)
│  │  └─ Use: PEFT + Accelerate
│  └─ With human feedback
│     └─ Use: TRL + PEFT
│
└─ Distribute training
   └─ Use: Accelerate
```

---

## 🎯 Key Takeaways

| Component | Purpose | When to Use |
|-----------|---------|-----------|
| **Hub** | Download/upload models | Always (to get models) |
| **datasets** | Load/process data | When you need data |
| **Transformers** | Core library | Every project |
| **PEFT** | Efficient fine-tuning | Large models (>1B params) |
| **TRL** | RL training | When aligning models |
| **Accelerate** | Distributed training | Multi-GPU setups |

---

## 🔗 Quick Links

- **Hub:** [huggingface.co/models](https://huggingface.co/models)
- **Docs:** [huggingface.co/docs](https://huggingface.co/docs)
- **Datasets:** [huggingface.co/datasets](https://huggingface.co/datasets)

---

## 💻 Setup Code

```bash
# Install everything
pip install transformers datasets accelerate peft trl huggingface-hub

# For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Login to Hub
huggingface-cli login
```

---

## 🌟 Pro Tips

1. **Always specify device** to avoid auto-detection issues
2. **Reuse pipelines** rather than creating new ones
3. **Use PEFT for large models** to save memory
4. **Cache models locally** for faster loading
5. **Use Accelerate for multi-GPU** training
6. **Profile your code** to identify bottlenecks