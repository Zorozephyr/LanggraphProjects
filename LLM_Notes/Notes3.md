# 🔤 Tokenization & Model Variants

## 📚 What is a Tokenizer?

Converts text ↔ token IDs. Models only understand numbers, not text.

```
Text Input → [Tokenizer] → Token IDs → LLM Model
```

**Why it matters:**
- Models need token IDs (not text)
- Each model has its own tokenizer
- Tokenizer affects context window size
- Special tokens control model behavior

---

## 🧠 Vocabulary & Special Tokens

**Vocabulary:** Mapping of tokens to IDs
```
Token 101 → "[CLS]"    Token 1045 → "hello"
Token 102 → "[SEP]"    Token 2054 → "world"
```

**Special Tokens by Model:**
```
BERT:   [CLS], [SEP], [PAD], [UNK], [MASK]
GPT:    <bos>, <eos>, <pad>, <unk>
Llama:  <bos>, <eos>, <pad>, <unk>
```

---

## 🔤 Tokenization Methods

| Method | Example | Pros | Cons |
|--------|---------|------|------|
| **Character** | "hello" → ['h','e','l','l','o'] | Small vocab | Very long sequences |
| **Word** | "hello world" → ['hello','world'] | Intuitive | Huge vocab, can't handle new words |
| **Subword (BPE)** ⭐ | "handcrafted" → ['hand','craft','ed'] | Balanced | Need understanding |

---

## 🔄 BPE (Byte Pair Encoding)

Iteratively merges most frequent character pairs into new tokens.

**Example:** `"low lower newest"` → `['low', 'low', 'er', 'new', 'est']`

1. Start with characters
2. Count pair frequencies
3. Merge most frequent pair
4. Repeat until vocab size reached

**Key benefit:** Handles unknown words by breaking into subwords!

---

## 🔧 Tokenizer in Python

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Encode: text → token IDs
encoded = tokenizer.encode("Hello world!")
# Output: [101, 7592, 2088, 999, 102]

# Decode: token IDs → text
decoded = tokenizer.decode(encoded)

# Full processing
tokens = tokenizer(
    "Hello world!",
    padding="max_length",
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
# Output: {'input_ids': [...], 'attention_mask': [...]}
```

---

## 🎯 Model Variants

| Variant | Training | Best For | Example |
|---------|----------|----------|---------|
| **Base** | Raw text | Research, fine-tuning | "The future of AI is..." (continues) |
| **Chat** ⭐ | Conversations | General use, Q&A | Follows instructions, maintains context |
| **Code** | GitHub, Stack Overflow | Programming | Generates syntactically correct code |
| **Math** | Math problems | Calculations | Reasoning chains |

---

## 💬 Chat Templates: apply_chat_template()

Different models need different formats:
```
GPT:    [{"role": "user", "content": "..."}]
Claude: Human: ... \n\nAssistant: ...
Llama:  <s>[INST] ... [/INST] ... </s>
```

**Solution:** Use `apply_chat_template()`

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat")

messages = [
    {"role": "user", "content": "What is 2+2?"}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
# Automatically converts to model-specific format!
```

**Why it matters:**
- Without it: Model confused, random output ❌
- With it: Perfect formatted response ✅

---

## 🎨 Chat Formats by Model

```
Llama 2:   <s>[INST] <<SYS>>...</SYS>> message [/INST]
Mistral:   [INST] message [/INST]
Claude:    Human: message\n\nAssistant:
GPT:       {"role": "user", "content": "message"}
```

---

## 🎯 Key Concepts

| Concept | Purpose |
|---------|---------|
| **Tokenizer** | Text ↔ Token IDs |
| **BPE** | Breaks words into subwords |
| **Special Tokens** | Control model behavior |
| **Base Model** | Raw model, good for research |
| **Chat Model** | Best for conversations |
| **Chat Template** | Ensures correct format |

---

## 💡 Pro Tips

1. **Use model-specific tokenizer** - Don't mix tokenizers!
2. **Always use chat templates** - Wastes tokens & error-prone otherwise
3. **Check vocabulary size:** `tokenizer.vocab_size` (usually 25K-130K)
4. **Different models = different token IDs** - "hello" might be ID 101 or ID 7592
5. **Context window = max tokens:** 1 token ≈ 0.75 words

---

## 🔗 Common Operations

```python
tokenizer.vocab_size          # Total tokens
tokenizer.model_max_length    # Max sequence length
tokenizer.tokenize(text)      # Text → token strings
tokenizer.encode(text)        # Text → token IDs
tokenizer.decode(ids)         # Token IDs → text
```