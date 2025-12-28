# Simple Pre-tokenization for G2P

## Core Insight

**The model should see the FULL sentence** and learn to:
1. Convert Hebrew characters → phonemes
2. Preserve everything else exactly as-is (punctuation, spaces, emojis, English, numbers)

This is **simpler** than the junior's approach because:
- No `<UNK>` tokens needed
- No position tracking needed
- No reconstruction complexity
- Model does all the work

## How It Works

### Training

The model is trained on pairs like:

```
Input:  "שלום עולם!"
Output: "ʃalom olam!"
```

```
Input:  "אמר: 'שלום'"
Output: "amar: 'ʃalom'"
```

```
Input:  "Hello שלום 😊"
Output: "Hello ʃalom 😊"
```

The model learns a **character-level transformation**:
- Hebrew letters → phoneme characters
- Everything else → identity (stays the same)

### Inference

```python
# User input
text = "שלום😊עולם!"

# Normalize quotes only (optional)
normalized = normalize_quotes(text)  # "שלום😊עולם!"

# Model sees full sentence
model_output = model.generate(normalized)  # "ʃalom😊olam!"

# That's it! No reconstruction needed.
```

## Why This is Better

### Junior's Approach (Complex)
```
Input: "שלום😊עולם!"
↓
Preprocess: "שלום <UNK> עולם !" + tracking {0: "😊"}
↓
Model: "ʃalom <UNK> olam !"
↓
Restore UNK: "ʃalom😊olam !"
↓
Remove spacing: "ʃalom😊olam!"
```

**Problems:**
- Need to track UNK positions
- Need to restore UNK content
- Need to remove punctuation spacing
- Model doesn't see emojis/English in training
- Complex edge cases

### Our Approach (Simple)
```
Input: "שלום😊עולם!"
↓
Model: "ʃalom😊olam!"
```

**Benefits:**
- ✅ Model sees actual content (learns to preserve it)
- ✅ No tracking needed
- ✅ No reconstruction needed
- ✅ Handles any character naturally
- ✅ Model has full context

## What the Model Learns

The model learns a **context-aware character mapping**:

| Input Char | Output | Context |
|------------|--------|---------|
| `ש` | `ʃ` | Hebrew letter |
| `ל` | `l` | Hebrew letter |
| `ו` | `o` | Hebrew letter (context-dependent) |
| `ם` | `m` | Hebrew letter |
| `😊` | `😊` | Non-Hebrew (identity) |
| `!` | `!` | Punctuation (identity) |
| ` ` | ` ` | Space (identity) |
| `H` | `H` | Latin letter (identity) |
| `1` | `1` | Digit (identity) |

**Key:** Hebrew letters are in the range `\u05d0-\u05ea` (א-ת). Everything else is identity.

## Training Data Format

### From Word Pairs to Sentences

If you have word-level data:
```
שלום → ʃalom
עולם → olam
```

Create sentence-level training data:

**Option 1: Keep it simple (individual words)**
```
Input:  "שלום"
Output: "ʃalom"
```

**Option 2: Add context (with punctuation)**
```
Input:  "שלום."
Output: "ʃalom."

Input:  "שלום!"
Output: "ʃalom!"
```

**Option 3: Combine into sentences**
```
Input:  "שלום עולם"
Output: "ʃalom olam"

Input:  "שלום עולם!"
Output: "ʃalom olam!"
```

**Option 4: Add non-Hebrew content**
```
Input:  "Hello שלום"
Output: "Hello ʃalom"

Input:  "שלום 😊"
Output: "ʃalom 😊"
```

The model will learn to:
- Transform Hebrew → phonemes
- Preserve everything else

## Implementation

### Core Functions

```python
def normalize_quotes(text: str) -> str:
    """Optional: Normalize Hebrew/curly quotes to ASCII."""
    # '׳' → "'"
    # '״' → '"'
    # etc.

def g2p_pipeline(text: str) -> str:
    """Full pipeline: normalize → model → done."""
    normalized = normalize_quotes(text)
    return model.generate(normalized)
```

That's it! No complex preprocessing or reconstruction.

### Model Training

```python
# Prepare training data
train_pairs = []
for hebrew_word, phonemes in word_pairs:
    # Option 1: Just words
    train_pairs.append((hebrew_word, phonemes))

    # Option 2: Add variations
    train_pairs.append((f"{hebrew_word}.", f"{phonemes}."))
    train_pairs.append((f"{hebrew_word}!", f"{phonemes}!"))

    # Option 3: Combine into sentences
    # (pair with other words randomly)

# Train ByT5 model
model.train(train_pairs)
```

### Model Inference

```python
def phonemize(text: str) -> str:
    """Convert Hebrew text to phonemes."""
    text = normalize_quotes(text)
    return model.generate(text)
```

## Edge Cases

All handled naturally by the model:

| Input | Output | Notes |
|-------|--------|-------|
| `"שלום😊"` | `"ʃalom😊"` | Emoji preserved |
| `"שלום 123"` | `"ʃalom 123"` | Numbers preserved |
| `"Hello שלום"` | `"Hello ʃalom"` | English preserved |
| `"שלום!"` | `"ʃalom!"` | Punctuation preserved |
| `"  שלום  "` | `"  ʃalom  "` | Spaces preserved |
| `"א־ב"` | `"a-b"` | Hyphens preserved |

## Comparison

### Complexity

| Aspect | Junior's Approach | Our Approach |
|--------|------------------|--------------|
| Preprocessing | Complex (UNK tracking) | Simple (quote normalization) |
| Model Input | Modified (`<UNK>`) | Original (full content) |
| Model Training | Learns to preserve `<UNK>` | Learns to preserve actual content |
| Reconstruction | Complex (restore, remove spacing) | None needed |
| Edge Cases | Many (UNK positioning, spacing) | None (model handles all) |

### Code Size

| Component | Junior's | Ours |
|-----------|----------|------|
| `preprocess` | ~100 lines | ~15 lines |
| `reconstruct` | ~50 lines | ~0 lines |
| `pipeline` | ~40 lines | ~10 lines |
| **Total** | **~190 lines** | **~25 lines** |

### Training Data

| Approach | Data Format | Model Learns |
|----------|-------------|--------------|
| Junior's | `"שלום <UNK> עולם" → "ʃalom <UNK> olam"` | Hebrew + `<UNK>` placeholder |
| Ours | `"שלום😊עולם" → "ʃalom😊olam"` | Hebrew + actual content |

Our approach is **more robust** because the model sees real examples during training.

## Summary

1. **Model sees full sentences** → Better context for phonemization
2. **No special tokens** → Model learns real content preservation
3. **No reconstruction** → Output is final result
4. **Simpler code** → Fewer bugs, easier maintenance
5. **Better training** → Model sees actual non-Hebrew content

The key insight: **Let the model do the work**. ByT5 is designed for character-level transformations. Just train it on the full content and it will learn to preserve what it shouldn't change.

## If Your Dataset is Clean (Hebrew + Punctuation Only)

**Good news:** You need to do **even less work**!

### Scenario: Your data only has Hebrew words and punctuation

If your existing dataset looks like this:
```
שלום → ʃalom
עולם → olam
שלום! → ʃalom!
מה? → ma?
```

**You can train directly with ZERO preprocessing!**

### Why this works

ByT5 will learn the character mapping:
- Hebrew letters → phonemes
- Punctuation → identity (stays the same)
- Spaces → identity (stays the same)

### What about emojis/English that users might input later?

The model will **automatically generalize** because:

1. **During training:** Model learns "only transform Hebrew letters"
2. **At inference:** When it sees emoji/English, it doesn't match any Hebrew letter, so it outputs it as-is

**Example:**
```python
# Training data (clean, only Hebrew + punctuation)
train_pairs = [
    ("שלום", "ʃalom"),
    ("עולם", "olam"),
    ("שלום!", "ʃalom!"),
]

# Train model
model.train(train_pairs)

# At inference - model automatically handles unseen characters!
model.generate("שלום")       # → "ʃalom"  ✓ (trained)
model.generate("שלום😊")      # → "ʃalom😊" ✓ (emoji passes through)
model.generate("Hello שלום")  # → "Hello ʃalom" ✓ (English passes through)
model.generate("שלום 123")    # → "ʃalom 123" ✓ (numbers pass through)
```

### Why the model generalizes

ByT5 is a **character-level** model. It learns patterns like:

| Input char | Output char | Pattern learned |
|------------|-------------|-----------------|
| `ש` | `ʃ` | Hebrew letter → transform |
| `ל` | `l` | Hebrew letter → transform |
| `!` | `!` | Punctuation → identity |
| ` ` | ` ` | Space → identity |
| `😊` | `😊` | Unknown → identity (no training example, so copy) |
| `H` | `H` | Unknown → identity (no training example, so copy) |

The model learns: "Transform Hebrew letters, copy everything else."

### Do you need to add non-Hebrew examples?

**No, but you can if you want to be extra safe.**

#### Option 1: Train on clean data only (recommended)
```python
# Just use your existing clean dataset
train_pairs = load_clean_hebrew_data()  # Only Hebrew + punctuation
model.train(train_pairs)
```

**Pros:**
- ✅ No extra work
- ✅ Model will generalize naturally
- ✅ Simpler training data

**Cons:**
- ⚠️ Not 100% guaranteed to preserve unseen characters (but very likely)

#### Option 2: Add a few non-Hebrew examples (extra safety)
```python
# Start with your clean data
train_pairs = load_clean_hebrew_data()

# Add a handful of examples with non-Hebrew content
train_pairs.extend([
    ("Hello שלום", "Hello ʃalom"),
    ("שלום😊", "ʃalom😊"),
    ("שלום 123", "ʃalom 123"),
    ("test שלום test", "test ʃalom test"),
])

model.train(train_pairs)
```

**Pros:**
- ✅ Explicitly teaches model to preserve non-Hebrew
- ✅ More robust
- ✅ Only need ~10-20 examples for this

**Cons:**
- ⚠️ Requires creating synthetic examples

### Recommendation

**If your dataset is clean (Hebrew + punctuation only):**

1. **First:** Train on your clean data as-is (no preprocessing needed!)
2. **Test:** Try some examples with emojis/English
3. **If needed:** Add 10-20 synthetic examples with non-Hebrew content

Most likely, step 1 will be sufficient. ByT5 is designed to generalize well.
