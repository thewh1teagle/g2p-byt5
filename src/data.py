from datasets import Dataset
import pandas as pd
from pathlib import Path
from transformers import DataCollatorForSeq2Seq
from config import MAX_LENGTH
from tqdm import tqdm


def load_tsv_data(file_path: str):
    # Count total lines for progress bar
    with Path(file_path).open("r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    rows = []
    with Path(file_path).open("r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Loading TSV data"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            input_text, output_text = parts
            rows.append({"input": input_text, "output": output_text})
    return pd.DataFrame(rows)


def prepare_dataset(df, tokenizer, max_length=MAX_LENGTH, cache_file=None):
    def tokenize_function(examples):
        # Use dynamic padding - DataCollatorForSeq2Seq will handle padding
        model_inputs = tokenizer(
            examples["input"],
            max_length=max_length,
            truncation=True,
        )

        # Tokenize labels directly without deprecated context manager
        labels = tokenizer(
            text_target=examples["output"],
            max_length=max_length,
            truncation=True,
        )

        # DataCollatorForSeq2Seq will handle padding and -100 replacement
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset = Dataset.from_pandas(df)
    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
        load_from_cache_file=bool(cache_file),
        cache_file_name=cache_file,
    )


def split_dataset(dataset, train_ratio=0.8, seed=42):
    # Shuffle the dataset before splitting
    dataset = dataset.shuffle(seed=seed)

    train_size = int(train_ratio * len(dataset))

    # Ensure minimum validation set size
    min_val_size = 10
    if len(dataset) - train_size < min_val_size:
        raise ValueError(
            f"Dataset too small for split. Need at least {int(min_val_size / (1 - train_ratio))} examples, "
            f"got {len(dataset)}"
        )

    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, len(dataset)))
    return train_dataset, val_dataset


def create_data_collator(tokenizer, model):
    """Create a data collator for seq2seq training."""
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
