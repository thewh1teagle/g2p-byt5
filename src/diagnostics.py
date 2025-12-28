try:
    import wandb
except ImportError:
    wandb = None


def print_trainable_params(model):
    """Print the number of trainable parameters."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_gb = trainable_params * 4 / 1024**3  # 4 bytes per float32
    print(f"💪 Trainable params: {trainable_params:,} / {all_params:,} ({trainable_gb:.2f} GB)")


def print_samples(dataset, tokenizer, split_name="Train", num_samples=2):
    """Print sample examples from dataset."""
    print(f"\n📋 {split_name} Samples:")
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        input_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        # Labels have -100 for padding, replace with pad_token_id for decoding
        labels = [l if l != -100 else tokenizer.pad_token_id for l in sample['labels']]
        target_text = tokenizer.decode(labels, skip_special_tokens=True)
        print(f"  [{i}] Input:  '{input_text}' ({len(sample['input_ids'])} tokens)")
        print(f"  [{i}] Target: '{target_text}' ({len(sample['labels'])} tokens)")


def print_dataset_info(train_dataset, val_dataset, tokenizer, num_samples=2):
    """Print dataset sizes and sample examples."""
    print(f"\n✂️  Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print_samples(train_dataset, tokenizer, "Train", num_samples=num_samples)
    print_samples(val_dataset, tokenizer, "Eval", num_samples=num_samples)


def check_vocab_range(predictions, vocab_size):
    """Check if predictions are within valid vocab range."""
    out_of_vocab = (predictions < 0) | (predictions >= vocab_size)
    if out_of_vocab.any():
        print(f"⚠️  WARNING: {out_of_vocab.sum()} tokens out of vocab range!")
        print(f"   Min: {predictions.min()}, Max: {predictions.max()}, Vocab size: {vocab_size}")


def print_eval_predictions(decoded_preds, decoded_labels, num_samples=5):
    """Print evaluation predictions for debugging."""
    print("\n" + "="*80)
    print("📝 EVALUATION SAMPLES")
    print("="*80)

    for i in range(min(num_samples, len(decoded_preds))):
        pred = decoded_preds[i]
        label = decoded_labels[i]
        match = pred == label
        symbol = "✓" if match else "✗"

        print(f"\n{symbol} Example {i+1}:")
        print(f"  Target: {repr(label)}")
        print(f"  Pred:   {repr(pred)}")

        if not match:
            print(f"  Length: Target={len(label)}, Pred={len(pred)}")
            # Show first difference
            for j, (t, p) in enumerate(zip(label, pred)):
                if t != p:
                    print(f"  First diff at pos {j}: '{t}' vs '{p}'")
                    break

    # Calculate and print summary
    exact_matches = sum(1 for p, l in zip(decoded_preds, decoded_labels) if p == l)
    total = len(decoded_preds)
    accuracy = exact_matches / total * 100

    print("\n" + "="*80)
    print(f"📊 Exact Match: {accuracy:.2f}% ({exact_matches}/{total})")
    print("="*80 + "\n")

    # Log examples to wandb
    if wandb and wandb.run is not None:
        examples = []
        for i in range(min(num_samples, len(decoded_preds))):
            examples.append([
                decoded_labels[i],  # Target
                decoded_preds[i],   # Prediction
                "✓" if decoded_preds[i] == decoded_labels[i] else "✗"  # Match
            ])
        wandb.log({
            "eval_examples": wandb.Table(
                columns=["Target", "Prediction", "Match"],
                data=examples
            )
        })
