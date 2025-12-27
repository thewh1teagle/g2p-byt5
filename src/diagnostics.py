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
