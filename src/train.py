"""
uv run src/train.py
"""
from transformers import T5ForConditionalGeneration, ByT5Tokenizer, Trainer, TrainingArguments
from config import get_config, MAX_LENGTH, set_random_seeds
from data import load_tsv_data, prepare_dataset, split_dataset, create_data_collator
from eval import create_compute_metrics
from diagnostics import print_trainable_params, print_samples


def main():
    config = get_config()

    # Set random seeds for reproducibility
    set_random_seeds(config.seed)

    # Load and prepare data
    df = load_tsv_data(config.data_file)
    print(f"📊 Loaded {len(df)} examples")

    # Load model and tokenizer
    print(f"\n🤖 Loading model: {config.model_name}")
    tokenizer = ByT5Tokenizer.from_pretrained(config.model_name)
    model = T5ForConditionalGeneration.from_pretrained(config.model_name)
    print_trainable_params(model)

    # Prepare datasets
    tokenized_dataset = prepare_dataset(df, tokenizer, max_length=MAX_LENGTH)
    train_dataset, val_dataset = split_dataset(tokenized_dataset, seed=config.seed)
    print(f"\n✂️  Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # Print sample examples
    print_samples(train_dataset, tokenizer, "Train", num_samples=2)
    print_samples(val_dataset, tokenizer, "Eval", num_samples=2)

    # Setup training components
    data_collator = create_data_collator(tokenizer, model)
    compute_metrics = create_compute_metrics(tokenizer)

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        logging_steps=10,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        seed=config.seed,
        data_seed=config.seed,
        report_to=config.report_to,
    )

    # Train model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\n🔥 Starting training...\n")
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    print("\n✅ Training complete! Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    print(f"💾 Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()
