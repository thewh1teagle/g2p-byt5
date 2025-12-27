"""
uv run src/infer.py --model_path models/byt5-large --text "היי, מה השבוע?"
"""
from transformers import T5ForConditionalGeneration, ByT5Tokenizer
from config import MAX_LENGTH


def load_model(model_path):
    """Load a trained model and tokenizer."""
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = ByT5Tokenizer.from_pretrained(model_path)
    return model, tokenizer


def predict(model, tokenizer, text, max_length=MAX_LENGTH, num_beams=4):
    """Generate prediction for a single input text."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    )

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=num_beams,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def predict_batch(model, tokenizer, texts, max_length=MAX_LENGTH, num_beams=4):
    """Generate predictions for a batch of input texts."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True,
    )

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=num_beams,
    )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--text", type=str, required=True, help="Input text to predict")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams for beam search")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path)
    prediction = predict(model, tokenizer, args.text, num_beams=args.num_beams)

    print(f"Input: {args.text}")
    print(f"Output: {prediction}")
