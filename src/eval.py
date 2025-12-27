import numpy as np
import jiwer


def calculate_wer(references, predictions):
    return jiwer.wer(references, predictions)


def calculate_cer(references, predictions):
    return jiwer.cer(references, predictions)


def preprocess_logits_for_metrics(logits, labels):
    """Convert logits to predictions before passing to compute_metrics."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return np.argmax(logits, axis=-1)


def create_compute_metrics(tokenizer):
    """Create a compute_metrics function for the Trainer."""
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # Predictions are already token IDs from preprocess_logits_for_metrics
        # Replace -100 in labels with pad token id (they were masked for loss calculation)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Calculate WER and CER
        wer = calculate_wer(decoded_labels, decoded_preds)
        cer = calculate_cer(decoded_labels, decoded_preds)

        return {
            "wer": wer,
            "cer": cer,
        }

    return compute_metrics
