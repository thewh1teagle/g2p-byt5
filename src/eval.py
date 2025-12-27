import numpy as np
import jiwer


def calculate_wer(references, predictions):
    return jiwer.wer(references, predictions)


def calculate_cer(references, predictions):
    return jiwer.cer(references, predictions)


def create_compute_metrics(tokenizer):
    """Create a compute_metrics function for the Trainer."""
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # Replace -100 in labels with pad token id (they were masked for loss calculation)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # Clip predictions to valid vocab range to avoid chr() errors
        # ByT5 vocab size is 259 (256 bytes + 3 special tokens)
        vocab_size = tokenizer.vocab_size
        predictions = np.clip(predictions, 0, vocab_size - 1)

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
