# G2P BYT5

Grapheme to Phoneme model using BYT5.

## Training

```console
uv run src/train.py \
    --model_name google/byt5-large \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --data_file data.tsv \
    --eval_file eval.tsv \
    --eval_steps 3000 \
    --save_steps 3000
```