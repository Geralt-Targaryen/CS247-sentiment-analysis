# CS247-sentiment-analysis

## Fine-tune roberta
Train (with default hyper-parameters)
```
python -u baseline.py --train \
    --cache_dir [cache_dir_for 🤗] \
    --train_file [train_file] \
    --test_file [test_file]
```

Test
```
python baseline.py --test \
    --cache_dir [cache_dir_for 🤗] \
    --train_file [train_file] \
    --test_file [test_file] \ 
    --checkpoint [saved_checkpoint]
```

## Prompt-tune roberta
#### Search for labels (with default hyper-parameters)
```
python -u label_search.py 
    --train [train_file] \
    --cache_dir [cache_dir_for 🤗] \
    --template '<s> {text} [T][T][T][P].</s>' \
    --label-map '{"0": 0, "1": 1}' \
    --initial-trigger 'This comment is'
```

### Search for triggers (with labels found in the previous step and default hyper-parameter)
**WARNING: I have not gone thorugh the whole dataset yet, and some extra-large batch may cause cuda out-of-memory with the current defaul batch size**
```
python -u create_trigger.py \
    --train data/train.csv \
    --dev data/dev.csv \
    --cache_dir [cache_dir_for 🤗] \
    --template '<s> {text} [T][T][T][P].</s>' \
    --label-map '{"0": ["Ġworkaround", "Ġlament", "Ġmourn", "Ġlamented", "Ġtragedies"], "1": ["Ġachievements", "ĠDai", "ä»", "æĺ", "ĠPhilippe"]}' --initial-trigger 'This comment is'
```