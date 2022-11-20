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