# CS247-sentiment-analysis

## SVM
SVM using CBOW as input feature:
```
python svm_w2v.py
```
SVM using RoBERTa embedding as input feature:
```
python svm_roberta.py --clean --enable_pos_embed \
    --cache_dir [cache_dir_for 🤗]
```

## Fine-tune RoBERTa
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

## Prompt-tune RoBERTa
See ```prompt_roberta.ipynb```. 

In ```finetune_roberta.ipynb``` there is a corresponding version for fine-tuning using 1k data.

In ```prompt_bert.ipynb``` we also provide the code for prompt-tuning BERT (a bit different from RoBERTa).

## AUTOPROMPT
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