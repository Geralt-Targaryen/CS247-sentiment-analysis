import argparse
import json
import logging
from pathlib import Path
from itertools import count

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig, AutoModelForMaskedLM, AutoTokenizer, RobertaForMaskedLM
)
from tqdm import tqdm

import utils
import create_trigger as ct


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_pretrained(model_name, cache_dir):
    """
    Loads pretrained HuggingFace config/model/tokenizer, as well as performs required
    initialization steps to facilitate working with triggers.
    """
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForMaskedLM.from_pretrained(model_name, config=config, cache_dir=cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    utils.add_task_specific_tokens(tokenizer)
    return config, model, tokenizer


def main(args):
    ct.set_seed(args.seed)

    logger.info('Loading model, tokenizer, etc.')
    config, model, tokenizer = load_pretrained(args.model_name, args.cache_dir)
    model.to(device)
    word_embeddings = model.lm_head.decoder.weight.data.clone().detach()
    model.lm_head.decoder = torch.nn.Identity().to(device)

    label_map = json.loads(args.label_map)
    reverse_label_map = {y: x for x, y in label_map.items()}
    templatizer = utils.TriggerTemplatizer(
        args.template,
        tokenizer,
        label_map=label_map,
        label_field=args.label_field,
        add_special_tokens=False
    )

    # one-layer classifier to help identify the best label words.
    projection = torch.nn.Linear(config.hidden_size, len(label_map)).to(device)

    # Obtain the initial trigger tokens and label mapping
    if args.initial_trigger:
        trigger_ids = tokenizer.encode(args.initial_trigger, add_special_tokens=False) 
        assert len(trigger_ids) == templatizer.num_trigger_tokens
    else:
        trigger_ids = [tokenizer.mask_token_id] * templatizer.num_trigger_tokens
    trigger_ids = torch.tensor(trigger_ids, device=device).unsqueeze(0)

    logger.info('Loading datasets')
    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)
    train_dataset = utils.load_trigger_dataset(args.train, templatizer)
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)

    optimizer = torch.optim.Adam(projection.parameters(), lr=args.lr)

    logger.info('Training')
    cur_step = 0
    for i in count():
        if cur_step >= args.max_step: break
        pbar = tqdm(train_loader)
        for model_inputs, labels in pbar:
            # evaluation
            if cur_step % args.eval_step == 0 or cur_step + args.bsz >= args.max_step:
                scores = torch.matmul(projection.weight, word_embeddings.transpose(0, 1))
                scores = F.softmax(scores, dim=0)
                for i, row in enumerate(scores):
                    _, top = row.topk(args.k)
                    decoded = tokenizer.convert_ids_to_tokens(top)
                    logger.info(f"Top {args.k} for class {reverse_label_map[i]}: {', '.join(decoded)}")
            
            # train
            optimizer.zero_grad()
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            trigger_mask = model_inputs.pop('trigger_mask')
            predict_mask = model_inputs.pop('predict_mask')
            model_inputs = ct.replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask)
            with torch.no_grad():
                # embeddings: (bs, len, 1024)
                embeddings = model(**model_inputs).logits

            predict_embeddings = embeddings.masked_select(predict_mask.unsqueeze(-1)).view(embeddings.size(0), -1)

            logits = projection(predict_embeddings)
            loss = F.cross_entropy(logits, labels.squeeze(-1))
            loss.backward()
            optimizer.step()
            pbar.set_description(f'loss: {loss : 0.4f}')
                
            cur_step += args.bsz
            if cur_step >= args.max_step: break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, default='/projects/models_ps/sa/data/train.csv', help='Train data path')
    parser.add_argument('--template', type=str, default='<s> {text} [T][T][T][P].</s>', help='Template string')
    parser.add_argument('--label-map', type=str, default='{"0": 0, "1": 1}', help='JSON object defining label map')
    parser.add_argument('--initial-trigger', type=str, default='This comment is', help='Manual prompt')
    parser.add_argument('--label-field', type=str, default='label', help='Name of the label field')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--k', type=int, default=10, help='Number of label tokens to print')
    parser.add_argument('--bsz', type=int, default=32, help='Batch size')
    parser.add_argument('--eval_step', default=32768, type=int)
    parser.add_argument('--max_step', default=1000000, type=int)
    parser.add_argument('--model-name', type=str, default='roberta-large')
    parser.add_argument('--cache_dir', type=str, default='/projects/cache')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    main(args)
