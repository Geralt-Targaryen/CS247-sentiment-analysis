import time
import argparse
import json
import logging
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm

import utils


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module):
        self._stored_gradient = None
        module.register_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient


class PredictWrapper:
    """
    PyTorch transformers model wrapper. Handles necc. preprocessing of inputs for triggers
    experiments.
    """
    def __init__(self, model):
        self._model = model

    def __call__(self, model_inputs, trigger_ids):
        # Copy dict so pop operations don't have unwanted side-effects
        model_inputs = model_inputs.copy()
        trigger_mask = model_inputs.pop('trigger_mask')
        predict_mask = model_inputs.pop('predict_mask')
        model_inputs = replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask)
        
        # input_ids: (bs, len)
        # logits: (bs, len, vocab_size)
        # predict_logits: (bs, vocab_size) - logits chosen by predict_mask in the len dimension
        logits = self._model(**model_inputs).logits
        predict_logits = logits.masked_select(predict_mask.unsqueeze(-1)).view(logits.size(0), -1)
        return predict_logits


class AccuracyFn:
    """
    Computing the accuracy when a label is mapped to multiple tokens is difficult in the current
    framework, since the data generator only gives us the token ids. To get around this we
    compare the target logp to the logp of all labels. If target logp is greater than all (but)
    one of the label logps we know we are accurate.
    """
    def __init__(self, tokenizer, label_map, device, tokenize_labels=False):
        self._all_label_ids = []
        self._pred_to_label = []
        logger.info(label_map)
        for label, label_tokens in label_map.items():
            self._all_label_ids.append(utils.encode_label(tokenizer, label_tokens, tokenize_labels).to(device))
            self._pred_to_label.append(label)
        logger.info(self._all_label_ids)

    def __call__(self, predict_logits, gold_label_ids):
        # Get total log-probability for the true label
        gold_logp = get_loss(predict_logits, gold_label_ids)

        # Get total log-probability for all labels
        bsz = predict_logits.size(0)
        all_label_logp = []
        for label_ids in self._all_label_ids:
            label_logp = get_loss(predict_logits, label_ids.repeat(bsz, 1))
            all_label_logp.append(label_logp)
        all_label_logp = torch.stack(all_label_logp, dim=-1)
        _, predictions = all_label_logp.max(dim=-1)
        predictions = [self._pred_to_label[x] for x in predictions.tolist()]

        # Add up the number of entries where loss is greater than or equal to gold_logp.
        ge_count = all_label_logp.le(gold_logp.unsqueeze(-1)).sum(-1)
        correct = ge_count.le(1)  # less than in case of num. prec. issues

        return correct.float()

    # TODO: @rloganiv - This is hacky. Replace with something sensible.
    def predict(self, predict_logits):
        bsz = predict_logits.size(0)
        all_label_logp = []
        for label_ids in self._all_label_ids:
            label_logp = get_loss(predict_logits, label_ids.repeat(bsz, 1))
            all_label_logp.append(label_logp)
        all_label_logp = torch.stack(all_label_logp, dim=-1)
        _, predictions = all_label_logp.max(dim=-1)
        predictions = [self._pred_to_label[x] for x in predictions.tolist()]
        return predictions


def load_pretrained(model_name, cache_dir):
    """
    Loads pretrained HuggingFace config/model/tokenizer, as well as performs required
    initialization steps to facilitate working with triggers.
    """
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, cache_dir=cache_dir)
    utils.add_task_specific_tokens(tokenizer)
    return config, model, tokenizer


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_embeddings(model, config):
    """Returns the wordpiece embedding module."""
    base_model = getattr(model, config.model_type)
    embeddings = base_model.embeddings.word_embeddings
    return embeddings


def hotflip_attack(averaged_grad,
                   embedding_matrix,
                   increase_loss=False,
                   num_candidates=1,
                   filter=None):
    """Returns the top candidate replacements."""
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(
            embedding_matrix,
            averaged_grad
        )
        if filter is not None:
            gradient_dot_embedding_matrix -= filter
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1
        _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

    return top_k_ids


def replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask):
    """Replaces the trigger tokens in input_ids."""
    out = model_inputs.copy()
    input_ids = model_inputs['input_ids']
    trigger_ids = trigger_ids.repeat(trigger_mask.size(0), 1)
    try:
        filled = input_ids.masked_scatter(trigger_mask, trigger_ids)
    except RuntimeError:
        filled = input_ids
    out['input_ids'] = filled
    return out


def get_loss(predict_logits, label_ids):
    predict_logp = F.log_softmax(predict_logits, dim=-1)
    target_logp = predict_logp.gather(-1, label_ids)
    target_logp = target_logp - 1e32 * label_ids.eq(0)  # Apply mask
    target_logp = torch.logsumexp(target_logp, dim=-1)
    return -target_logp


def run_model(args):
    set_seed(args.seed)
    
    logger.info('Loading model, tokenizer, etc.')
    config, model, tokenizer = load_pretrained(args.model_name, args.cache_dir)
    model.to(device)
    embeddings = get_embeddings(model, config)
    embedding_gradient = GradientStorage(embeddings)
    predictor = PredictWrapper(model)

    label_map = json.loads(args.label_map)
    logger.info(f"Label map: {label_map}")

    templatizer = utils.TriggerTemplatizer(
        args.template,
        tokenizer,
        label_map=label_map,
        label_field=args.label_field,
        tokenize_labels=True,
        add_special_tokens=False,
    )

    # Obtain the initial trigger tokens and label mapping
    if args.initial_trigger:
        trigger_ids = tokenizer.encode(args.initial_trigger, add_special_tokens=False) 
        logger.debug(f'Initial trigger: {args.initial_trigger}')
        logger.debug(f'Trigger ids: {trigger_ids}')
        assert len(trigger_ids) == templatizer.num_trigger_tokens
    else:
        trigger_ids = [tokenizer.mask_token_id] * templatizer.num_trigger_tokens
    trigger_ids = torch.tensor(trigger_ids, device=device).unsqueeze(0)
    best_trigger_ids = trigger_ids.clone()

    evaluation_fn = AccuracyFn(tokenizer, label_map, device)

    logger.info('Loading datasets')
    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)

    train_dataset = utils.load_trigger_dataset(args.train, templatizer, limit=args.limit)
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)

    dev_dataset = utils.load_trigger_dataset(args.dev, templatizer)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)

    # To "filter" unwanted trigger tokens, we subtract a huge number from their logits.
    # [0, 2, 3, 1, 50264, 50265, 50266, 50267]
    # ['<s>', '</s>', '<unk>', '<pad>', '<mask>', '[T]', '[P]', '[Y]']
    filter = torch.zeros(tokenizer.vocab_size, dtype=torch.float32, device=device)
    logger.info('Filtering special tokens.')
    for word, idx in tokenizer.get_vocab().items():
        if len(word) == 1 or idx >= tokenizer.vocab_size:
            continue
        if idx in tokenizer.all_special_ids:
            logger.debug('Filtered: %s', word)
            filter[idx] = -1e32

    logger.info('Evaluating')
    numerator = 0
    denominator = 0
    for model_inputs, labels in tqdm(dev_loader):
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        with torch.no_grad():
            predict_logits = predictor(model_inputs, trigger_ids)
        numerator += evaluation_fn(predict_logits, labels).sum().item()
        denominator += labels.size(0)
    dev_metric = numerator / (denominator + 1e-13)
    logger.info(f'Dev metric: {dev_metric}')

    best_dev_metric = -float('inf')
    
    tic = time.time()

    for i in range(args.iters):
        logger.info(f'Iteration: {i}')
        logger.info('Accumulating Gradient')
        model.zero_grad()

        pbar = tqdm(range(args.accumulation_steps))
        train_iter = iter(train_loader)
        averaged_grad = None

        # Accumulate
        for step in pbar:

            # Shuttle inputs to GPU
            try:
                model_inputs, labels = next(train_iter)
            except:
                logger.warning(
                    'Insufficient data for number of accumulation steps. '
                    'Effective batch size will be smaller than specified.'
                )
                break
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            predict_logits = predictor(model_inputs, trigger_ids)
            loss = get_loss(predict_logits, labels).mean()
            loss.backward()

            grad = embedding_gradient.get()
            bsz, _, emb_dim = grad.size()
            selection_mask = model_inputs['trigger_mask'].unsqueeze(-1)
            grad = torch.masked_select(grad, selection_mask)
            grad = grad.view(bsz, templatizer.num_trigger_tokens, emb_dim)

            if averaged_grad is None:
                averaged_grad = grad.sum(dim=0) / args.accumulation_steps
            else:
                averaged_grad += grad.sum(dim=0) / args.accumulation_steps

        logger.info('Evaluating Candidates')
        pbar = tqdm(range(args.accumulation_steps))
        train_iter = iter(train_loader)

        token_to_flip = random.randrange(templatizer.num_trigger_tokens)
        candidates = hotflip_attack(averaged_grad[token_to_flip],
                                    embeddings.weight,
                                    increase_loss=False,
                                    num_candidates=args.num_cand,
                                    filter=filter)

        current_score = 0
        candidate_scores = torch.zeros(args.num_cand, device=device)
        denom = 0
        for step in pbar:

            try:
                model_inputs, labels = next(train_iter)
            except:
                logger.warning(
                    'Insufficient data for number of accumulation steps. '
                    'Effective batch size will be smaller than specified.'
                )
                break
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            with torch.no_grad():
                predict_logits = predictor(model_inputs, trigger_ids)
                eval_metric = evaluation_fn(predict_logits, labels)

            # Update current score
            current_score += eval_metric.sum()
            denom += labels.size(0)

            # NOTE: Instead of iterating over tokens to flip we randomly change just one each
            # time so the gradients don't get stale.
            for i, candidate in enumerate(candidates):
                temp_trigger = trigger_ids.clone()
                temp_trigger[:, token_to_flip] = candidate
                with torch.no_grad():
                    predict_logits = predictor(model_inputs, temp_trigger)
                    eval_metric = evaluation_fn(predict_logits, labels)

                candidate_scores[i] += eval_metric.sum()

        if (candidate_scores > current_score).any():
            logger.info('Better trigger detected.')
            best_candidate_score = candidate_scores.max()
            best_candidate_idx = candidate_scores.argmax()
            trigger_ids[:, token_to_flip] = candidates[best_candidate_idx]
            logger.info(f'Train metric: {best_candidate_score / (denom + 1e-13): 0.4f}')
        else:
            logger.info('No improvement detected. Skipping evaluation.')
            continue

        logger.info('Evaluating')
        numerator = 0
        denominator = 0
        for model_inputs, labels in tqdm(dev_loader):
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            with torch.no_grad():
                predict_logits = predictor(model_inputs, trigger_ids)
            numerator += evaluation_fn(predict_logits, labels).sum().item()
            denominator += labels.size(0)
        dev_metric = numerator / (denominator + 1e-13)

        logger.info(f'Trigger tokens: {tokenizer.convert_ids_to_tokens(trigger_ids.squeeze(0))}')
        logger.info(f'Dev metric: {dev_metric}')

        if dev_metric > best_dev_metric:
            logger.info('Best performance so far')
            best_trigger_ids = trigger_ids.clone()
            best_dev_metric = dev_metric

    best_trigger_tokens = tokenizer.convert_ids_to_tokens(best_trigger_ids.squeeze(0))
    logger.info(f'Best tokens: {best_trigger_tokens}')
    logger.info(f'Best dev metric: {best_dev_metric}')

    toc = time.time()
    logger.info(f'Training time: {toc-tic}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, default='/projects/models_ps/sa/data/train.csv', help='Train data path')
    parser.add_argument('--dev', type=Path, default='/projects/models_ps/sa/data/dev.csv', help='Dev data path')
    parser.add_argument('--template', type=str, default='<s> {text} [T][T][T][P].</s>', help='Template string')
    parser.add_argument('--label-map', type=str, 
                        default='{"0": ["Ġworkaround", "Ġlament", "Ġmourn", "Ġlamented", "Ġtragedies"], "1": ["Ġachievements", "ĠDai", "ä»", "æĺ", "ĠPhilippe"]}', 
                        help='JSON object defining label map')

    parser.add_argument('--initial-trigger', type=str, default='This comment is', help='Manual prompt')
    parser.add_argument('--label-field', type=str, default='label', help='Name of the label field')

    parser.add_argument('--bsz', type=int, default=24, help='Batch size')
    parser.add_argument('--eval-size', type=int, default=32, help='Eval size')
    parser.add_argument('--iters', type=int, default=180, help='Number of iterations to run trigger search algorithm')
    parser.add_argument('--accumulation-steps', type=int, default=30)
    parser.add_argument('--model-name', type=str, default='roberta-large')
    parser.add_argument('--cache_dir', type=str, default='/projects/cache')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--num-cand', type=int, default=100)

    args = parser.parse_args()

    run_model(args)
