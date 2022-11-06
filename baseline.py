import pickle

from transformers import AutoTokenizer, AutoModelForTokenClassification, RobertaForSequenceClassification, RobertaTokenizer
import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from itertools import count
import numpy as np
import pandas as pd
import argparse
import time
import os
from utils import SimpleDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', default='data/training.1600000.processed.noemoticon.csv', type=str)
parser.add_argument('--test_file', default='data/testdata.manual.2009.06.14.csv', type=str)
parser.add_argument('--save_dir', default='models', type=str)
parser.add_argument('--model', default='roberta-large', type=str)
parser.add_argument('--cache_dir', default='/projects/cache', type=str)
parser.add_argument('--train_batch_size', default=8, type=int)
parser.add_argument('--eval_batch_size', default=32, type=int)
parser.add_argument('--learning_rate', default=1e-5, type=float)
parser.add_argument('--weight_decay', default=1e-2, type=float)
parser.add_argument('--lr_shrink', default=0.5, type=float)
parser.add_argument('--tolerance', default=3, type=int)
parser.add_argument('--lr_min', default=1e-7, type=float)
parser.add_argument('--eval_step', default=10000, type=int)
parser.add_argument('--max_step', default=3200000, type=int)
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--checkpoint', default=None, type=str)
args = parser.parse_args()


class Trainer:
    def __init__(self, args):
        self.args = args
        self.train_x = None
        self.train_y = None
        self.dev_x = None
        self.dev_y = None
        self.test_x = None
        self.test_y = None
        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None
        self.train_dataloader = None
        self.dev_dataloader = None
        self.test_dataloader = None
        self.num_train_samples = 0
        self.num_dev_samples = 0
        self.num_test_samples = 0

        # for shrinking lr
        self.tol = args.tolerance
        self.accs = []
        self.lr_shrink = args.lr_shrink
        self.last_checkpoint = ''

        # tokernizer, model, and optimizer
        self.tokenizer: RobertaTokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
        self.model: RobertaForSequenceClassification = AutoModelForTokenClassification.from_pretrained(args.model, cache_dir=args.cache_dir).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()

    def seed(self, seed=1):
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _load_data(self, file, mode):
        df = pd.read_csv(file, encoding="ISO-8859-1", header=None)
        if mode != 'train':
            labels = list(map(lambda x: 1 if x == 4 else 0, df[0].to_list()))
            inputs = df[5].to_list()
            print(f'Loaded test set, number of samples: {len(labels)}')
            return inputs, labels
        else:
            labels = np.array(list(map(lambda x: 1 if x == 4 else 0, df[0].to_list())))
            inputs = df[5].to_numpy()
            p = np.random.permutation(labels.shape[0])
            labels, inputs = labels[p], inputs[p]
            div = round(labels.shape[0] * 0.99)
            train_labels, dev_labels = labels[:div], labels[div:]
            train_inputs, dev_inputs = inputs[:div], inputs[div:]
            print(
                f'Loaded training and development set, number of samples: {train_labels.shape[0]}, {dev_labels.shape[0]}')
            return train_inputs, train_labels, dev_inputs, dev_labels

    def load_data(self):
        self.train_x, self.train_y, self.dev_x, self.dev_y = self._load_data(self.args.train_file, 'train')
        self.test_x, self.test_y = self._load_data(self.args.test_file, 'test')
        self.train_dataset = SimpleDataset(self.train_x, self.train_y)
        self.test_dataset = SimpleDataset(self.test_x, self.test_y)
        self.dev_dataset = SimpleDataset(self.dev_x, self.dev_y)
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.args.train_batch_size, shuffle=True, collate_fn=self.collate_fn
        )
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=self.args.eval_batch_size, shuffle=False, collate_fn=self.collate_fn
        )
        self.dev_dataloader = DataLoader(
            self.dev_dataset, batch_size=self.args.eval_batch_size, shuffle=False, collate_fn=self.collate_fn
        )
        self.num_train_samples = self.train_y.shape[0]
        self.num_dev_samples = self.dev_y.shape[0]
        self.num_test_samples = len(self.test_y)

    def collate_fn(self, batch):
        inputs = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt').to(device)
        labels = torch.tensor(labels, device=device)
        return inputs, labels

    def shrink_lr(self, acc, cur_step):
        if cur_step > 0:
            torch.save(self.model.state_dict(), os.path.join(self.args.save_dir, f'{self.args.model}_{cur_step}.pth'))

        if len(self.accs) <= self.tol:
            self.accs.append(acc)
        else:
            self.accs = self.accs[1:] + [acc]

        if len(self.accs) > 1 and self.accs[-1] < self.accs[-2] * 0.9:
            self.model.load_state_dict(torch.load(self.last_checkpoint))
        else:
            self.last_checkpoint = os.path.join(self.args.save_dir, f'{self.args.model}_{cur_step}.pth')
            if len(self.accs) <= self.tol:
                return False
            for i in range(1, self.tol + 1):
                if self.accs[i] > self.accs[i - 1]:
                    return False

        self.accs.clear()
        self.optimizer.param_groups[0]['lr'] *= self.lr_shrink
        return True

    def train(self):
        dev_accs = []
        cur_step = 0
        self.model.train()
        for epoch in count():
            if cur_step >= self.args.max_step: break
            for step, (x, y) in enumerate(self.train_dataloader):

                if x.input_ids.shape[1] <= 256:
                    # raw output is (bs, len, 2)
                    output = self.model(**x).logits[:, 0, :]
                    loss = self.criterion(output, y)
                    loss.backward()

                else:
                    x1 = {
                        'input_ids': x.input_ids[:self.args.train_batch_size//2,],
                        'attention_mask': x.attention_mask[:self.args.train_batch_size//2,],
                    }
                    output1 = self.model(**x1).logits[:, 0, :]
                    loss = self.criterion(output1, y[:self.args.train_batch_size // 2]) / 2
                    loss.backward()
                    del x1

                    x2 = {
                        'input_ids': x.input_ids[self.args.train_batch_size // 2:,],
                        'attention_mask': x.attention_mask[self.args.train_batch_size // 2:,],
                    }
                    output2 = self.model(**x2).logits[:, 0, :]
                    loss = self.criterion(output2, y[self.args.train_batch_size//2:]) / 2
                    loss.backward()
                    del x2

                self.optimizer.step()
                self.optimizer.zero_grad()

                if cur_step % self.args.eval_step == 0:
                    del x
                    self.model.eval()
                    dev_acc = Accuracy(num_classes=2).to(device)
                    with torch.no_grad():
                        for _, (x_, y_) in enumerate(self.dev_dataloader):
                            output = self.model(**x_).logits[:, 0, :]
                            dev_acc(output.argmax(dim=1), y_)
                    dev_acc = float(dev_acc.compute())
                    dev_accs.append(dev_acc)
                    print('Dev accuracy: %.4f', dev_acc)

                    if self.shrink_lr(dev_acc, cur_step):
                        print(f'Shrinking lr to {self.optimizer.param_groups[0]["lr"]}')
                        if self.optimizer.param_groups[0]["lr"] < self.args.lr_min:
                            cur_step = self.args.max_step
                            break
                    self.model.train()

                cur_step += y.shape[0]
                if cur_step % 50 == 0:
                    print(f'epoch: {epoch}, total step: {cur_step}, loss: %.4f', float(loss))
                if cur_step >= self.args.max_step: break

        with open(os.path.join('figures', 'dev_acc.pkl'), 'wb') as f:
            pickle.dump(dev_accs, f)

    def test(self):
        if self.args.checkpoint:
            self.model.load_state_dict(torch.load(args.checkpoint))
        self.model.eval()
        test_acc = Accuracy(num_classes=2).to(device)
        with torch.no_grad():
            for _, (x_, y_) in enumerate(self.test_dataloader):
                output = self.model(**x_).logits[:, 0, :]
                test_acc(output.argmax(dim=1), y_)
        dev_acc = float(test_acc.compute())
        print('Test accuracy: %.4f', dev_acc)


if __name__ == '__main__':

    trainer = Trainer(args)
    trainer.seed(1)
    trainer.load_data()
    if args.train:
        tic = time.time()
        trainer.train()
        print(f'Training time: {time.time()-tic}s')
    if args.test:
        trainer.test()

