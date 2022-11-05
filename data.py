from transformers import AutoTokenizer, RobertaTokenizer
import pickle
import os
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', default='data/training.1600000.processed.noemoticon.csv', type=str)
parser.add_argument('--test_file', default='data/testdata.manual.2009.06.14.csv', type=str)
parser.add_argument('--model', default='roberta-large', type=str)
parser.add_argument('--cache_dir', default='/projects/cache', type=str)

args = parser.parse_args()


class Trainer:
    def __init__(self, args):
        self.args = args
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

        self.input_length = []
        self.token_count_pos = {}
        self.token_count_neg = {}

        self.tokenizer: RobertaTokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)

    def _load_data(self, file):
        df = pd.read_csv(file, encoding="ISO-8859-1", header=None)
        labels = list(map(lambda x: 1 if x == 4 else 0, df[0].to_list()))
        inputs = df[5].to_list()
        return inputs, labels

    def load_data(self):
        self.train_x, self.train_y = self._load_data(self.args.train_file)
        self.test_x, self.test_y = self._load_data(self.args.test_file)

    def count_lengths(self):
        for i, sentence in enumerate(self.train_x):
            self.input_length.append(len(self.tokenizer([sentence], padding=True, truncation=True).input_ids[0]))
            if i % 10000 == 0: print(i)
        with open(os.path.join('figures', 'length_count.pkl'), 'wb') as f:
            pickle.dump(self.input_length, f)

    def plot_lengths(self):
        with open(os.path.join('figures', 'length_count.pkl'), 'rb') as f:
            self.input_length: list = pickle.load(f)
        self.input_length.sort()
        print(self.input_length[:100])
        print(self.input_length[-100:])
        x = np.array(range(len(self.input_length)))
        plt.plot(x, self.input_length)
        plt.savefig(os.path.join('figures', 'length_plot'), dpi=300)

    def count_tokens(self):
        self.train_x, self.train_y = np.array(self.train_x), np.array(self.train_y)
        x_pos = self.train_x[self.train_y==1]
        x_neg = self.train_x[self.train_y==0]
        for i, sentence in enumerate(x_pos):
            tokens = self.tokenizer.tokenize(sentence, padding=True, truncation=True)
            for token in tokens:
                self.token_count_pos[token] = self.token_count_pos.get(token, 0) + 1
            if i % 10000 == 0: print(i)
        with open(os.path.join('figures', 'token_count_pos.pkl'), 'wb') as f:
            pickle.dump(self.token_count_pos, f)

        for i, sentence in enumerate(x_neg):
            tokens = self.tokenizer.tokenize(sentence, padding=True, truncation=True)
            for token in tokens:
                self.token_count_neg[token] = self.token_count_neg.get(token, 0) + 1
            if i % 10000 == 0: print(i)
        with open(os.path.join('figures', 'token_count_neg.pkl'), 'wb') as f:
            pickle.dump(self.token_count_neg, f)

    def plot_token_count(self):
        with open(os.path.join('figures', 'token_count_pos.pkl'), 'rb') as f:
            self.token_count_pos: dict = pickle.load(f)
        with open(os.path.join('figures', 'token_count_neg.pkl'), 'rb') as f:
            self.token_count_neg: dict = pickle.load(f)

        token_pos_list = sorted(self.token_count_pos, key=lambda k: -self.token_count_pos[k])
        token_neg_list = sorted(self.token_count_neg, key=lambda k: -self.token_count_neg[k])
        tokens_pos = []
        tokens_neg = []
        # print(token_pos_list[:300])
        # print(token_neg_list[:300])
        for i, token in enumerate(token_pos_list):
            if token not in token_neg_list[:i+100]:
                tokens_pos.append(token)
            if len(tokens_pos) > 50:
                break
        for i, token in enumerate(token_neg_list):
            if token not in token_pos_list[:i+100]:
                tokens_neg.append(token)
            if len(tokens_neg) > 50:
                break

        print(tokens_pos)
        print(tokens_neg)
        df_pos = pd.DataFrame({
            'token': [t[1:] if t[0] == 'Ġ' else t for t in tokens_pos],
            'count': [self.token_count_pos[t] for t in tokens_pos]
        })
        df_neg = pd.DataFrame({
            'token': [t[1:] if t[0] == 'Ġ' else t for t in tokens_neg],
            'count': [self.token_count_neg[t] for t in tokens_neg]
        })

        fig = px.treemap(df_pos, path=[px.Constant('tokens'), 'token'], values='count',
                         color='count', color_continuous_scale='RdBu',
                         color_continuous_midpoint=np.average(df_pos['count'], weights=df_pos['count']),
                         width=700)
        fig.update_layout(margin=dict(t=25, l=25, r=25, b=25))
        fig.show()

        fig = px.treemap(df_neg, path=[px.Constant('tokens'), 'token'], values='count',
                         color='count', color_continuous_scale='RdBu',
                         color_continuous_midpoint=np.average(df_neg['count'], weights=df_neg['count']),
                         width=700)
        fig.update_layout(margin=dict(t=25, l=25, r=25, b=25))
        fig.show()


if __name__ == '__main__':
    trainer = Trainer(args)
    trainer.load_data()
