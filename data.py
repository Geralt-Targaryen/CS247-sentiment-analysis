from transformers import AutoTokenizer, RobertaTokenizer
import pickle
import os
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


if __name__ == '__main__':
    trainer = Trainer(args)
    trainer.load_data()
    trainer.plot_lengths()

