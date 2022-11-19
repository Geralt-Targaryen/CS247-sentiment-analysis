from transformers import AutoTokenizer, RobertaTokenizer, AutoModelForTokenClassification, RobertaForSequenceClassification
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import pickle
import torch
import os
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', default='data/training.1600000.processed.noemoticon.csv', type=str)
parser.add_argument('--test_file', default='data/testdata.manual.2009.06.14.csv', type=str)
parser.add_argument('--train_size', default=300000, type=int)
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

        self.tokenizer: RobertaTokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)

    def _load_data(self, file, mode='train'):
        df = pd.read_csv(file, encoding="ISO-8859-1", header=None)
        if mode != 'train':
            labels = np.array(list(map(lambda x: 1 if x == 4 else 0, df[0].to_list())))
            inputs = df[5].to_numpy()
        else:
            labels = np.array(list(map(lambda x: 1 if x == 4 else 0, df[0].to_list())))
            inputs = df[5].to_numpy()
            p = np.random.permutation(labels.shape[0])
            labels, inputs = labels[p][:self.args.train_size], inputs[p][:self.args.train_size]

        return inputs, labels

    def load_data(self):
        self.train_x, self.train_y = self._load_data(self.args.train_file)
        self.test_x, self.test_y = self._load_data(self.args.test_file, 'test')

    def embed(self, pos=True):
        model: RobertaForSequenceClassification = AutoModelForTokenClassification.from_pretrained('roberta-large', cache_dir='/projects/cache')
        model = model.roberta.embeddings
        if not pos:
            model.position_embeddings.weight.data = torch.zeros((514, 1024))
        model.eval()
        embeddings = []
        for i, sentence in enumerate(self.train_x):
            input = self.tokenizer(sentence, padding=True, truncation=True, return_tensors='pt').input_ids
            with torch.no_grad():
                embedding = model(input)[:,1:-1,:].mean(dim=1)
            embeddings.append(embedding)
            if i % 1000 == 0:
                print(i)
        embeddings = torch.concat(embeddings).numpy()
        with open(os.path.join('data', 'train_x.pkl'), 'wb') as f:
            pickle.dump(embeddings, f)

        embeddings = []
        for i, sentence in enumerate(self.test_x):
            input = self.tokenizer(sentence, padding=True, truncation=True, return_tensors='pt').input_ids
            with torch.no_grad():
                embedding = model(input)[:,1:-1,:].mean(dim=1)
            embeddings.append(embedding)
        embeddings = torch.concat(embeddings).numpy()
        with open(os.path.join('data', 'test_x.pkl'), 'wb') as f:
            pickle.dump(embeddings, f)

        with open(os.path.join('data', 'train_y.pkl'), 'wb') as f:
            pickle.dump(self.train_y, f)

        with open(os.path.join('data', 'test_y.pkl'), 'wb') as f:
            pickle.dump(self.test_y, f)
    
    def svm_cv(self, kernel='linear'):
        with open(os.path.join('data', 'train_x.pkl'), 'rb') as f:
            train_dev_x = pickle.load(f)
        with open(os.path.join('data', 'train_y.pkl'), 'rb') as f:
            train_dev_y = pickle.load(f)
        with open(os.path.join('data', 'test_x.pkl'), 'rb') as f:
            test_x = pickle.load(f)
        with open(os.path.join('data', 'test_y.pkl'), 'rb') as f:
            test_y = pickle.load(f)

        kf = KFold(n_splits=5)
        print(f'C\tfold 1\tfold 2\tfold 3\tfold 4\tfold 5\t Avg')
        for C in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]:
            print(C, end='\t')
            accs = []
            for train_index, dev_index in kf.split(train_dev_x):
                train_x, dev_x = train_dev_x[train_index], train_dev_x[dev_index]
                train_y, dev_y = train_dev_y[train_index], train_dev_y[dev_index]

                if kernel == 'linear':
                    model = svm.LinearSVC(C=C)
                else:
                    model = svm.SVC(kernel=kernel, C=C)
                model.fit(train_x, train_y)

                prediction = model.predict(dev_x)
                acc = accuracy_score(dev_y, prediction)
                print('%.4f' % acc, end='\t')
                accs.append(acc)
            print('%.4f' % np.mean(accs))

    def svm(self, C, kernel='linear'):
        with open(os.path.join('data', 'train_x.pkl'), 'rb') as f:
            train_x = pickle.load(f)
        with open(os.path.join('data', 'train_y.pkl'), 'rb') as f:
            train_y = pickle.load(f)
        with open(os.path.join('data', 'test_x.pkl'), 'rb') as f:
            test_x = pickle.load(f)
        with open(os.path.join('data', 'test_y.pkl'), 'rb') as f:
            test_y = pickle.load(f)

        if kernel == 'linear':
            model = svm.LinearSVC(C=C)
        else:
            model = svm.SVC(kernel=kernel, C=C)
        model.fit(train_x, train_y)

        prediction = model.predict(test_x)
        acc = accuracy_score(test_y, prediction)
        print(f'SVM with {kernel} kernel, C={C}, test acc: %.4f' % acc)


if __name__ == '__main__':
    trainer = Trainer(args)
    trainer.load_data()
    trainer.embed(pos=False)
    # trainer.svm_cv()
    trainer.svm(C=0.01)