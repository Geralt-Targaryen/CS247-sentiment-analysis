from transformers import AutoTokenizer, RobertaTokenizer, AutoModelForTokenClassification, RobertaForSequenceClassification
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import torch
import os
import pandas as pd
import numpy as np
import argparse
import re
from tqdm import tqdm
from string import punctuation
def sentence_clean(x):
    x=x.lower()
    x=re.sub(r'@[A-Za-z0-9\._]*','',x)
    x=re.sub(r'[A-Za-z]+://[^\s]*','',x)
    x=re.sub(r'[{}]+'.format(punctuation),'',x)
    x=re.sub(r':\)','smile',x)
    x=re.sub(r':\(','sad',x)
    x=re.sub(r' +',' ',x)
    return x

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', default='data/training.1600000.processed.noemoticon.csv', type=str)
parser.add_argument('--test_file', default='data/testdata.manual.2009.06.14.csv', type=str)
parser.add_argument('--train_size', default=10000, type=int)
parser.add_argument('--clean', action='store_true')
parser.add_argument('--enable_pos_embed', action='store_true')
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
        self.train_feature = None
        self.dev_feature = None
        self.test_feature = None
        self.train_label = None
        self.dev_label = None
        self.test_label = None
        self.svm = None

        self.tokenizer: RobertaTokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)

    def _load_data(self, file, mode='train'):
        df = pd.read_csv(file, encoding="ISO-8859-1", header=None)
        df = df.loc[:,[0,5]]
        df.columns = ['label','sentence']
        if mode == 'train': df=df.sample(frac=1, random_state=0)[:self.args.train_size]
        df['label'] = df['label'].apply(lambda x:1 if x==4 else 0)

        if args.clean: df['sentence']=df['sentence'].apply(sentence_clean)
        print(df.head())

        labels = df['label'].to_numpy()
        inputs = df['sentence'].to_numpy()

        return inputs, labels

    def load_data(self):
        self.train_x, self.train_y = self._load_data(self.args.train_file)
        self.test_x, self.test_y = self._load_data(self.args.test_file, 'test')

    def embed(self):
        model: RobertaForSequenceClassification = AutoModelForTokenClassification.from_pretrained('roberta-large', cache_dir='/projects/cache')
        model = model.roberta.embeddings
        if not self.args.enable_pos_embed:
            model.position_embeddings.weight.data = torch.zeros((514, 1024))
        model.eval()
        embeddings = []
        for sentence in tqdm(self.train_x):
            input = self.tokenizer(sentence, padding=True, truncation=True, return_tensors='pt').input_ids
            with torch.no_grad():
                embedding = model(input)[:,1:-1,:].mean(dim=1)
            embeddings.append(embedding)
        embeddings = torch.concat(embeddings).numpy()
        self.train_feature, self.dev_feature, self.train_label, self.dev_label = train_test_split(embeddings, self.train_y,train_size=0.9)

        embeddings = []
        for sentence in tqdm(self.test_x):
            input = self.tokenizer(sentence, padding=True, truncation=True, return_tensors='pt').input_ids
            with torch.no_grad():
                embedding = model(input)[:,1:-1,:].mean(dim=1)
            embeddings.append(embedding)
        self.test_feature = torch.concat(embeddings).numpy()
        self.test_label = self.test_y

        print('trianing features: ', self.train_feature.shape)
        print('validation featuers: ', self.dev_feature.shape)
        print('test features: ', self.test_feature.shape)
    
    def svm_cv(self):
        print('kernel\tC\tacc')
        for kernel in ['linear']:
            for C in [0.01]:
                svm = SVC(kernel=kernel,C=C)
                svm.fit(self.train_feature, self.train_label)
                self.svm = svm
                acc = svm.score(self.dev_feature, self.dev_label)
                print(f'{kernel}\t{C}\t%.4f'%(acc))

    def test(self):
        print('test acc')
        print('%.4f' % self.svm.score(self.test_feature, self.test_label))


if __name__ == '__main__':
    trainer = Trainer(args)
    trainer.load_data()
    trainer.embed()
    trainer.svm_cv()
    trainer.test()