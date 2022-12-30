PREPROCESS = False

import pandas as pd
import numpy as np
df=pd.read_csv('data/training.1600000.processed.noemoticon.csv', encoding="ISO-8859-1",header=None)
df=df.loc[:,[0,5]]
df.columns=['label','sentence']
print(df)
print(df[df['label']==0].shape[0])
df=df.sample(frac=1)[:10000]
df['label']=df['label'].apply(lambda x:1 if x==4 else 0)

#清洗数据
import re
from string import punctuation
def sentence_clean(x):
    x=x.lower()
    x=re.sub(r'@[A-Za-z0-9\._]*','',x)
    x=re.sub(r'[A-Za-z]+://[^\s]*','',x)
    x=re.sub(r'[{}]+'.format(punctuation),'',x)
    x=re.sub(r':\)','smile',x)
    x=re.sub(r':\(','sad',x)
    x=re.sub(r' +',' ',x)
    return x.split()

if PREPROCESS: df['token']=df['sentence'].apply(sentence_clean)
else: df['token']=df['sentence'].apply(lambda x:x.split())
print(df.head())

#学习word2vec
from gensim.models import Word2Vec
df['token']=df['token'].apply(lambda x:None if x==[] else x)
df=df[df['token'].notnull()]
corpus_token=df['token'].values
label=df['label'].values

#测试
df=pd.read_csv('data/testdata.manual.2009.06.14.csv',encoding="ISO-8859-1",header=None)
df=df.loc[:,[0,5]]
df.columns=['label','sentence']
print(df[df['label']==0].shape[0])
df['label']=df['label'].apply(lambda x:1 if x==4 else 0)
if PREPROCESS: df['token']=df['sentence'].apply(sentence_clean)
else: df['token']=df['sentence'].apply(lambda x:x.split())
df['token']=df['token'].apply(lambda x:None if x==[] else x)
df=df[df['token'].notnull()]
corpus_token_test=df['token'].values

model=Word2Vec(corpus_token,vector_size=1024,window=5,min_count=1,workers=8, epochs=50)


# print(model.wv.key_to_index.keys())
# print(model.wv['sad'])

#计算句向量
features=[np.mean(model.wv[x],axis=0) for x in corpus_token]
print(np.array(features).shape)


#svm分类器
from sklearn.model_selection import train_test_split

x_train,x_valid,y_train,y_valid=train_test_split(features,label,train_size=0.9)
from sklearn.svm import SVC
from sklearn.metrics import f1_score

print('kernel\tC\tacc\tf1')
for kernel in ['linear']:
    for C in [1e-2]:
        svm=SVC(kernel=kernel,C=C)
        svm.fit(x_train,y_train)
        #预测
        acc = svm.score(x_valid,y_valid)
        y_pred = svm.predict(x_valid)
        f1 = f1_score(y_valid,y_pred,average='binary')
        print(f'{kernel}\t{C}\t%.4f\t%.4f'%(acc, f1))





for i, sent in enumerate(corpus_token_test):
    sent_new = [token for token in sent if token in model.wv]
    corpus_token_test[i] = sent_new

#计算句向量
features=[np.mean(model.wv[x],axis=0) for x in corpus_token_test]
print(np.array(features).shape)
#svm分类器
label=df['label'].values
print('test acc')
acc = svm.score(features,label)
print(acc)