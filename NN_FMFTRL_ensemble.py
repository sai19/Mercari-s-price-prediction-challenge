import gc
import time
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import lightgbm as lgb

import sys

#Add https://www.kaggle.com/anttip/wordbatch to your kernel Data Sources, 
#until Kaggle admins fix the wordbatch pip package installation
sys.path.insert(0, '../input/wordbatch/wordbatch/')
import wordbatch

from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL
from nltk.corpus import stopwords
import re
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import string
import multiprocessing
import threading
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn import preprocessing
from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate, GRU, Embedding, Flatten, BatchNormalization,AlphaDropout,GaussianDropout
from keras.models import Model                                                                                                          
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Flatten,Activation 
from keras.layers import Merge
from sklearn.model_selection import train_test_split
import nltk
import gensim
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop, Nadam
from keras import regularizers
from subprocess import check_output
#translator = str.maketrans('', '', string.punctuation)
from nltk.corpus import stopwords
from keras import backend as K
import gc
import time
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import itertools
from collections import Counter

translator = str.maketrans('', '', string.punctuation)
stops = set(stopwords.words("english")) 
total = 0;
np.random.seed(123)
def get_model(nCat,nbrand,nDescrip,nName,maxlenName,maxlenDescrip):
    modelCat = Sequential()
    modelCat.add(Embedding(nCat+1,16, input_length=1,embeddings_initializer='he_normal'))
    modelCat.add(Flatten())
    modelCat.add(Activation('tanh'))
    modelCat.add(BatchNormalization())
    modelCat.add(Dropout(0.1))
    
    modelBran = Sequential()  
    modelBran.add(Embedding(nbrand+1,32,input_length=1,embeddings_initializer='he_normal'))
    modelBran.add(Flatten())
    modelBran.add(Activation('tanh'))
    modelBran.add(BatchNormalization())
    modelBran.add(Dropout(0.1))
    
    modelItemC = Sequential()
    modelItemC.add(Embedding(6,4,input_length=1,embeddings_initializer='he_normal'))
    modelItemC.add(Flatten())
    modelItemC.add(Activation('tanh'))
    modelItemC.add(Dropout(0.1))
    
    
    modelShipp = Sequential() 
    modelShipp.add(Embedding(3,2,input_length=1,embeddings_initializer='he_normal'))									
    modelShipp.add(Flatten())
    modelShipp.add(Activation('tanh'))
    modelShipp.add(Dropout(0.1))
    
    modelDescrip = Sequential()
    modelDescrip.add(Embedding(nDescrip+1,16, input_length=maxlenDescrip,embeddings_initializer='he_normal'))
    modelDescrip.add(Flatten())
    modelDescrip.add(Dropout(0.6))
    modelDescrip.add(Dense(32,kernel_initializer='he_normal'))
    modelDescrip.add(Activation('tanh'))
    modelDescrip.add(BatchNormalization())
    
    modelName = Sequential()
    modelName.add(Embedding(nName+1,8, input_length=maxlenName,embeddings_initializer='he_normal'))
    modelName.add(Flatten())
    modelName.add(Dropout(0.2))
    modelName.add(Dense(32,kernel_initializer='he_normal'))
    modelName.add(Activation('tanh'))
    modelName.add(BatchNormalization())
    
    merged = Merge([modelCat,modelBran,modelItemC,modelShipp,modelDescrip,modelName], mode='concat') 
    
    model = Sequential()
    model.add(merged) 
    model.add(Dense(128,kernel_initializer='he_normal'))
    model.add(Activation('tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='he_normal',activation='relu'))
    return model


def replace_synonyms(sentence_string,word2index):
    out = [];
    for w in sentence_string:
        try:
            out.append(word2index[w])
        except:
            pass
    return out

def review_to_wordlist(review):
    review = str(review)
    review = review.translate(translator)
    review = review.lower()
    review = review.split()
    review = [w for w in review if w not in stops]
    return review 

NUM_BRANDS = 4500
NUM_CATEGORIES = 1250

develop = False
# develop= True

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))


def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("No Label", "No Label", "No Label")


def handle_missing_inplace(dataset):
    dataset['general_cat'].fillna(value='missing', inplace=True)
    dataset['subcat_1'].fillna(value='missing', inplace=True)
    dataset['subcat_2'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)


def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category1 = dataset['general_cat'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category2 = dataset['subcat_1'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category3 = dataset['subcat_2'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    dataset.loc[~dataset['general_cat'].isin(pop_category1), 'general_cat'] = 'missing'
    dataset.loc[~dataset['subcat_1'].isin(pop_category2), 'subcat_1'] = 'missing'
    dataset.loc[~dataset['subcat_2'].isin(pop_category3), 'subcat_2'] = 'missing'


def to_categorical(dataset):
    dataset['general_cat'] = dataset['general_cat'].astype('category')
    dataset['subcat_1'] = dataset['subcat_1'].astype('category')
    dataset['subcat_2'] = dataset['subcat_2'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')


# Define helpers for text normalization
stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')


def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])

def get_pred_nn(submission):
    t1 = time.time()
    train = pd.read_table('../input/mercari-price-suggestion-challenge/train.tsv', engine='c')
    test = pd.read_table('../input/mercari-price-suggestion-challenge/test.tsv', engine='c')
    train["item_description"] = train["item_description"].fillna(value="missing")
    test["item_description"] = test["item_description"].fillna(value="missing")
    train["words"] = train["item_description"].apply(lambda x:review_to_wordlist(x))
    test["words"] = test["item_description"].apply(lambda x:review_to_wordlist(x))
    combined = list(itertools.chain.from_iterable(list(train["words"])+list(test["words"])))
    index2word = [word for word,count in dict(Counter(combined)).items() if count>=100]
    word2index = dict((word,index+1) for index,word in enumerate(index2word))
    nDescrip = len(word2index)
    train["words"] = train["words"].apply(lambda x:replace_synonyms(x,word2index))
    test["words"] = test["words"].apply(lambda x:replace_synonyms(x,word2index))
    train["name"] = train["name"].fillna(value="missing")
    test["name"] = test["name"].fillna(value="missing")
    train["name"] = train["name"].apply(lambda x:review_to_wordlist(x))
    test["name"] = test["name"].apply(lambda x:review_to_wordlist(x))
    combined = list(itertools.chain.from_iterable(list(train["name"])+list(test["name"])))
    index2word = [word for word,count in dict(Counter(combined)).items() if count>=40]
    word2index = dict((word,index+1) for index,word in enumerate(index2word))
    nName = len(word2index)
    print(nDescrip,nName)
    train["name"] = train["name"].apply(lambda x:replace_synonyms(x,word2index))
    test["name"] = test["name"].apply(lambda x:replace_synonyms(x,word2index))
    n1 = max(train["name"].apply(lambda x:len(x)))
    n2 = max(test["name"].apply(lambda x:len(x)))
    maxlenName = 8
    n1 = max(train["words"].apply(lambda x:len(x)))
    n2 = max(test["words"].apply(lambda x:len(x)))
    maxlenDescrip = 32
    #'''
    train["category_name"] = train["category_name"].fillna("missing")  
    test["category_name"] = test["category_name"].fillna("missing")
    train["brand_name"] = train["brand_name"].fillna("missing")
    test["brand_name"] = test["brand_name"].fillna("missing")
    nCat = len(set(list(train["category_name"])+list(test["category_name"])))
    nBrand = len(set(list(train["brand_name"])+list(test["brand_name"])))
    leCat = preprocessing.LabelEncoder()
    leCat.fit(np.array(list(set(list(train["category_name"])+list(test["category_name"])))))
    leBran = preprocessing.LabelEncoder()
    leBran.fit(np.array(list(set(list(train["brand_name"])+list(test["brand_name"])))))
    leCon = preprocessing.LabelEncoder()
    leCon.fit(np.array(list(set(train["item_condition_id"]))))
    leShipp = preprocessing.LabelEncoder()
    leShipp.fit(np.array(list(set(train["shipping"]))))
    train["category_name"] = leCat.transform(train["category_name"])
    test["category_name"] = leCat.transform(test["category_name"])
    train["brand_name"] = leBran.transform(train["brand_name"])
    test["brand_name"] = leBran.transform(test["brand_name"])
    train["item_condition_id"] = leCon.transform(train["item_condition_id"])  
    test["item_condition_id"] = leCon.transform(test["item_condition_id"])  
    train["shipping"] = leShipp.transform(train["shipping"])   
    test["shipping"] = leShipp.transform(test["shipping"])
    del leCat,leBran,leCon,leShipp
    train = train[train["price"]!=0]
    Xtrain, Xvalid = train_test_split(train, test_size=0.01,random_state=1)
    del train
    valid_data_features = pad_sequences(Xvalid["words"],maxlen=maxlenDescrip,padding='post')
    train_data_features = pad_sequences(Xtrain["words"],maxlen=maxlenDescrip,padding='post')
    test_data_features = pad_sequences(test["words"],maxlen=maxlenDescrip,padding='post')
    valid_name_features = pad_sequences(Xvalid["name"],maxlen=maxlenName,padding='post')
    train_name_features = pad_sequences(Xtrain["name"],maxlen=maxlenName,padding='post')
    test_name_features = pad_sequences(test["name"],maxlen=maxlenName,padding='post')
    print(test_data_features.shape,test_name_features.shape)
    XtrainData = np.concatenate((Xtrain[["category_name","brand_name","item_condition_id","shipping"]].values,train_data_features),axis=1)
    XtrainData = np.concatenate((XtrainData,train_name_features),axis=1)
    y_train = np.array(np.log1p(Xtrain["price"]))
    del Xtrain,train_data_features,train_name_features
    XvalidData = np.concatenate((Xvalid[["category_name","brand_name","item_condition_id","shipping"]].values,valid_data_features),axis=1)
    XvalidData = np.concatenate((XvalidData,valid_name_features),axis=1)
    y_valid = np.array(np.log1p(Xvalid["price"]))
    del Xvalid,valid_data_features,valid_name_features
    testData = np.concatenate((test[["category_name","brand_name","item_condition_id","shipping"]].values,test_data_features),axis=1)
    testData = np.concatenate((testData,test_name_features),axis=1)
    del test_name_features,test_data_features
    epochs = 25
    batch_size = 8192
    model_train = get_model(nCat,nBrand,nDescrip,nName,maxlenName,maxlenDescrip)
    eval_score = 20.0
    learning_rate = 0.1
    for i in range(epochs):
        if i%10==0:
            learning_rate /= 10
            nadam = Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
            model_train.compile(loss="mse", optimizer=nadam)
        model_train.fit(batch_size=batch_size,verbose=2,epochs=1,x=[XtrainData[:,0],XtrainData[:,1],XtrainData[:,2],XtrainData[:,3],XtrainData[:,4:4+maxlenDescrip],XtrainData[:,4+maxlenDescrip:]],y=y_train)
        val_score = model_train.evaluate(x=[XvalidData[:,0],XvalidData[:,1],XvalidData[:,2],XvalidData[:,3],XvalidData[:,4:4+maxlenDescrip],XvalidData[:,4+maxlenDescrip:]],y=y_valid,verbose=2)
        print(val_score)
        if val_score<eval_score:
            eval_score = val_score
            pred = model_train.predict(batch_size=batch_size,x=[testData[:,0],testData[:,1],testData[:,2],testData[:,3],testData[:,4:4+maxlenDescrip],testData[:,4+maxlenDescrip:]])
    t2 = time.time()
    print(t2-t1)
    del XtrainData,testData
    del model_train
    gc.collect()
    submission["price_nn"] = pred

def get_pred_ftrl(submission):
    start_time = time.time()
    from time import gmtime, strftime
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    # if 1 == 1:
    train = pd.read_table('../input/mercari-price-suggestion-challenge/train.tsv', engine='c')
    test = pd.read_table('../input/mercari-price-suggestion-challenge/test.tsv', engine='c')

    #train = pd.read_table('../input/train.tsv', engine='c')
    #test = pd.read_table('../input/test.tsv', engine='c')

    print('[{}] Finished to load data'.format(time.time() - start_time))
    print('Train shape: ', train.shape)
    print('Test shape: ', test.shape)
    nrow_test = train.shape[0]  # -dftt.shape[0]
    train = train[train["price"]!=0]
    #Xtrain,Xvalid = train_test_split(train, test_size=0.01,random_state=1)
    nrow_train = train.shape[0]
    #nrow_valid = Xvalid.shape[0]
    # print(nrow_train, nrow_test)
    y = np.log1p(train["price"])
    merge: pd.DataFrame = pd.concat([train, test])
    #submission: pd.DataFrame = test[['test_id']]

    del train
    del test
    gc.collect()

    merge['general_cat'], merge['subcat_1'], merge['subcat_2'] = \
        zip(*merge['category_name'].apply(lambda x: split_cat(x)))
    merge.drop('category_name', axis=1, inplace=True)
    print('[{}] Split categories completed.'.format(time.time() - start_time))

    handle_missing_inplace(merge)
    print('[{}] Handle missing completed.'.format(time.time() - start_time))

    cutting(merge)
    print('[{}] Cut completed.'.format(time.time() - start_time))

    to_categorical(merge)
    print('[{}] Convert categorical completed'.format(time.time() - start_time))

    wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0],
                                                                  "hash_size": 2 ** 29, "norm": None, "tf": 'binary',
                                                                  "idf": None,
                                                                  }), procs=8)
    wb.dictionary_freeze= True
    X_name = wb.fit_transform(merge['name'])
    del(wb)
    X_name = X_name[:, np.array(np.clip(X_name.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
    print('[{}] Vectorize `name` completed.'.format(time.time() - start_time))

    wb = CountVectorizer()
    X_category1 = wb.fit_transform(merge['general_cat'])
    X_category2 = wb.fit_transform(merge['subcat_1'])
    X_category3 = wb.fit_transform(merge['subcat_2'])
    print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))

    # wb= wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 3, "hash_ngrams_weights": [1.0, 1.0, 0.5],
    wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 1.0],
                                                                  "hash_size": 2 ** 28, "norm": "l2", "tf": 1.0,
                                                                  "idf": None})
                             , procs=8)
    wb.dictionary_freeze= True
    X_description = wb.fit_transform(merge['item_description'])
    del(wb)
    X_description = X_description[:, np.array(np.clip(X_description.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
    print('[{}] Vectorize `item_description` completed.'.format(time.time() - start_time))

    lb = LabelBinarizer(sparse_output=True)
    X_brand = lb.fit_transform(merge['brand_name'])
    print('[{}] Label binarize `brand_name` completed.'.format(time.time() - start_time))

    X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                          sparse=True).values)
    print('[{}] Get dummies on `item_condition_id` and `shipping` completed.'.format(time.time() - start_time))
    print(X_dummies.shape, X_description.shape, X_brand.shape, X_category1.shape, X_category2.shape, X_category3.shape,
          X_name.shape)
    sparse_merge = hstack((X_dummies, X_description, X_brand, X_category1, X_category2, X_category3, X_name)).tocsr()

    print('[{}] Create sparse merge completed'.format(time.time() - start_time))

    #    pd.to_pickle((sparse_merge, y), "xy.pkl")
    # else:
    #    nrow_train, nrow_test= 1481661, 1482535
    #    sparse_merge, y = pd.read_pickle("xy.pkl")

    # Remove features with document frequency <=1
    print(sparse_merge.shape)
    mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 1, 0, 1), dtype=bool)
    sparse_merge = sparse_merge[:, mask]
    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_train:]
    print(sparse_merge.shape)

    gc.collect()
    train_X, train_y = X, y
    #'''
    if develop:
        train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.05, random_state=100)

    model = FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0, D=sparse_merge.shape[1], iters=50, inv_link="identity", threads=1)

    model.fit(train_X, train_y)
    print('[{}] Train FTRL completed'.format(time.time() - start_time))
    if develop:
        preds = model.predict(X=valid_X)
        print("FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))

    predsF = model.predict(X_test)
    submission['price_FTRL'] = predsF
    #print(rmsle(np.expm1(predsF),y_valid))
    #'''
    print('[{}] Predict FTRL completed'.format(time.time() - start_time))
    model = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, D=sparse_merge.shape[1], alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
                    D_fm=200, e_noise=0.0001, iters=17, inv_link="identity", threads=4)

    model.fit(train_X, train_y)
    print('[{}] Train ridge v2 completed'.format(time.time() - start_time))
    if develop:
        preds = model.predict(X=valid_X)
        print("FM_FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))

    predsFM = model.predict(X_test)
    print('[{}] Predict FM_FTRL completed'.format(time.time() - start_time))
    submission['price_FM_FTRL'] = predsFM

if __name__ == '__main__':
    test = pd.read_table('../input/mercari-price-suggestion-challenge/test.tsv', engine='c')
    submission: pd.DataFrame = test[['test_id']]
    del test
    gc.collect()
    get_pred_nn(submission)
    get_pred_ftrl(submission)
    submission["price"] = submission[["price_nn","price_FTRL","price_FM_FTRL"]].apply(lambda x:x["price_nn"]*0.488+x["price_FM_FTRL"]*0.616-x["price_FTRL"]*0.101,axis=1)
    submission["price"] = np.expm1(submission["price"])
    submission[["test_id","price"]].to_csv("submission.csv",index=False)
