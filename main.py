# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os.path
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns

file_directory = "D:\\news_scrap\\202105121459\\"

category_2_dict = {}
category_index = 0
category_list = []

file_index = 1

sentences = []


def GetCategoryIndex(cate):
    global category_index
    catidx = category_2_dict.get(cate)
    print("## catidx : ", catidx)
    if catidx is None:
        category_2_dict[cate] = category_index
        catidx = category_index
        category_index = category_index + 1
    return catidx


def ReadTextFileToSentence(file):
    with open(file, 'r', encoding='UTF-8') as f:
        category_1 = f.readline()
        category_1 = category_1.strip()
        category_2 = f.readline()
        category_2 = category_2.strip()
        print(category_1)
        print(category_2)
        text = f.read()
        print(text)

        catidx = GetCategoryIndex(category_2)
        category_list.append(catidx)
        sentences.append(text.split(' '))


for i in range(10000):
    file = file_directory + str(file_index) + ".txt"

    if not os.path.isfile(file):
        print("### FILE NOT FOUND!", file)
        break

    ReadTextFileToSentence(file)

    file_index = file_index + 1

# print(text_list)
print(sentences)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

print("##########")
print(tokenizer.word_index)
print("##########")
print(tokenizer.word_counts)
print("##########")
encoded = tokenizer.texts_to_sequences(sentences)
print(encoded)
print("##########")

max_len = max(len(item) for item in encoded)
print(max_len)  # 최대 단어 길이

print("########## padded : \n")
padded = pad_sequences(encoded, padding='post', maxlen=max_len)
print(padded)

dataList = []
cateList = []

for index, category in enumerate(category_list):
    #dataList.append([padded[index], category])
    dataList.append(padded[index])
    cateList.append(category)
print("########## datalist : \n")
print(dataList)
print("########## cateList : \n")
print(cateList)

#dataF = pd.DataFrame(dataList, columns=['data', 'category'])
#print(dataF)

#X_train = dataF.data
#Y_train = dataF.category
#X_train = np.array(padded) # dataList
X_train = padded
Y_train = cateList

print("### X_train type 1 : ", type(X_train))
print("### Y_train type 1 : ", type(Y_train))

print("######### X-Train #############\n", X_train)
print("######### Y-Train #############\n", Y_train)

print('훈련용 뉴스 기사 갯수 : {}'.format(len(X_train)))
print('훈련용 카테고리 갯수 : {}'.format(len(Y_train)))
#print('테스트용 뉴스 기사 : {}'.format(len(X_test)))

"""
# 위에서 맥스 길이를 맞춰서 패딩을 했기 때문에 아레는 의미없다.
print('뉴스 기사의 최대 길이 :{}'.format(max(len(l) for l in X_train)))
print('뉴스 기사의 평균 길이 :{}'.format(sum(map(len, X_train))/len(X_train)))

plt.hist([len(s) for s in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
"""

"""
# 기사가 속한 카테고리 값을 본다.
fig, axe = plt.subplots(ncols=1)
fig.set_size_inches(12,5)
sns.countplot(Y_train)
"""

unique_elements, counts_elements = np.unique(Y_train, return_counts=True)
print("각 레이블에 대한 빈도수:")
print(np.asarray((unique_elements, counts_elements)))

print("### Y_Train ###\n", Y_train)
Y_train = to_categorical(Y_train)
print("### Y_train ###\n", Y_train)
print("### Y_train type : ", type(Y_train))

catlen = len(category_list)
print("### catlen : ", catlen)

"""
model = Sequential()
model.add(Embedding(1000, 120))
model.add(LSTM(120))
model.add(Dense(catlen, activation='softmax'))
"""

#X_train = X_train.tolist()
print("# X-train Len : ", len(X_train[0]))
print("#### X-train 2 : \n", X_train)

model = Sequential([
    tf.keras.layers.Embedding(len(X_train[0]), 1),    #, input_length=ndim),
    tf.keras.layers.LSTM(units=50),
    tf.keras.layers.Dense(len(Y_train), activation='softmax')
])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

print(type(X_train))
print(type(Y_train))

history = model.fit(X_train, Y_train, batch_size=128, epochs=5, callbacks=[es, mc])
