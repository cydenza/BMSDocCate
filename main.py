# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os.path
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns

file_directory = "D:\\news_scrap\\202105161627\\"
category_1 = ""
category_2 = ""

file_index = 1

category_list = []
sentences = []

for i in range(10000):
    file = file_directory + str(file_index) + ".txt"

    if not os.path.isfile(file):
        print("### FILE NOT FOUND!", file)
        break

    with open(file, 'r', encoding='UTF-8') as f:
        category_1 = f.readline()
        category_1 = category_1.strip()
        category_2 = f.readline()
        category_2 = category_2.strip()
        print(category_1)
        print(category_2)
        text = f.read()
        print(text)

        category_list.append(category_2)
        sentences.append(text.split(' '))

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

padded = pad_sequences(encoded, padding='post', maxlen=max_len)
print(padded)

print("##########")

dataList = []

for index, category in enumerate(category_list):
    dataList.append([padded[index], category])
print(dataList)

dataF = pd.DataFrame(dataList, columns=['data', 'category'])
print(dataF)

X_train = dataF.data
Y_train = dataF.category

print("######### X-Train #############\n", X_train)
print("######### Y-Train #############\n", X_train)

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

