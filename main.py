# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os.path
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

file_directory = "D:\\news_scrap\\202105121459\\"
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

#print(text_list)
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
print(max_len)      # 최대 단어 길이

padded = pad_sequences(encoded, padding='post', maxlen=max_len)
print(padded)

print("##########")

dataList = []

for index, category in enumerate(category_list):
    dataList.append([padded[index], category])
print(dataList)

dataF = pd.DataFrame(dataList) #, padded) #, columns=['category', 'data'])
print(dataF)
