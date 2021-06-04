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
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

## 학습을 진행할지 아니면 저장된 데이터를 사용할지 여부
Is_NewTaining = True
#Is_NewTaining = False

## 그램 Path
#file_directory = "D:\\news_scrap\\202105101818\\"

## 대영 Path
file_directory = "C:\\Users\\cydenza\\news_scrap\\202105311857\\"

category_2_dict = {}
category_index = 0
category_list = []

file_index = 1

sentences = []

testX = "남양주시(시장 조광한)는 31일 코로나19로부터 안전한 음식점 100곳을 발굴,  남양주 안심식당 으로 선정했다고 밝혔다. " \
    "사진제공=남양주시남양주시(시장 조광한)는 31일 코로나19로부터 안전한 음식점 100곳을 발굴,  남양주 안심식당 으로 선정했다고 밝혔다." \
    "안심식당 이란 코로나19 확산에 따라 외식 기피 현상이 심각한 상황에서 방역 및 위생에 대한 관심을 높이고 안전한 음식문화를 정착시켜 시민들이" \
    "안심하고 식당을 이용할 수 있도록 인증하는 제도이다.남양주시는 지난해부터 방역수칙을 철저히 준수하는 업소를 발굴해  남양주 안심식당 으로 지정" \
    "관리해 오고 있으며, 안심식당으로 지정된 식당에  남양주 안심식당  표지판을 부여하고 있다.안심식당 지정조건은 종사자 마스크 착용, 손소독제 비치 " \
    "및 주기적 소독 환기, 음식 덜어먹기 적극 실천(떠먹는 국자, 개인접시 제공 등), 위생적인 수저관리(개별포장, 개인수저 사전 비치 등), 남은 음식 재사용 안하기" \
    "5가지 항목으로, 안심식당 지정을 희망하는 업소에서 신청서를 제출하면 현장조사 및 평가절차를 거쳐 자격조건을 충족한 경우 안심식당으로 지정된다. 시 관계자는" \
    "앞으로도 방역수칙 및 위생을 철저히 준수하는 업소를 지속적으로 발굴해 안심식당으로 지정할 예정이며, 안심식당으로 지정된 업소는 남양주시 홈페이지 및 카카오맵," \
    "티맵, 네이버 등 포털사이트를 활용하여 적극 홍보할 예정 이라고 밝혔다. 머니s 주요뉴스           남양주=김동우 기자 bosun1997@mt.co.kr 저작권자 " \
    "'성공을 꿈꾸는 사람들의 경제 뉴스' 머니s, 무단전재 및 재배포 금지"
testY = "취업/창업"

"""
### GPU 사용 확인 코드
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print(tf.reduce_sum(tf.random.normal([1000, 1000])))
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
quit()
"""

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

tokenizer = Tokenizer()
model = Sequential()

if Is_NewTaining is True:
    # 파일에서 학습 데이터를 가져온다.
    for i in range(2000):
        file = file_directory + str(file_index) + ".txt"

        if not os.path.isfile(file):
            print("### FILE NOT FOUND!", file)
            break

        ReadTextFileToSentence(file)

        file_index = file_index + 1

    # print(text_list)
    print(sentences)

    tokenizer.fit_on_texts(sentences)
    vocab_size = len(tokenizer.word_index) + 1
    print("# vocab_size : ", vocab_size)
    print("# word_count : ", tokenizer.word_counts)
    print("### encoded : \n")
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

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

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
    x_len = len(X_train[0])
    print("# X-train Len : ", x_len)
    print("#### X-train 2 : \n", X_train)

    """
    model = Sequential([
        tf.keras.layers.Embedding(vocab_size, 100, input_length=x_len),
        tf.keras.layers.LSTM(units=50),
        tf.keras.layers.Dense(category_index, activation='softmax')
    ])
    """
    model.add(Embedding(vocab_size, 100, input_length=x_len))
    model.add(LSTM(units=50))
    model.add(Dense(category_index, activation='softmax'))

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
    mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.07), metrics=['accuracy'])
    model.summary()

    print(type(X_train))
    print(type(Y_train))

    history = model.fit(X_train, Y_train, batch_size=128, epochs=3, validation_split=0.25, callbacks=[es, mc])

    loaded_model = load_model('best_model.h5')
    print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_train, Y_train)[1]))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], 'b-', label="loss")
    plt.plot(history.history['val_loss'], 'r--', label='val_loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], 'g-', label="accuracy")
    plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylim(0, 1)
    plt.legend()

    plt.show()
else:
    print("aaa")

# 테스트 데이터 평가
testY = GetCategoryIndex(testY)
testY = np.array(testY)
print("# TestY : ", testY)
testX = tokenizer.texts_to_sequences(testX)
print("# TestX : ", testX)
testX = pad_sequences(testX, padding='post')
testX = np.array(testX)
print(testX)
print("# TestX : ", testX)

testX = np.expand_dims(testX,0)
print(testX.shape)
predict = model.predict(testX)
print("# predict max : ", np.argmax(predict[0]))
print("# testY ", testY)
#model.evaluate(testX, testY)
print("$$$$$$$$ PREDICT : ", predict)

