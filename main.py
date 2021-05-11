# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os.path
import nltk
from nltk.corpus import stopwords

file_directory = "D:\\news_scrap\\202105111507\\"
category_1 = ""
category_2 = ""

file_index = 1

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

    file_index = file_index + 1
