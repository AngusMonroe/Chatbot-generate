# -*- coding: utf-8 -*-

import os


def classify_divide_data(name, data_file_path, lower=True):
    data_file = open(data_file_path, "r", encoding="utf8")
    intent = []

    if not os.path.exists("dataset/" + name):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs("dataset/" + name)
        os.makedirs("dataset/" + name + "/ner/")
        os.makedirs("dataset/" + name + "/classify/")
    train_file = open("dataset/" + name + "/classify/train.dat", "w", encoding="utf8")
    test_file = open("dataset/" + name + "/classify/test.dat", "w", encoding="utf8")

    lines = data_file.readlines()
    num = len(lines)
    print("Get " + str(num) + " lines in the data file.")

    file = train_file
    # print('Writing in train_file...')
    file.write('label\tbody\n')
    for i, line in enumerate(lines):
        if lower:
            line = line.lower()
        # train: test = 4: 1
        word = line.split('\t')[0]
        intent.append(word)
        file.write(line)
        if file == train_file and i > num * 0.8:
            file = test_file
            print('Writing in dev_file...')
            file.write('label\tbody\n')

    data_file.close()
    train_file.close()
    test_file.close()

    print("The classify dataset have been already generated.")
    print("Distinct intents:", len(set(intent)) - 1)

    return len(set(intent)) 


if __name__ == '__main__':
    classify_divide_data(name="test", data_file_path="../dataset/classify/intent.txt", lower=True)
