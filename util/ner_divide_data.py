# -*- coding: utf-8 -*-

import os


def ner_divide_data(name, data_file_path, lower=True):
    data_file = open(data_file_path, "r", encoding="utf8")

    if not os.path.exists("dataset/" + name):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs("dataset/" + name)
        os.makedirs("dataset/" + name + "/ner/")
        os.makedirs("dataset/" + name + "/classify/")
    train_file = open("dataset/" + name + "/ner/train.dat", "w", encoding="utf8")
    dev_file = open("dataset/" + name + "/ner/dev.dat", "w", encoding="utf8")
    test_file = open("dataset/" + name + "/ner/test.dat", "w", encoding="utf8")
    train_test_file = open("dataset/" + name + "/ner/train_test.dat", "w", encoding="utf8")
    lines = data_file.readlines()
    num = len(lines)
    print("Get " + str(num) + " lines in the data file.")

    file = train_file
    # print('Writing in train_file...')
    for i, line in enumerate(lines):
        if lower:
            line = line.lower()
        # train: dev: test: train_test = 80: 8: 8: 4
        file.write(line)
        if line == '\n':
            if file == train_file and i > num * 0.8:
                file.write('-DOCSTART- -X- O O')
                file = dev_file
                # print('Writing in dev_file...')
            elif file == dev_file and i > num * 0.88:
                file.write('-DOCSTART- -X- O O')
                file = test_file
                # print('Writing in test_file...')
            elif file == test_file and i > num * 0.96:
                file.write('-DOCSTART- -X- O O')
                file = train_test_file
                # print('Writing in train_test_file...')

    data_file.close()
    train_file.close()
    dev_file.close()
    test_file.close()
    train_file.close()

    print("The NER dataset have been already generated.")


if __name__ == '__main__':
    ner_divide_data(name="aminer", data_file_path="dataset/aminer/ner.txt", lower=True)
