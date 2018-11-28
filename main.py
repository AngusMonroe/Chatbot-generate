# -*- coding: utf-8 -*-

"""
main.py
生成bot的主函数，训练NER及classifier的入口
"""

import os
import numpy as np
from util.ner_divide_data import ner_divide_data
from util.classify_divide_data import classify_divide_data


def get_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >gpu_info')  # 查看GPU memory,并将结果保存在gpu_info中
    memory_gpu = [int(x.split()[2]) for x in open('gpu_info', 'r').readlines()]  # 读取gpu memory
    os.system('rm gpu_info')  # 删除gpu_info
    return str(np.argmax(memory_gpu))  # 返回剩余memory最多的显卡号


def prepare_datset(bot_id, ner_file_path, classify_file_path):
    ner_divide_data(bot_id, ner_file_path, lower=True)
    class_num = classify_divide_data(bot_id, classify_file_path, lower=True)
    return class_num


def train_ner(name, gpu_num):
    cmd = 'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=' + gpu_num + ' nohup python3 -u train_ner.py' \
                                              ' --train ' + 'dataset/' + name + '/ner/train.dat' \
                                              ' --dev ' + 'dataset/' + name + '/ner/dev.dat' \
                                              ' --test ' + 'dataset/' + name + '/ner/test.dat' \
                                              ' --test_train ' + 'dataset/' + name + '/ner/train_test.dat' \
                                              ' --name ' + name + ' > ' + 'log/' + name + '_ner.log 2>&1  &'
    print(cmd)
    log_file = open('log/cmd.log', 'a', encoding='utf8')
    log_file.write(cmd + '\n')
    status = os.system(cmd)
    return status


def train_classify(name, gpu_num, class_num):
    cmd = 'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=' + gpu_num + ' nohup python3 train_classify.py' \
                                              ' --train ' + 'dataset/' + name + '/classify/train.dat' \
                                              ' --test ' + 'dataset/' + name + '/classify/test.dat' \
                                              ' --classes ' + str(class_num) + \
                                              ' --name ' + name + ' > ' + 'log/' + name + '_classify.log 2>&1  &'
    print(cmd)
    log_file = open('log/cmd.log', 'a', encoding='utf8')
    log_file.write(cmd + '\n')
    status = os.system(cmd)
    return status


def generate_chatbot(bot_id, class_num):
    gpu_num = get_gpu()
    ner_status = train_ner(name=bot_id, gpu_num=gpu_num)
    if ner_status:
        print('NER training error!')
    classify_status = train_classify(name=bot_id, gpu_num=gpu_num, class_num=class_num)
    if classify_status:
        print('Classify training error!')


def main(bot_id, ner_file_path, classify_file_path):
    class_num = prepare_datset(bot_id, ner_file_path, classify_file_path)
    generate_chatbot(bot_id, class_num)


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 4:
        print("Usage: {} bot_id ner_file_path classify_file_path".format(sys.argv[0]))
        exit(-1)
    bot_id, ner_file_path, classify_file_path = sys.argv[1:]
    main(bot_id=bot_id, ner_file_path=ner_file_path, classify_file_path=classify_file_path)
