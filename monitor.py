# -*- coding: utf-8 -*-

"""
monitor.py
用于监控train_ner以及train_classify进程是否存在，如果不存在，则表明训练完成，启动app.py
"""

import os
import schedule
import sys


def work(name):
    cmd = 'ps ux|grep -e ' + name + '/ner -e ' + name + '/classify > log/monitor.log'
    os.system(cmd)
    monitor_log = open('log/monitor.log', 'r').readlines()
    os.system('rm log/monitor.log')
    if not monitor_log:
        os.system('python app.py')
    sys.exit(0)


def main(name):
    schedule.every(1).minutes.do(work(name=name))


if __name__ == '__main__':
    main('test')
