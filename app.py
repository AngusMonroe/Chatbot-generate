# -*- coding: utf-8 -*-

"""
app.py
用于启动服务，为model提供api
"""

import json
import logging
from service.ner import *
from service.classify import *
from util.mongo_writer import load_db, write_db
from flask import Flask
import datetime

logger = logging.getLogger('query')
logger.setLevel(logging.INFO)
fh = logging.FileHandler(
    filename='log/query.log',
    mode='a',
    encoding='utf-8',
    delay=False
)
fh.setFormatter(
    logging.Formatter(
        '[%(asctime)s]\t%(message)s'
    )
)
fh.setLevel(logging.INFO)
logger.addHandler(fh)
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'QA-API'


@app.route('/query/<name>/<text>')
def query(text, name):

    ret = {
        'query': text,
        'intent': classify(text, name),
        'entity': ner(text, name)
    }
    log_data = {
        'time': datetime.datetime.now(),
        'bot_name': name
    }

    coll = load_db('dataset/mongo.txt')
    write_db(coll, log_data.update(ret))

    result = json.dumps(ret, ensure_ascii=False)
    logger.info(result)
    return result


def main():
    app.run(host='0.0.0.0', port=55014, debug=False)

if __name__ == '__main__':
    main()
