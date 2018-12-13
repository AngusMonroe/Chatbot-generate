from pymongo import MongoClient
coll = None

def load_conf(conf_file):
    fp = open(conf_file, 'r', encoding='utf8')
    lines = fp.readlines()
    address = lines[1].strip()  # 加载db地址
    port = lines[3].strip()  # 加载连接端口
    user = lines[5].strip()  # 加载登录用户
    password = lines[7].strip()  # 加载密码
    collection_name = lines[9].strip()  # 加载登录用户
    fp.close()
    return address, port, user, password, collection_name


def load_db(conf_file):
    global coll
    if coll:
        return coll
    address, port, user, password, collection_name = load_conf(conf_file)
    conn = MongoClient(address, int(port))
    db_auth = conn.admin
    db_auth.authenticate(user, password)  # 用户认证
    db = conn.bigsci  # 连接库
    coll = db[collection_name]
    return coll


def write_db(collection, data):
    collection.insert_one(data)


if __name__ == '__main__':
    # address, port, user, password = load_conf('../dataset/mongo.txt')
    coll = load_db('../dataset/mongo.txt')
    write_db(coll, {'data': 'test'})
