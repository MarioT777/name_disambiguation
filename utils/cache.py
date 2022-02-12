import os
from os.path import join
import lmdb
from utils import data_utils
from utils import settings

map_size = 107374182400


class LMDBClient(object):
    def __init__(self, db_name):
        lmdb_dir = join(settings.DATA_DIR, 'lmdb')
        os.makedirs(lmdb_dir, exist_ok=True)
        self.db = lmdb.open(join(lmdb_dir, db_name), map_size=map_size)

    def get(self, key):
        with self.db.begin() as txn:  # 建立事物
            value = txn.get(key.encode())
        if value:
            return data_utils.deserialize_embedding(value)
        else:
            return None

    def get_batch(self, keys):
        values = []
        with self.db.begin() as txn:
            for key in keys:
                value = txn.get(key.encode())
                if value:
                    values.append(data_utils.deserialize_embedding(value))
        return values

    def set(self, key, vector):
        # 修改数据库内容
        with self.db.begin(write=True) as txn:
            txn.put(str(key).encode("utf-8"), data_utils.serialize_embedding(vector))

    def set_batch(self, generator):
        with self.db.begin(write=True) as txn:
            for key, vector in generator:
                txn.put(key.encode("utf-8"), data_utils.serialize_embedding(vector))
                print(key, self.get(key))
