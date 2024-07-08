import math
import sys


class SplitHashContainer:

    def __init__(self, n_splits: int = 1):
        self.n_splits: int = n_splits
        #self.split_size: int = int(math.ceil(2 ** 64 / n_splits))
        self.splits: list = [{} for _ in range(self.n_splits)]
        self.system_max_size = sys.maxsize
        # ASSERT 

    def insert(self, hash_: int, value: int):
        split = self.splits[(hash_%self.n_splits)]
        split[str(hash_)] = value

    def check(self, hash_: int):
        split = self.splits[(hash_%self.n_splits)]
        return str(hash_) in split

    def get_split(self, n: int):
        return self.splits[n]

    def get_system_max_size(self):
        return self.system_max_size