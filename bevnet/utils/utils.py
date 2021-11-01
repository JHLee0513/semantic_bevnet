import tabulate
import os
from collections import defaultdict
import numpy as np
import time

class Timer(object):
    def __init__(self):
        self.start_times = dict()
        self.total_times = defaultdict(np.float32)

    def start(self, name):
        self.start_times[name] = time.time()

    def pause(self, name):
        self.total_times[name] += time.time() - self.start_times[name]
        return self.total_times[name]

    def stop(self, name):
        t = self.pause(name)
        self.total_times[name] = 0.0
        return t

    def print(self):
        for k,t in self.total_times.items():
            print('{}: {}'.format(k, t)) 


def pprint_dict(x, fmt='simple'):
    """
    :param x: a dict
    :return: a string of pretty representation of the dict
    """
    def helper(d):
        ret = {}
        for k, v in d.items():
            if isinstance(v, dict):
                ret[k] = helper(v)
            else:
                ret[k] = v
        return tabulate.tabulate(ret.items(), tablefmt=fmt)
    return helper(x)


def get_data_dir():
    return os.path.normpath(os.path.join(os.path.dirname(__file__) + '/../data'))
