import tabulate
import os


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
