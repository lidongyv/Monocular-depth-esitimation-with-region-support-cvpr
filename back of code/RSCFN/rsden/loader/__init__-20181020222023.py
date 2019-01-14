import json

from rsden.loader.SceneFlow import SceneFlow
from rsden.loader.NYU1 import NYU1
from rsden.loader.NYU2 import NYU2
from rsden.loader.NYU import NYU
from rsden.loader.KITTI import KITTI
def get_loader(name):
    """get_loader

    :param name:
    """
    print(name)
    return {
        'sceneflow': SceneFlow,
        'nyu1':NYU1,
        'nyu2':NYU2,
        'nyu':NYU,
        'kitti':KITTI,
    }[name]


def get_data_path(name, config_file='config.json'):
    """get_data_path

    :param name:
    :param config_file:
    """
    data = json.load(open(config_file))
    return data[name]['data_path']
