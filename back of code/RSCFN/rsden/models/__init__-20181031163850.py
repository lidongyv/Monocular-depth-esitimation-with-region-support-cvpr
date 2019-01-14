# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-03-18 15:24:33
# @Last Modified by:   yulidong
# @Last Modified time: 2018-07-31 17:35:05

import torchvision.models as models
from rsden.models.rsn_mask import *
from rsden.models.rsn_cluster import *
from rsden.models.rsn import *
from rsden.models.rsn_v2 import *
from rsden.models.drn import *
from rsden.models.rsdin import *
def get_model(name):
    model = _get_model_instance(name)

    model = model()

    return model

def _get_model_instance(name):
    try:
        return {
            'rsnet': rsn,
            'rsn_mask': rsn_mask,
            'rsn_cluster': rsn_cluster,
            'rsnet_v2':rsn_v2,
            'drnet':drn,
            'rsdin':rsdin,
        }[name]
    except:
        print('Model {} not available'.format(name))
