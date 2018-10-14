# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-05-11 17:01:00
# @Last Modified by:   yulidong
# @Last Modified time: 2018-05-11 17:11:44
import zipfile
out_dir='/home/lidong/Documents/datasets/single_driver/images/'
def un_zip(file_name):
    zip_file=zipfile.ZipFile(file_name)
    if os.path.isdir(os.path.join(out_dir,file_name)):
        pass
    else:
        os.mkdir(os.path.join(out_dir,file_name))
    for names in zip_file.namelist():
        

