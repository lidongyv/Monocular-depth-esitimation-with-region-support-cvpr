# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-05-11 16:33:59
# @Last Modified by:   yulidong
# @Last Modified time: 2018-05-12 10:34:11
import os
import numpy as np
import zipfile
import shutil
def un_zip(file_name):
    zip_file=zipfile.ZipFile(file_name)
    if os.path.isdir(os.path.join(out_dir,file_name)):
        pass
    else:
        os.mkdir(os.path.join(out_dir,file_name))
    for names in zip_file.namelist():
        zip_file.extract(names,file_name)
    zip_file.close()
#train
# images=[]
# ground=[]
# i_o_dir='/home/dataset/datasets/kitti'
# g_o_dir='/home/lidong/Documents/datasets/single_driver/data_depth_annotated/train/'
# g_m_dir='proj_depth/groundtruth'
# i_m_dir='proj_depth/images'
# days=os.listdir(g_o_dir)
# for i in days:
#     zip_name=os.path.join(i_o_dir,i+'.zip')
#     if not os.path.isfile(zip_name):
#         continue
#     zip_file=zipfile.ZipFile(zip_name)
#     g_i_dir=os.listdir(os.path.join(g_o_dir,i,g_m_dir))
#     if os.path.exists(os.path.join(g_o_dir,i,i_m_dir)):
#         shutil.rmtree(os.path.join(g_o_dir,i,i_m_dir))
#         print(os.path.join(g_o_dir,i,i_m_dir))
#     os.mkdir(os.path.join(g_o_dir,i,i_m_dir))
#     for j in g_i_dir:
#         tground=os.listdir(os.path.join(g_o_dir,i,g_m_dir,j))
#         os.mkdir(os.path.join(g_o_dir,i,i_m_dir,j))
#         for m in tground:
#             ground.append(os.path.join(g_o_dir,i,g_m_dir,j,m))
#             for n in zip_file.namelist():
#                 if i in n and m in n and j in n and 'png' in n:
#                     image_path=os.path.join(g_o_dir,i,i_m_dir,j)                    
#                     zip_file.extract(n,image_path)
#                     images.append(os.path.join(image_path,n))
#                     print(os.path.join(image_path,n))
#                     print(len(images))
#     zip_file.close()
# print(len(ground))
# np.save('/home/lidong/Documents/RSDEN/RSDEN/kitti_ground.npy',ground)
# np.save('/home/lidong/Documents/RSDEN/RSDEN/kitti_images.npy',images)

#eval
image=[]
ground=[]
eval_dir='/home/lidong/Documents/datasets/single_driver/depth_selection/val_selection_cropped'
g_dir='groundtruth_depth'
i_dir='image'
g_f=os.listdir(os.path.join(eval_dir,g_dir))
i_f=os.listdir(os.path.join(eval_dir,i_dir))
#print(g_f)
for i in g_f:
    for j in i_f:
        split=i.split('_')
        if split[0] in j and split[1] in j and split[2] in j and split[4] in j and split[-1] in j and split[-3] in j:
            image.append(os.path.join(eval_dir,i_dir,j))
            ground.append(os.path.join(eval_dir,g_dir,i))
            print(os.path.join(eval_dir,i_dir,j))
            break
np.save('/home/lidong/Documents/RSDEN/RSDEN/kitti_ground.npy',ground)
np.save('/home/lidong/Documents/RSDEN/RSDEN/kitti_images.npy',image)


