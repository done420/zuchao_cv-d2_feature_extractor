#coding:utf-8
import time
import os,cv2,glob,h5py,time
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from numpy import linalg as LA
import torchvision.models as models
from torch.autograd import Variable
from annoy import AnnoyIndex
from image_feature_extractor import image_extractor as extractor

model_vgg19 = models.vgg19(pretrained=True)
model = model_vgg19

use_gpu = False

for param in model.parameters():
    param.requires_grad = False
    model.eval()

if use_gpu:
    model = model.cuda()
    model.eval()



def load_images(images_root):
    img_type = [".jpg",".png",".jpeg",'.bmp','.tif']

    db_imgs = {}

    for category in os.listdir(images_root):
        img_dir = os.path.join(images_root,category)
        if os.path.isdir(img_dir):

            for file_name in os.listdir(img_dir):
                file_type = os.path.splitext(file_name)[-1]
                img_path = os.path.join(img_dir,file_name)

                if file_type in img_type:
                    if file_name not in db_imgs:
                        db_imgs[file_name] = img_path

    return db_imgs



def creat_images_feat_db(imageList,extractor, db_file):

    if not os.path.exists(db_file):
        extract_start = time.time()

        print("***********  start image-feature extraction ...")
        names,feats = [],[]
        n = 0
        for img_path in imageList:
            if os.path.exists(img_path):
                n+=1
                print("image_feature_extract: {} , {}/{}".format(img_path, n , len(imageList)))
                src = cv2.imread(img_path)
                image_name = os.path.basename(img_path)
                name, feat = extractor(image_name, src)
                names.append(name)
                feats.append(feat)

        feats = np.array(feats)

        print("db: writing ....")
        h5f = h5py.File(db_file, 'w')
        h5f.create_dataset('dataset_1', data=feats)
        h5f.create_dataset('dataset_2', data=np.string_(names))
        h5f.close()

        extract_end = time.time()

        extract_cost = extract_end - extract_start

        if os.path.exists(db_file):
            print("features saved:\n{}".format(db_file))

        print("################  {}  image-feature extraction takes {}".format(len(imageList), extract_cost))

    else:
        print('feature_db exits: \n {}'.format(db_file))



# if __name__=="__main__":
#     images_root = "/data/dadi/houzz_5space"
#     imgs_dict = load_images(images_root)
#
#     test_db = '/data/1_qunosen/3_label_image_40caterory/1.pth'
#     imageList = [imgs_dict.get(ele) for ele in imgs_dict]
#
#     imageList = imageList[:3]
#     creat_images_feat_db(imageList,extractor, test_db)



