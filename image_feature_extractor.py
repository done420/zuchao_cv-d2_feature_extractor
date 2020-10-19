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
import urllib.request

model_vgg19 = models.vgg19(pretrained=True)


def resize_to_square(image, size):
    h, w, d = image.shape
    ratio = size / max(h, w)
    resized_image = cv2.resize(image, (int(w*ratio), int(h*ratio)), cv2.INTER_AREA)
    return resized_image

def pad(image, min_height, min_width):
    h,w,d = image.shape

    if h < min_height:
        h_pad_top = int((min_height - h) / 2.0)
        h_pad_bottom = min_height - h - h_pad_top
    else:
        h_pad_top = 0
        h_pad_bottom = 0

    if w < min_width:
        w_pad_left = int((min_width - w) / 2.0)
        w_pad_right = min_width - w - w_pad_left
    else:
        w_pad_left = 0
        w_pad_right = 0

    return cv2.copyMakeBorder(image, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))


def image_extractor(image_name, cv2_mat):
    img_size = 224
    img_to_tensor = transforms.ToTensor()

    use_gpu = False

    model = model_vgg19
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    if use_gpu:
        model = model.cuda()
        model.eval()

    # src = cv2.imread(file_path)
    # name = os.path.basename(file_path)

    src = cv2_mat
    name = image_name

    try:
        # img_resized = cv2.resize(src, (img_size, img_size))
        img_resized = pad(resize_to_square(src, img_size), img_size, img_size)

        img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        img_tensor = img_to_tensor(img).unsqueeze(0)

        if use_gpu:
            img_tensor = img_tensor.cuda()

        output = Variable(img_tensor.float(), requires_grad=False)

        feature = model(output).cpu()
        feature = feature.data.numpy()

        feat = feature[0]
        norm_feat = feat / LA.norm(feat)

        # print("image: {}, roi_dict: {} \n feature.shape:{}".format(name,roi_dict,feat.shape))

    except:
        print("check cv2_mat data!")
        name, norm_feat = None, None


    return name, norm_feat




def roi_extractor(image_name,cv2_mat,top_left_x,top_left_y,roi_w, roi_h):
    img_size = 224
    img_to_tensor = transforms.ToTensor()
    use_gpu = False

    model = model_vgg19
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    if use_gpu:
        model = model.cuda()
        model.eval()

    x1 = top_left_x
    y1 = top_left_y
    w = roi_w
    h = roi_h

    x2 = x1 + w
    y2 = y1 + h

    roi = cv2_mat[y1:y2,x1:x2]
    # roi_resized = cv2.resize(cv2_mat, (img_size, img_size))
    roi_resized = pad(resize_to_square(roi, img_size), img_size, img_size)

    roi_mat = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    img_tensor = img_to_tensor(roi_mat).unsqueeze(0)

    if use_gpu:
        img_tensor = img_tensor.cuda()

    output = Variable(img_tensor.float(), requires_grad=False)

    feature = model(output).cpu()
    feature = feature.data.numpy()

    feat = feature[0]
    roi_feat = feat / LA.norm(feat)

    return image_name, roi_feat




def localPath_Extractor(imageName, imagePath):##### 从本地读取图像,提取特征
    try:
        image = cv2.imread(imagePath)
        cv2_mat = image
        name, image_feat = image_extractor(imageName, cv2_mat)

    except:
        print("please check imagePath:{}".format(imagePath))
        name, image_feat = None, None

    return imageName, image_feat



def urlPath_Extractor(imageName, urlPath):#### 网络url读取图像，提取特征
    #resp = urllib.request.urlopen(urlPath)
    try:
        resp = urllib.request.urlopen(urlPath)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        cv2_mat = image

        name, image_feat = image_extractor(imageName, cv2_mat)

    except IOError:
        print("can not open url to cv2_mat:{}".format(urlPath))
        name, image_feat = None, None

    return imageName, image_feat

def appFlow_Extractor(imageName, encodeFlow): ##### app数据流，提取特征

    try:
        imgArry = np.asarray(bytearray(encodeFlow), dtype="uint8")
        cv2_mat = cv2.imdecode(imgArry, cv2.IMREAD_COLOR)
        name, image_feat = image_extractor(imageName, cv2_mat)

    except IOError:
        print("data transform rrror")
        name, image_feat = None, None

    return imageName, image_feat



def test_roi_extractor():

    test_img = "/home/user/tmp/pycharm_project_310/1_detectron2/ImageDetectionAPI/demo.jpg"
    src = cv2.imread(test_img)
    name = os.path.basename(test_img)
    x = 247
    y = 169
    w = 114
    h = 151
    feat = roi_extractor(name,src,x,y,w,h)
    print(feat)







if __name__=="__main__":

    #### 本地读取图片测试：
    imageName = "test"
    imagePath = "/home/user/tmp/pycharm_project_310/1_detectron2/ImageDetectionAPI/demo.jpg"
    name, feat = localPath_Extractor(imageName, imagePath)
    print(feat)

    # #### url图像测试：
    # imageName = "test"
    # imagePath = 'http://www.pyimagesearch.com/wp-content/uploads/2015/01/opencv_logo.png'
    # name, feat = urlPath_Extractor(imageName, imagePath)
    # print(feat)


    # #### app数据流测试:
    # imageName = "test"
    # test_img = "/home/user/tmp/pycharm_project_310/1_detectron2/ImageDetectionAPI/demo.jpg"
    #
    # with open(test_img,'rb') as f:
    #     app_data = f.read()
    #
    # name, feat = appFlow_Extractor(imageName, app_data)
    # print(feat)
