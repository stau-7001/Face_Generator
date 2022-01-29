from unicodedata import name
import cv2
import argparse
import os
import json
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from nbformat import read
from align_face_FFHQ import align_face 
import face_recognition
from PIL import Image
import numpy as np
import string, random
from tqdm import tqdm
from PIL import Image
from align_face_FFHQ import align_face
from skimage.metrics import structural_similarity as ssim
from insightface_func.face_detect_crop_single import *
from id_generator import get_id

Max_int = 40
Min_int = 5

def getParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--res', type=int, default=512,
                                            help="expected image resolution")
    parser.add_argument('-t', '--tol', type=int, default=700,
                                            help="tolerance range for the size of the captured face image")
    parser.add_argument('-m', '--mode', type=int, default=0,
                                            help="0:VGG crop, 1:FFHQ crop")
    parser.add_argument('-i', '--interval', type=int, default=5,
                                            help="number of frames interval")
    parser.add_argument('-s', '--skip', type=int, default=60,
                                            help="number of frames to be skipped")
    parser.add_argument('-b', '--blur', type=int, default=10,
                                            help="number of frames to be skipped")
    return parser.parse_args()

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
 
def res_checker(face_locations = None, res=512, tol = 400):
    top, right, bottom, left = face_locations[0]
    if abs(top-bottom)-abs(right-left) in range(-50, 51):
        # print(face_locations[0])
        if abs(top-bottom) > res-tol and abs(right-left) > res-tol:
            return 1
    return 0
    

def frame_generator(Vid,filename, opt = None):
    cap = cv2.VideoCapture('./video/'+filename)
    if opt.mode !=1:
        app = Face_detect_crop(name='antelope', root='./insightface_func/models')
        app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode=None)

    frame_id = -1
    discard = 0
    interval = 0
    gray_lap = 0
    flag = False
    pre_img = None
    while cap.isOpened():
        frame_id = frame_id +1
        ret, frame = cap.read()
        if discard > 0 or interval > 0:
            discard = discard - 1
            interval = interval -1
            continue
        if ret==True:
            cv2.imwrite("temp.png", frame)
            raw_image = face_recognition.load_image_file("temp.png")
            face_locations = face_recognition.face_locations(raw_image)
            if len(face_locations) > 0 and res_checker(face_locations, opt.res, opt.tol):
                if opt.mode == 1:
                    pil_image = align_face("temp.png", opt.res)
                else:
                    img_a_whole = cv2.imread("temp.png")
                    cv_images, _ = app.get(img_a_whole,opt.res)
                    pil_image = Image.fromarray(cv2.cvtColor(cv_images[0],cv2.COLOR_BGR2RGB))

                img_array = np.asarray(pil_image)
                if np.all(pre_img != None):
                        ss=ssim(pre_img,img_array,multichannel=True)
                        print(ss)
                        if ss > 0.6:
                            opt.interval = min(opt.interval*2, Max_int)
                        else:
                            opt.interval = Min_int
                pre_img = img_array
                interval = opt.interval
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                gray_lap = cv2.Laplacian(img_gray,cv2.CV_64F).var()
                if gray_lap >= opt.blur:
                        img_id = get_id(pil_image)
                        f_path ="./dataset/"+str(Vid).zfill(4)+"_"+str(frame_id).zfill(6)+".png"
                        pil_image.save(fp=f_path)
                        write_dict = {
                            f_path:img_id
                            }
                        with open("./id_record.json","a") as f:
                            if Vid != 0 or flag: 
                                f.write(',')
                            json.dump(write_dict,f,indent=1)
                            flag = True
                            
            else:
                discard = opt.skip
        else:
            break
    cap.release()
   
        
if __name__ == '__main__':
    config = getParameters()
    test_list=[]
    with open("./id_record.json","a") as f:
        f.write('[')
    with open("List_of_testing_videos.txt", "r") as f:
        test_list= f.readlines()
    id=0
    li=len(test_list)
    for i in tqdm(range(0,li)):
        print("\n Starting processing video", test_list[i])
        # print(i)
        frame_generator(id, test_list[i], config)
        id=id+1
    with open("./id_record.json","a") as f:
        f.write(']')
    #print(test_list)
        