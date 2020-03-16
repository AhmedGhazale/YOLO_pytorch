import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from utils import *
import numba
import cv2
import configs as cfg
import sys
from predict import *
from tqdm import tqdm

def main():

    model = torch.load(cfg.MODEL_PATH)
    video_path =  sys.argv[1]
    video  = cv2.VideoCapture(video_path)

    ret, frame = video.read()
    out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame.shape[1],frame.shape[0]))
    frames_count = int(out.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frames_count)
    #while ret:
    for i in tqdm(range(1000)):
        result = predict(frame ,model)

        for left_up, right_bottom, class_name, prob in result:
            cv2.rectangle(frame, left_up, right_bottom, (124,32,225), 2)
            label = class_name + str(round(prob, 2))
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            p1 = (left_up[0], left_up[1] - text_size[1])
            cv2.rectangle(frame, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), (124,32,225),-1)
            cv2.putText(frame, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)

        
        out.write(frame)
        ret, frame = video.read()
    out.release()
if __name__ == '__main__':
	main()
    
    
