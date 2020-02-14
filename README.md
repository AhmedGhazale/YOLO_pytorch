# YOLO in pytorch

### introduction:
this is an implementation of YOLO object detection algorithme.
the implementation is a mix between YOLO v1 and v2.
I used pytorch models trained on ImageNet as backbone model then added final conv layer for the output  **THERE IS NO FULLY CONNECTED LAYERS**   
**NOTE** I used YOLO loss function implementation from [this repo](https://github.com/xiongzihua/pytorch-YOLO-v1)
### Dependencies:
* torch==1.3.1
* torchvision==0.4.2
* opencv-python==4.1.2.30
* albumentations==0.4.3
* numba==0.46.0
### installation:
clone and install the requirements
```bash
git clone https://github.com/AhmedGhazale/cifar100-classifier.git
cd cifar100-classifier
pip3 install -r requirements.txt
```
### Demo :
**to run an image**
```bash
python3 predict.py path/to/image
```
**to run video**
```bash
python3 video_demo.py path/to/video
```
the output will be a video named output.avi in the same directory

### Traininig:
* download voc dataset
* edit the dataset path in **config.py** 
* run 
```bash
python3 train.py
```
### evaluation:
**TODO**


