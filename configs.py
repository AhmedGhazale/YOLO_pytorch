
# dataset parameters

CLASSES = ['aeroplane', #0
           'bicycle',   #1
           'bird',      #2
           'boat',      #3
           'bottle',    #4
           'bus',       #5
           'car',       #6
           'cat',       #7
           'chair',     #8
           'cow',       #9
           'diningtable',#10
           'dog',       #11
           'horse',     #12
           'motorbike', #13
           'person',    #14
           'pottedplant',#15
           'sheep',     #16
           'sofa',      #17
           'train',     #18
           'tvmonitor'] #19
IMAGE_SIZE = 448
DATASET_PATH = '/home/ahmed/PycharmProjects/yolo1/voc/VOCdevkit/'

# model parameters

GRID_SIZE = 14
BASE_MODEL = 'mobilenet_v2'
BOXES_PER_CELL = 2

# training parameters

BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = .001

# predict parameters
MODEL_PATH = 'models/mobilenet_v2.pth'

