from torch.utils.data import Dataset
import os
import xml.etree.cElementTree as ET
import configs as cfg
import collections
import cv2
from albumentations.pytorch import ToTensorV2
import torch
from utils import post_processing
import configs as cfg
import albumentations as  A
import numpy as np

class VOCDataset(Dataset):

        def __init__(self, root,classes_list, image_set='train', transforms=None, s = 14, b = 2, number_classes = 20, image_size = 448 ):

            annotations_path = os.path.join(root,'Annotations')
            images_path = os.path.join(root,'JPEGImages')
            splits_dir = os.path.join(root, 'ImageSets/Main')
            split_f = os.path.join(splits_dir, image_set + '.txt')

            with open(os.path.join(split_f), "r") as f:
                file_names = [x.strip() for x in f.readlines()]

            self.images = [os.path.join(images_path, x + ".jpg") for x in file_names]
            self.annotations = [os.path.join(annotations_path, x + ".xml") for x in file_names]
            self.transforms = transforms
            self.s = s
            self.b = b
            self.number_classes = number_classes
            self.image_size = image_size
            self.classes_list = classes_list

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):

            image = cv2.imread(self.images[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            annot = self.parse_voc_xml(ET.parse(self.annotations[idx]).getroot())

            bboxes =[]
            category_id =[]
            objs = annot['annotation']['object']
            if not isinstance(objs, list):
                objs = [objs]

            for obj in objs:
                b = obj['bndbox']
                x1 = int(b['xmin'])
                x2 = int(b['xmax'])
                y1 = int(b['ymin'])
                y2 = int(b['ymax'])
                label = cfg.CLASSES.index(obj['name'])
                bboxes.append([x1, y1, x2, y2])
                category_id.append(label)

            sample = {'image':image, 'bboxes':bboxes, 'category_id':category_id}

            if self.transforms:
                sample = self.transforms(**sample)

            label = self.get_label(sample['bboxes'], sample['category_id'])


            return sample['image'], label

        def parse_voc_xml(self, node):
            voc_dict = {}
            children = list(node)
            if children:
                def_dic = collections.defaultdict(list)
                for dc in map(self.parse_voc_xml, children):
                    for ind, v in dc.items():
                        def_dic[ind].append(v)
                voc_dict = {
                    node.tag:
                        {ind: v[0] if len(v) == 1 else v
                         for ind, v in def_dic.items()}
                }
            if node.text:
                text = node.text.strip()
                if not children:
                    voc_dict[node.tag] = text
            return voc_dict

        def to_xywh(self, xmin, ymin, xmax, ymax):

            w = xmax - xmin
            h = ymax - ymin

            cx = xmin + w // 2
            cy = ymin + h // 2

            return cx, cy, w, h

        def get_label(self, bboxes, labels):

            # reads the labels and makes one label example
            out = np.zeros((self.s, self.s, self.number_classes + self.b*5), dtype=np.float32)

            for i in range(len(bboxes)):
                idx = labels[i]

                xmin, ymin, xmax, ymax = [int(c) for c in bboxes[i]]

                x, y, w, h = self.to_xywh(xmin, ymin, xmax, ymax)

                cell_size = self.image_size // self.s

                # TRIAL
                cell_x = x // cell_size
                cell_y = y // cell_size

                x %= cell_size
                y %= cell_size

                x /= cell_size
                y /= cell_size
                w /=  self.image_size
                h /=  self.image_size
                for i in range(self.b):
                    out[cell_y, cell_x, 0 + 5*i] = x
                    out[cell_y, cell_x, 1 + 5*i] = y
                    out[cell_y, cell_x, 2 + 5*i] = w
                    out[cell_y, cell_x, 3 + 5*i] = h
                    out[cell_y, cell_x, 4 + 5*i] = 1

                out[cell_y, cell_x, idx + self.b*5] = 1

            # return one label of shape [grid size, gride size, classes+5]
            return out


def get_aug(aug, min_area=2., min_visibility=0.):
    return A.Compose(aug, A.BboxParams(format='pascal_voc', min_area=min_area, min_visibility=min_visibility, label_fields=['category_id']))




def get_train_data_loader(path,classes_list, batch_size, shuffle = True, num_workers = 4):

    aug = get_aug([
        A.HorizontalFlip(p=.5),
        A.RandomSizedBBoxSafeCrop(width=448, height=448, erosion_rate=0, interpolation=cv2.INTER_CUBIC),
        A.RGBShift(p=.5),
        A.Blur(blur_limit=5, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.CLAHE(p=0.5),
        A.Normalize(),
        ToTensorV2()
    ])

    voc = VOCDataset(path,classes_list= classes_list,image_set='train',transforms=aug)
    dataloader = torch.utils.data.DataLoader(voc,batch_size,shuffle=shuffle, num_workers=num_workers)
    return dataloader


def get_test_data_loader(path,classes_list, batch_size, shuffle = False, num_workers = 4):

    aug_test = get_aug([
        A.Resize(448, 448),
        A.Normalize(),
        ToTensorV2()
    ])

    voc = VOCDataset(path,classes_list=classes_list,image_set='val',transforms=aug_test)
    dataloader = torch.utils.data.DataLoader(voc,batch_size,shuffle=shuffle, num_workers=num_workers)
    return dataloader


def main():
    aug = get_aug([
        A.HorizontalFlip(p=.5),
        A.RandomSizedBBoxSafeCrop(width=448, height=448, erosion_rate=0, interpolation=cv2.INTER_CUBIC),
        A.RGBShift(p=.5),
        A.Blur(blur_limit=5, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.CLAHE(p=0.5),
    ])

    voc = VOCDataset(cfg.DATASET_PATH ,classes_list=cfg.CLASSES,image_set='train',transforms=aug)

    for i in range(1000):
        image,out = voc[i]

        boxes = post_processing(out)
        im_size = image.shape[0]
        for det in boxes:
            pt1 = (det[0]-det[2]/2  , det[1]-det[3]/2 )
            pt2 = (det[0]+det[2]/2, det[1]+det[3]/2)
            pt1 = (int(pt1[0] * im_size), int(pt1[1] * im_size ))
            pt2 = (int(pt2[0] * im_size), int(pt2[1] * im_size))

            cv2.rectangle(image,pt1, pt2, (255,33,44))

        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        cv2.imshow("test",image)
        cv2.waitKeyEx(-1)


if __name__==  '__main__':
    main()
