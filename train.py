import torch
from YOLO import YOLO
import configs as cfg
from DataLoader import get_train_data_loader
import time
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from YoloLoss import yoloLoss
import os

def train(model, dataloader, epochs):
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    loss_func = yoloLoss(cfg.GRID_SIZE,cfg.BOXES_PER_CELL,5,.5)
    save_path = 'models/'
    log_file = open(os.path.join(save_path, BASE_MODEL+'_log.txt'),'w')

    for epoch in range(epochs):
        epcoh_loss = 0
        i = 0
        for x,y in dataloader:

            x_batch = x.cuda()
            y_batch = y.cuda()

            optimizer.zero_grad()

            out = model(x_batch)
            optimizer.zero_grad()
            loss = loss_func(out, y_batch)
            loss.backward()
            optimizer.step()

            step_loss = loss.item()
            epcoh_loss+=step_loss
            print('epoch ',epoch,', step ',i,', with loss',  step_loss)
            i +=1

        torch.save(model, os.path.join(save_path,cfg.BASE_MODEL+'.pth'))
        log_file.write(str(epcoh_loss/i)+'\n')
        log_file.flush()
        print(epcoh_loss/i)

        
if __name__ == '__main__':
  
    model = YOLO(cfg.BASE_MODEL,cfg.GRID_SIZE, len(cfg.CLASSES), cfg.BOXES_PER_CELL).cuda()
    dataloader = get_train_data_loader(cfg.DATASET_PATH, cfg.CLASSES,cfg.BATCH_SIZE)

    model.train()

	train(model,dataloader,cfg.EPOCHS)


