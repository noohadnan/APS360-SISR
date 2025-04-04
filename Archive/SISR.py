# PyTorch model and training necessities
import os
import PIL.Image
import torch
from torch import nn
from torch.utils.data import DataLoader
torch.backends.cudnn.benchmark = True

import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms.functional
torch.manual_seed(1)
from torch.utils.data import random_split, Subset
import time

# Image datasets and image manipulation
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

# Image display
import matplotlib.pyplot as plt
import numpy as np
import PIL

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter

'''
===============================================================================================
                                        Helpers
===============================================================================================
'''
class TrainableDataset(Dataset):
    def __init__(self, basedir, dirs: list, transform, patch_size=96, stride=48):
        self.basedir = basedir
        self.dirs = dirs
        self.dataPairs = []
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride




    def generateDatapairs(self):
        
        for dir in self.dirs:
            origPath = os.path.join(self.basedir, dir, "original_images")
            procPath = os.path.join(self.basedir, dir, "processed_images")

            for fileNum in range(len(os.listdir(origPath))):
                self.dataPairs.append(
                    (os.path.join(origPath, str(fileNum + 1) + ".jpg"),
                     os.path.join(procPath, str(fileNum + 1) + ".jpg"))
                     )
    
    def extract_patches(self, image):
        patches = []
        _, h, w = image.shape
        for i in range(0, h - self.patch_size + 1, self.stride):
            for j in range(0, w - self.patch_size + 1, self.stride):
                patch = image[:, i:i+self.patch_size, j:j+self.patch_size]
                patches.append(patch)
        return patches

    def __len__(self):
        return len(self.dataPairs)
    
    def __getitem__(self, index):
        original_image = PIL.Image.open(self.dataPairs[index][0]).convert("RGB")
        processed_image = PIL.Image.open(self.dataPairs[index][1]).convert("RGB")
        
        if self.transform != None:
            original_image = self.transform(original_image)
            processed_image = self.transform(processed_image)

        orig_patches = self.extract_patches(original_image)
        proc_patches = self.extract_patches(processed_image)

        return orig_patches, proc_patches


# get_model_name adapted from APS360 Lab 2
def get_model_name(name, batch_size, learning_rate, epoch):
    path = "./model_savepoint/model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path

# plot_training_curve adapted from APS360 Lab 2
def plot_training_curve(path):
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    n = len(train_loss) # number of epochs
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

def saveBatchOutput(inputTensor: torch.Tensor, path: str, fileNames: str, epoch: int):
    
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    

    for i, img in enumerate(inputTensor):
        img.cpu()
        name = fileNames + f"epoch{epoch}_image{i}.jpg"
        torchvision.utils.save_image(img, os.path.join(path, name))
        print("Saved " + name)

#def collate_fn(batch):
#    max_height = max(max(orig.shape[1], proc.shape[1]) for orig, proc in batch)
#    max_width = max(max(orig.shape[2], proc.shape[2]) for orig, proc in batch)
#    pad_batch_orig = [
#      torchvision.transforms.functional.pad(orig, (0, 0, max_width - orig.shape[2], max_height - orig.shape[1])) for orig, _ in batch]
#    pad_batch_proc = [
#      torchvision.transforms.functional.pad(proc, (0, 0, max_width - proc.shape[2], max_height - proc.shape[1])) for _, proc in batch]
   
#    return torch.stack(pad_batch_orig), torch.stack(pad_batch_proc)

def collate_fn(batch):
    orig_patches = []
    proc_patches = []
    for orig_list, proc_list in batch:
        orig_patches.extend(orig_list)
        proc_patches.extend(proc_list)

    return torch.stack(orig_patches), torch.stack(proc_patches)

'''
===============================================================================================
                                Setting up neural network
===============================================================================================
'''
# Architecture based on https://arxiv.org/pdf/1501.00092
# Model + hyperparameters from 3.1 and 3.2
class SISR(nn.Module):
    def __init__(self):
        super(SISR, self).__init__()
        self.name = "SISR"
        self.conv1 = nn.Conv2d(3, 64, 9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, 1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, 5, padding=2)


    def forward(self, x):
        x = F.relu(self.conv1(x)) # Patch extraction and representation
        x = F.relu(self.conv2(x)) # Non-linear mapping
        x = self.conv3(x) #Reconstruction

        return x

'''
===============================================================================================
                                     Training Helpers
===============================================================================================
'''
enableCuda = True

# getLossAcc written by Michael Del Duca for Lab 3, adapted for use here
def getLossAcc(model, criterion, dataLoader, batchSize):
    
  runningLoss = 0.0

  i = 0
  with torch.no_grad():
    for orig, proc in iter(dataLoader):

        if enableCuda and torch.cuda.is_available():
            orig = orig.cuda()
            proc = proc.cuda()


        output = model(proc)
        loss = criterion(output, orig)

        # Loss
        runningLoss += loss.item()

        i += 1

  avgLoss = runningLoss / i

  return avgLoss

# trainNet written by Michael Del Duca for Lab 3, adapted for use here
def trainNet(model, data, validationData, batchSize=32, learningRate=0.005, numEpochs=4, getVal=True, saveCheckpoints=True):
  model.train()

  criterion = nn.MSELoss()
  optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum=0.9)

  trainLoader = DataLoader(dataset=data, batch_size=batchSize, shuffle=False, 
                            collate_fn=collate_fn, num_workers=8, pin_memory=True, persistent_workers=True)
  valLoader = DataLoader(dataset=validationData, batch_size=batchSize, shuffle=False, 
                        collate_fn=collate_fn, num_workers=8, pin_memory=True, persistent_workers=True)

  trainLossArr = np.zeros(numEpochs)
  iterationsArr = np.zeros(numEpochs)
  valLossArr = np.zeros(numEpochs)

  t1 = time.time()
  n = 0
  for epoch in range(numEpochs):
    runningLoss = 0.0

    i = 0
    for orig, proc in iter(trainLoader):

        if enableCuda and torch.cuda.is_available():
            orig = orig.cuda()
            proc = proc.cuda()

        output = model(proc)

        loss = criterion(output, orig)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Loss
        runningLoss += loss.item()
        
        if (i == int(len(data)/batchSize) - 1):
            saveBatchOutput(output, f"outputs/epoch_{epoch}_data/output", "output", epoch)
            saveBatchOutput(orig, f"outputs/epoch_{epoch}_data/original", "orig", epoch)
            saveBatchOutput(proc, f"outputs/epoch_{epoch}_data/processed", "proc", epoch)

        i += 1


    # Average loss across epoch
    trainLossArr[epoch] = (runningLoss / i)

    iterationsArr[epoch] = (epoch)

    if getVal:
      valLoss = getLossAcc(model, criterion, valLoader, batchSize)

      valLossArr[epoch] = (valLoss)

    print(f"Epoch {iterationsArr[n]}:\n Training Loss = {trainLossArr[n]}\n")

    if getVal:
     print(f"\n Validation Loss = {valLossArr[n]}\n")

    # Save the current model (checkpoint) to a file - Code from lab2
    modelPath = get_model_name(model.name, batchSize, learningRate, epoch)

    if saveCheckpoints:
      torch.save(model.state_dict(), modelPath)


    n += 1

  t2 = time.time()

  dt = t2 - t1
  print(f"Total time to train: {dt}")

  # Write the train/test loss/err into CSV file for plotting later
  if (saveCheckpoints):
    np.savetxt("{}_train_loss.csv".format(modelPath), trainLossArr)
    np.savetxt("{}_val_loss.csv".format(modelPath), valLossArr)


'''
===============================================================================================
                                     Training Network
===============================================================================================
'''
if __name__ == "__main__":
    transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
                )

### Comment out this block 

              
    #dataset = TrainableDataset("D:", ["book_vids"], transform)
    #dataset.generateDatapairs()

    #trainSize = round(0.7 * len(dataset.dataPairs))
    #valSize = round(0.15 * len(dataset.dataPairs))
    #testSize = len(dataset.dataPairs) - trainSize - valSize

    #trainSet, valSet, testSet = random_split(dataset, [trainSize, valSize, testSize])

    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

####

        # Load datasets from pre-split directories


    train_dataset = TrainableDataset("/content/extracted_folder/dataset/train", [""], transform)
    train_dataset.generateDatapairs()

        # Reduce size to 50%
    half_size = len(train_dataset) // 2
    train_dataset, _ = random_split(train_dataset, [half_size, len(train_dataset) - half_size])

    val_dataset = TrainableDataset("/content/extracted_folder/dataset/validation", [""], transform)
    val_dataset.generateDatapairs()

    # Reduce size to 50%
    half_size = len(val_dataset) // 2
    val_dataset, _ = random_split(val_dataset, [half_size, len(val_dataset) - half_size])

    test_dataset = TrainableDataset("/content/extracted_folder/dataset/test", [""], transform)
    test_dataset.generateDatapairs()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")


    SISRModel = SISR()

    if enableCuda and torch.cuda.is_available():
        SISRModel.cuda()
        print("CUDA Enabled\n")
    else:
        print("CUDA not available\n")

    trainNet(
        SISRModel, 
        train_dataset, 
        val_dataset,
        batchSize=8,
        learningRate=5e-3,
        numEpochs=40,
        getVal=True,
        saveCheckpoints=True
        )
