from torchvision import datasets, transforms

from Model.SISR import enableCuda, SISR, saveBatchOutput, TrainableDataset, collate_fn, unnormalize_image
import torch
import PIL
import PIL.Image
from metrics import calculate_psnr, calculate_ssim
from torch.utils.data import DataLoader, random_split
import numpy as np
import torch.nn.functional as F

torch.manual_seed(1)


# get_model_name adapted from APS360 Lab 2
def get_model_name(name, batch_size, learning_rate, epoch):
  path = "./Model/model_savepoint/model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                  batch_size,
                                                  learning_rate,
                                                  epoch)
  return path

if __name__ == "__main__":
  bestModel = SISR()

  if enableCuda and torch.cuda.is_available():
    bestModel.cuda()
    print("CUDA Enabled\n")
  else:
    print("CUDA not available\n")

  best_Model_batch_size = 8
  best_Model_learning_rate = 0.005
  best_Model_epoch = 39

  bestModelPath = get_model_name( name=bestModel.name,
                                  batch_size=best_Model_batch_size,
                                  learning_rate=best_Model_learning_rate,
                                  epoch=best_Model_epoch)
  state = torch.load(bestModelPath, map_location = torch.device('cpu'))
  bestModel.load_state_dict(state)

  transform = transforms.Compose([
              transforms.Resize((1024, 1024)),
              transforms.ToTensor(),
              transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
              )

  test_dataset = TrainableDataset("D:/dataset/test", [""], transform)
  test_dataset.generateDatapairs()
  
  half_size = len(test_dataset) // 10
  test_dataset, _ = random_split(test_dataset, [half_size, len(test_dataset) - half_size])


  testLoader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, 
                              collate_fn=collate_fn, num_workers=8, pin_memory=True, persistent_workers=True)

  runningSSIM = 0
  runningPSNR = 0
  runningBicubicSSIM = 0
  runningBicubicPSNR = 0

  numIters = len(test_dataset)

  print(numIters)

  i = 0

  with torch.no_grad():
    for orig, proc in iter(testLoader):
        if enableCuda and torch.cuda.is_available():
            orig = orig.cuda()
            proc = proc.cuda()

        h, w = proc.shape[2:]

        output = bestModel(proc)

        outputNorm = unnormalize_image(output).cpu().numpy()
        origNorm = unnormalize_image(orig).cpu().numpy()
        procNorm = unnormalize_image(proc).cpu()

        output = np.transpose(outputNorm[0], (1, 2, 0))
        orig = np.transpose(origNorm[0], (1, 2, 0))

        bicubic = F.interpolate(procNorm, size=(h, w), mode='bicubic', align_corners=False).numpy()
        bicubic = np.transpose(bicubic[0], (1, 2, 0))


        psnr = calculate_psnr(output, orig)
        ssim = calculate_ssim(output, orig)
        bicubicPSNR = calculate_psnr(bicubic, orig)
        bicubicSSIM = calculate_ssim(bicubic, orig)

        runningPSNR += psnr
        runningSSIM += ssim
        runningBicubicPSNR += bicubicPSNR
        runningBicubicSSIM += bicubicSSIM

        if (i % 100 == 0):
           print(i)
        i+=1

        # print(psnr)
        # print(ssim)
  averagePSNR = runningPSNR / numIters
  averageSSIM = runningSSIM / numIters
  averageBicubicPSNR = runningBicubicPSNR / numIters
  averageBicubicSSIM = runningBicubicSSIM / numIters

  print(f"Average PSNR = {averagePSNR}")
  print(f"Average SSIM = {averageSSIM}")
  print(f"Average Bicubic PSNR = {averageBicubicPSNR}")
  print(f"Average Bicubic SSIM = {averageBicubicSSIM}")
  exit(0)