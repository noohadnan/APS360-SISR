from torchvision import datasets, transforms
from SISR import enableCuda, SISR, get_model_name, plot_training_curve, saveBatchOutput
import torch
import PIL
import PIL.Image

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

plot_training_curve(bestModelPath)

transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
            )

imgToDeblur =  PIL.Image.open("C:/Users/mdeld/OneDrive/Desktop/Computer Engineering Year 3/Semester 2/Applied Fundamentals of Deep Learning/Project/APS360-SISR/Model/new/epoch_new_data/output/outputepoch2_image0.jpg").convert("RGB")

new = bestModel(transform(imgToDeblur).cuda())

new = new.unsqueeze(0)

saveBatchOutput(new, "new/epoch_new_data/output", "output", 3)
