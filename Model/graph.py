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
state = torch.load(bestModelPath)
bestModel.load_state_dict(state)

# plot_training_curve(bestModelPath)

transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
            )

imgToDeblur =  PIL.Image.open("D:/action_camera_dashcam_march_1st_1080p/processed_images/2.jpg").convert("RGB")

new = bestModel(transform(imgToDeblur).cuda())

unnormalize = new 
new = (new * 0.5 + 0.5)
new = new.unsqueeze(0)

saveBatchOutput(new, "new/epoch_new_data/output", "output", 1)
