from SISR import enableCuda, SISR, get_model_name, plot_training_curve
import torch

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

plot_training_curve(bestModelPath)