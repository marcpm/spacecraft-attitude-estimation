

import json
from pose_utils import PyTorchSatellitePoseEstimationDataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms




def training_log(model, criterion, optim, scheduler, data_transforms, log_time):
  hyperdict = {}
  hyperdict["model"] = str(model.__class__)
  hyperdict["transforms"] = str(data_transforms.__dict__)
  hyperdict["criterion"] = str(criterion.__class__)
  hyperdict["optim"] = {"class": str(optim.__class__),"hyperparams": str(optim.__dict__["defaults"])  }
  hyperdict["scheduler"] = {"class": str(scheduler.__class__),"gamma": str(scheduler.__dict__["gamma"]), "step": str(scheduler.__dict__["step_size"])  }
  
  json_dict = {"hyperdict": hyperdict}
  json_name = log_time + "-hyper_specs.txt"
  with open (json_name,"w") as f:
    f.write(json.dumps(hyperdict))
  
  #hyperdata = pd.DataFrame(hyperdict)
 
  #hyperdata.to_csv(csv_name)
  upload_file_mounted(json_name)
  
  return


def compute_normalization_params(speed_root):
  
  data_basictransforms = transforms.Compose([transforms.Resize((224*1.6,224)),transforms.ToTensor()])
  full_dataset = PyTorchSatellitePoseEstimationDataset('train', speed_root, data_basictransforms)
  
  dataloader = DataLoader(full_dataset, batch_size=20, shuffle=True, num_workers=8)
  
  #train_means = torch.stack([t.mean(1).mean(1) for t, c in tqdm(full_dataset)])
  #train_stds = torch.stack([t.std(1).std(1) for t,c in tqdm(full_dataset)])
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  means = []
  stds = []
  i = 0
  for inputs, labels in dataloader:


    inputs = inputs.to(device)
    if i==0:
      print(f"inputs size: {inputs.shape}/n")
    means.append(inputs.mean(0).mean(1).mean(1))
    stds.append(inputs.std(0).std(1).std(1))
    i+=1

  means = torch.stack(means)
  stds = torch.stack(stds)  
  data_mean = means.mean(0)
  data_std = stds.mean(0) 
  print("\n")
  print(f" mean: {data_mean}\n std:{data_std}" )

  return list(data_mean.cpu().numpy()), list(data_std.cpu().numpy())
  
  
  def find_mean_std2():
  	"""Slower
  	"""

    mean = 0.
    std = 0.

    for images, _ in loader:
      batch_samples = images.size(0)
      images = images.view(batch_samples, images.size(1), -1)
      mean += images.mean(2).sum(0)
      std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)

    print(mean,std)
    return mean, std
  
 