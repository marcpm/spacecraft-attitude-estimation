from pose_utils import PyTorchSatellitePoseEstimationDataset
from pose_utils.colab import upload_file_mounted
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
from submission import SubmissionWriter
import pandas as pd
import time
import json

def train_model(model, scheduler, optimizer, criterion, dataloaders, device, dataset_sizes, num_epochs, epoch_start, log_time):

    """ Training function, looping over epochs and batches. Returns the trained model. """
    since = time.time()
    start_time = time.strftime("%b%d_%H%M",time.localtime(time.time()))
    print(f"Training starting at {start_time} ...\n")
    losses_hist = {"train": [], "val": []}

    best_loss = 99999.0

    # epoch loop
    for epoch in range(epoch_start,num_epochs):
      print("_" * 20)
      print("Epoch {}/{}".format(epoch, num_epochs - 1))
      for phase in ["train", "val"]:
            if phase == "train":
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()

            running_loss = 0.0

            # batch loop
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.float().cuda())

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            losses_hist[phase].append(epoch_loss)
            df = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in losses_hist.items()])   )  
            csv_name = log_time + "-train_val_losses.csv"
            df.to_csv(csv_name)
            upload_file_mounted(csv_name)
            print("{} Loss: {:.4f}".format(phase, epoch_loss)) 
            
            state = { "epoch": epoch,
                      "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(),
                      "loss": loss}
            
            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_state = state
                
                best_name = log_time + "-best-model.pth"
                torch.save(state, best_name)
                #delete_file("best-model.pth")
                upload_file_mounted(best_name)
                                
            if phase == "val":
                latest_name = log_time + "-latest-model.pth"
                torch.save(state, latest_name)
                #delete_file("recent-model.pth")
                upload_file_mounted(latest_name)

                

    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_loss))
    model.load_state_dict(best_model_state["model"])
    return model


def evaluate_model(model, dataset, device, submission_writer, batch_size, real=False,):

    """ Function to evaluate model on \'test\' and \'real_test\' sets, and collect pose estimations to a submission."""

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    model.eval()
    for inputs, filenames in dataloader:
        with torch.set_grad_enabled(False):
            inputs = inputs.to(device)
            outputs = model(inputs)

        q_batch = outputs[:, :4].cpu().numpy()
        r_batch = outputs[:, -3:].cpu().numpy()

        append = submission_writer.append_real_test if real else submission_writer.append_test
        for filename, q, r in zip(filenames, q_batch, r_batch):
            append(filename, q, r)
    return


def submission_from_model(speed_root, data_transforms, batch_size, model, optimizer, scheduler, loss, device, submission, checkpoint_path):

      
    checkpoint = torch.load(checkpoint_path)
    epoch_start = checkpoint["epoch"] + 1
    initialized_model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])


        # Generating submission
    submission = SubmissionWriter()
    test_set = PyTorchSatellitePoseEstimationDataset('test',  speed_root, data_transforms)
    real_test_set = PyTorchSatellitePoseEstimationDataset('real_test',  speed_root, data_transforms)

    evaluate_model(initialized_model, test_set, device, submission, batch_size, real=False)
    evaluate_model(initialized_model, real_test_set, device, submission, batch_size, real=True)
    
    timestamp = str(time.asctime( time.localtime(time.time()) ))
    #sub_name = str(random.randint(10,2000))
    submission.export(suffix=timestamp)
    file_string = "submission_" + timestamp+".csv"
    upload_file_mounted(file_string)

    return



  