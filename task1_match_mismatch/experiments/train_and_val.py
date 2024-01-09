"""Example experiment for the 2 mismatched segments dilation model."""
# DA3 给EEG增加白噪声，使信噪比在0-15dB之间
import glob
import json
import logging
import os, sys
import tensorflow as tf
import torch
import sys
import torch.optim as optim
import torch.nn.functional as F
import math
from torch import nn
import time
import csv
# add base path to sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','..'))
from task1_match_mismatch.models.dilated_convolution_torch import Dilated_Conv_Model
from util.dataset_generator import  batch_equalizer_fn, create_tf_dataset, DataGenerator
from util.sublist import TRAIN_SUB_LIST, TEST_SUB_LIST
import random
import argparse
from task1_match_mismatch.models.DenseNet import BMMSNet


LR=0.001 
parser = argparse.ArgumentParser()
parser.add_argument('--DA', type=str, default='0')   # DA=0: no data augmentation
args = parser.parse_args()
    
def DA0(eeg, stimulus, label):  
    return eeg, stimulus, label
def DA1(eeg, stimulus, label):      # for num_mismatch == 4
    r = random.uniform(0,1)
    if r < 2:            # chenge the aug probability by changing r
        B = eeg.shape[0]
        eeg = eeg[0:B//5]
        stimuli_att = stimulus[0][0:B//5]
        indices = torch.randperm(B//5).to(eeg.device)
        eeg_shuffle = eeg[indices]
        stimuli_att_shuffle = stimuli_att[indices]
        rand = torch.rand(B//5).unsqueeze(-1).unsqueeze(-1).to(eeg.device) * 0.2
        eeg = eeg + eeg_shuffle * rand
        label = torch.cat((label, rand.squeeze(-1).repeat(5, 1)), -1) / (1 + rand.squeeze(-1).repeat(5, 1))    # (B, 6)
        eeg_stack = torch.cat([eeg, eeg, eeg, eeg, eeg], 0)
        wat_shuffle_stack = torch.cat([stimuli_att_shuffle, stimuli_att_shuffle, stimuli_att_shuffle, stimuli_att_shuffle, stimuli_att_shuffle], 0)
        stimulus = stimulus + [wat_shuffle_stack]

        return eeg_stack, stimulus, label
    else:
        return eeg, stimulus, label

def DA2(eeg, stimulus=None, label=None):   


    data_cp = eeg.clone()
    batch_size, n_samples, n_channels = eeg.shape
    mask = torch.ones((batch_size, n_samples, n_channels), device=eeg.device)
    r = random.uniform(0,1)
    if r < 2:
        for i in range(batch_size):

            # Randomly select a 1-second segment to mask
            start_idx = torch.randint(0, n_samples - 96, size=(1,))
            end_idx = start_idx + random.randint(1, 96)
            # Set the values in the segment to zero
            mask[i, start_idx:end_idx, :] = 0

            channel_start_idx = torch.randint(0, n_channels - 10, size=(1,))
            channel_end_idx = channel_start_idx + random.randint(1, 10)
            mask[i, :,  channel_start_idx: channel_end_idx] = 0
            # Apply the mask to the data
            data_cp[i] = data_cp[i] * mask[i]

        return data_cp, stimulus, label
    else:
        return eeg,  stimulus, label

def DA3(x, stimulus=None, label=None):  # 加随机白噪声
    snr_low  = -5
    snr_high = 10
    r = random.uniform(0,1)
    if r < 2:
        snr = torch.rand(x.shape[0], 1, 1).to(x.device) * (snr_high - snr_low) + snr_low
        noise = torch.randn_like(x).to(x.device)
        noise_norm = torch.norm(noise, dim=(1, 2), keepdim=True)
        signal_norm = torch.norm(x, dim=(1, 2), keepdim=True)
        noise = noise / noise_norm * signal_norm * (10 ** (-snr/10))
        return x + noise, stimulus, label
    else:
        return x, stimulus, label

def DA4(eeg, stimulus=None, label=None):  
    min_val=0.3
    max_val=2
    r = random.uniform(0,1)
    if r < 2:
        random_tensor = torch.rand(eeg.shape[0], 1, eeg.shape[2])
        random_tensor = random_tensor.to(eeg.device)
        random_tensor = random_tensor * (max_val - min_val) + min_val
        return eeg * random_tensor, stimulus, label
    else:
        return eeg, stimulus, label
    


def choose_by_sub(file_list, sub_list):
    file_list_filtered = []
    for each in file_list:
        
        if int(each.split('_-_')[1].split('-')[1]) in sub_list:
            file_list_filtered.append(each)
    return file_list_filtered



def view_bar(message, num, total, batch_mean_loss, all_mean_loss):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s: loss: %.05f   \t accuracy: %.05f   \t [%s%s]  %d %%  \t%d/%d' % (message, batch_mean_loss, all_mean_loss, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total)
    sys.stdout.write(r)
    sys.stdout.flush()




def write_log(log_path, content,mode='a'):
    with open(log_path, mode) as f:
        writer = csv.writer(f) 
        writer.writerow(content)


def validation(net, criterion, validation_data_loader, val_batch_num, word='Validation', log_path=''):
    

    net.eval()
    val_loss = 0
    num_total = 0
    num_correct = 0
    
    for batch_idx, batch_info in enumerate(validation_data_loader):
        input = [torch.tensor(each.numpy()).to('cuda') for each in batch_info[0]]
        eeg = input[0]
        stimulus = input[1:]
        label = torch.tensor(batch_info[1].numpy()).float().to('cuda')
 
        out = net(input)
        
        
        loss = F.cross_entropy(out, label)
        
        num_total = num_total + input[0].shape[0]
        num_correct = num_correct + float(torch.sum(torch.argmax(out,1)==torch.argmax(label, 1)).detach().cpu().numpy())
        
    
        
        val_loss += loss.item() 

        view_bar(word, batch_idx+1, val_batch_num, val_loss/(batch_idx + 1), num_correct / num_total)
        del  label, out, batch_info, loss
        
        
    del validation_data_loader
    write_log(log_path, ['Val', '    ', val_loss/(batch_idx + 1), num_correct / num_total])   
    
    return num_correct / num_total


def train(net, train_data_loader, validation_data_loader, test_data_loader, epoch, model_dir, tr_batch_num, val_batch_num, test_batch_num, model_path, log_path):




    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3)
    
    for i_epoch in range(epoch):

        tr_loss = 0
        num_total = 0
        num_correct = 0
        net.train()            
        for batch_idx, batch_info in enumerate(train_data_loader):      


            lr = LR * (0.98 ** i_epoch) 
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr 

            label = torch.tensor(batch_info[1].numpy()).float().to('cuda')    # (batch, 5)
            
            input = [torch.tensor(each.numpy()).to('cuda') for each in batch_info[0]]  # [eeg, stimulus0, stimulus1, ..., stimulus5]
            
            eeg = input[0]   # (batch, T, channel)
            B = eeg.shape[0]

            stimulus = input[1:]      # [stimulus0, stimulus1, ..., stimulus5]. The shape of each stimulus is (batch, T, channel)

  
            
            eeg, stimulus, label = eval(f'DA{args.DA}(eeg, stimulus, label)')
            
            
  
            optimizer.zero_grad()

            out = net([eeg]+stimulus)
   
            loss = F.cross_entropy(out, label)

            nn.utils.clip_grad_norm_(net.parameters(), max_norm=5)
    
            loss.backward()
            optimizer.step()

            num_total = num_total + input[0].shape[0]

            num_correct = num_correct + float(torch.sum(torch.argmax(out,1)==torch.argmax(label, 1)).detach().cpu().numpy())
            
            tr_loss += loss.item()
        
            view_bar('Training  ', batch_idx+1, tr_batch_num, tr_loss /(batch_idx+1), num_correct/(num_total+1e-8))

            del input, label, out, batch_info, loss, eeg, stimulus
        print(f'\n the {i_epoch}th epoch training finished:\n')    
        write_log(log_path, ['Training', i_epoch+1, tr_loss /(batch_idx+1), num_correct/(num_total+1e-8)])      

        if  i_epoch % 1 == 0:
            model_name = model_path[:-4]
            torch.save(net, f'{model_name}_epoch{i_epoch}.pkl')

            val_accuracy = validation(net, None, validation_data_loader, val_batch_num, 'validing:',log_path=log_path)
            print(f'\nvalidation finished, accuracy: {val_accuracy}')

    

    return None


if __name__ == "__main__":




    window_length_s = 5
    fs = 64

    window_length = window_length_s * fs  # 5 seconds
    hop_length = int(1*fs)
    epochs = 100
    batch_size = 60
    number_mismatch = 4 



    


    # Get the current path 
    experiments_folder = os.path.dirname(__file__)
    # filepath for train
    data_folder = '/data/qiuzelin/Data/EEG/SPARRKULEE/derivatives/split_data'       
    # stimulus feature which will be used for training the model. Can be either 'envelope' ( dimension 1) or 'mel' (dimension 10) or 'MelEnv' (dimension 15)
    stimulus_features = ["mel"]
    stimulus_dimension = 10
    features = ["eeg"] + stimulus_features

    model =  BMMSNet(stimulus_dimension)
    model = torch.nn.DataParallel(model)  # use multiple GPU 
    model = model.to(device='cuda')

    # create the folder to save model and results
    results_folder = os.path.join(experiments_folder, f"results_{model.module.name}_{number_mismatch}_MM_{window_length_s}_s_{stimulus_features[0]}_dim{stimulus_dimension}_DA{args.DA}")
    os.makedirs(results_folder, exist_ok=True)
    formatted_time = time.strftime("%Y%m%d_%H%M", time.localtime(time.time()))
    model_path = os.path.join(results_folder, f"model_{number_mismatch}_MM_{window_length_s}_s_{stimulus_features[0]}_{formatted_time}.pkl")
    log_path = os.path.join(results_folder, f"model_{number_mismatch}_MM_{window_length_s}_s_{stimulus_features[0]}_{formatted_time}.csv")
    write_log(log_path, ['stage', 'epoch', 'loss', 'accuracy'],'w')

    


    train_files = [x for x in glob.glob(os.path.join(data_folder, "*_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
    train_files = choose_by_sub(train_files, TRAIN_SUB_LIST)
    train_generator = DataGenerator(train_files, window_length)
    dataset_train = create_tf_dataset(train_generator, window_length, batch_equalizer_fn,
                                        hop_length, batch_size,
                                        number_mismatch=number_mismatch,
                                        data_types=(tf.float32, tf.float32),
                                        feature_dims=(64, stimulus_dimension),
                                        wav2vec=False
                                        )
    print(f'computing sample number in triaining dataset.......')
    num_batch_train = 0
    for batch_idx, batch_info in enumerate(dataset_train): 
        num_batch_train = num_batch_train + 1
    print(f'Train batch num: {num_batch_train}')

    

    val_files = [x for x in glob.glob(os.path.join(data_folder, "*_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
    val_files = choose_by_sub(val_files, TEST_SUB_LIST)
    val_generator = DataGenerator(val_files, window_length)
    dataset_val = create_tf_dataset(val_generator,  window_length, batch_equalizer_fn,
                                        hop_length, batch_size,
                                        number_mismatch=number_mismatch,
                                        data_types=(tf.float32, tf.float32),
                                        feature_dims=(64, stimulus_dimension),
                                        wav2vec=False
                                        )
    print(f'computing sample number in val dataset.......')
    num_batch_val = 0
    for batch_idx, batch_info in enumerate(dataset_val): 
        num_batch_val = num_batch_val + 1
    print(f'Val batch num: {num_batch_val}')



    train(model, dataset_train, dataset_val, None, epochs, 'model_dir', num_batch_train, num_batch_val, None, model_path, log_path)



