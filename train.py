import argparse
import torch
import sys
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.utils.data as Data
from conformer import Conformer
from fe import AIshellData
from fe import get_ground_truth, get_num_class
from config import get_fbank_dim
from utils import cer_for_batch

# sentence 的最大长度
sentence_max_len = 64

# 拿到一个音频数据本来的长度
def pre_process_audio(data):
    len_list = []
    for batch in range(data.shape[0]):
        audio_data = data[batch]
        for dim in range(audio_data.shape[0]):
            if torch.all(audio_data[dim] == 0):
                len_list.append(dim+1)
                break
    
    # 用0填补的代码在这里:
    max_len = max(len_list)
    new_data = []
    for batch in range(data.shape[0]):
        audio_data = data[batch]
        audio_data = audio_data[:max_len, :].tolist()
        new_data.append(audio_data)
    new_data = torch.tensor(new_data)
    return new_data, torch.LongTensor(len_list)
    
    
    # 在这里做一个改进，是不是2048太大了，让他一直识别出来0？
    # max_len = max(len_list)
    # new_data = []
    # for batch in range(data.shape[0]):
    #     audio_data = data[batch]
    #     audio_data = audio_data[:max_len, :]
    #     new_audio_data = torch.rand(audio_data.shape[0], audio_data.shape[1])
    #     # 这里我也改了一下，用随机数填充，而不是用0填充
    #     act_len = len_list[batch]
    #     new_audio_data[:act_len, :] = audio_data[:act_len, :]
    #     new_data.append(new_audio_data)
    # new_data = torch.stack(new_data)
    # return new_data, torch.LongTensor(len_list)
    
    
# 是不是target中的0太多了？
def process_target_sentence(target_sentences, target_len):
    max_len = max(target_len)
    new_sentence = []
    for sentence in target_sentences:
        new_sentence.append(sentence[:max_len])
    
    return new_sentence

# TODO：给我一个index的元组，返回他们对应的ground_truth，把一个batch弄成一样长和他们的长度
def pre_process_sentence(indexs):
    indexs = list(indexs)
    target_sentences = []
    target_len = []
    global sentence_max_len
    for index in indexs:
        str_sentence, int_sentence = get_ground_truth(index[:-4])
        int_sentence = np.insert(int_sentence,0,1)
        int_sentence = np.append(int_sentence,2)
        target_len.append(int_sentence.shape[0])
        target_shape=(sentence_max_len)
        expanded_target = np.zeros(target_shape)
        expanded_target[:int_sentence.shape[0]] = int_sentence
        int_sentence = expanded_target
        int_sentence = int_sentence.astype(int)
        target_sentences.append(int_sentence)
    target_sentences = process_target_sentence(target_sentences, target_len)
    return torch.LongTensor(target_sentences), torch.LongTensor(target_len)


parser = argparse.ArgumentParser(description="Train conformer")
parser.add_argument('--epoch', type=int, help='train epoch', default=1000000)
parser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
parser.add_argument('--device', type=str, help='device used', default='cuda')
parser.add_argument('--input_dim', type=int, help='input_dim', default=80)
parser.add_argument('--encoder_dim', type=int, help='encoder_dim', default=512)
parser.add_argument('--num_encoder_layers', type=int, help='num_encoder_layers', default=3)
parser.add_argument('--conv_kernel_size', type=int, help='conv_kernel_size at conformer ConvMoudle used by Depwise1d', default=31)
parser.add_argument('--train_data_path', type=str, help='train_data_path', default='/root/audodl-tmp/AIshell1_train')

args = parser.parse_args()


train_loader = Data.DataLoader(AIshellData('/root/autodl-tmp/AIshell1_train/'),batch_size=16, shuffle=True)
test_loader = Data.DataLoader(AIshellData('/root/autodl-tmp/AIshell1_test/'),batch_size=16, shuffle=True)
criterion = nn.CTCLoss().to(args.device)

model = Conformer(num_classes=get_num_class(), 
                  input_dim=get_fbank_dim(), 
                  encoder_dim=args.encoder_dim, 
                  num_encoder_layers=args.num_encoder_layers,
                  conv_kernel_size = args.conv_kernel_size).to(args.device)


optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,200], gamma=0.1)

for epoch in range(0, args.epoch):
    count = 0
    for data, file_name in tqdm(train_loader):
        # 声明需要的数据
        inputs = data.float()
        inputs = inputs
        inputs, inputs_lengths = pre_process_audio(inputs)
        inputs = inputs.to(args.device)
        targets, targets_lengths = pre_process_sentence(file_name)
        targets = targets.to(args.device)
        # 训练并计算损失
        outputs, output_lengths = model(inputs, inputs_lengths)
        loss = criterion(outputs.transpose(0, 1), targets, output_lengths, targets_lengths)
        if (count % 100) == 0:
            print("=====================================================================")
            print("EPOCH:", str(epoch) , " batch:", count ," LOSS:", loss.item(), " Train CER:", sum(cer_for_batch(outputs, targets, output_lengths, epoch, count, True))/outputs.shape[0], end=' ')
            # 在测试集上进行检测
            if (count % 1000) == 0:
                model.eval()
                with torch.no_grad():
                    test_batch_count = 0
                    test_CER_count = 0.0
                    for test_data, test_file_name in tqdm(test_loader):
                        test_batch_count += test_data.shape[0]
                        test_inputs = test_data.float()
                        test_inputs = test_inputs
                        test_inputs, test_inputs_lengths = pre_process_audio(test_inputs)
                        test_inputs = test_inputs.to(args.device)
                        test_targets, test_targets_lengths = pre_process_sentence(test_file_name)
                        test_targets = test_targets.to(args.device)
                        # 训练并计算损失
                        test_outputs, test_output_lengths = model(test_inputs, test_inputs_lengths)
                        test_CER_count += sum(cer_for_batch(test_outputs, test_targets, test_output_lengths, epoch, count))
                print('Test CER:', test_CER_count/test_batch_count)
                model.train()

            if (count==0) and (epoch==0):
                pass
            else:
                scheduler.step()
            # just for test
            # sys.exit()
        # print(outputs.shape)
        # print(outputs.transpose(0, 1).shape)
                
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count+=1
    
    

