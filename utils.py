import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset


## 0. 设定种子 ##
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

## 1. 定义数据集 ##

class ChatDataset(Dataset):
    def __init__(self, X : np.ndarray, Y : np.ndarray, Y_tgt : np.ndarray, X_mask : np.ndarray, Y_mask : np.ndarray):
        '''
        Y_target： 是 Y shift left 的结果，因为decoder输出是“下一位”
        '''
        self.X = torch.tensor(X) # 创建 tensor (padding)
        self.Y = torch.tensor(Y)
        self.Y_tgt = torch.tensor(Y_tgt)
        self.X_mask = torch.tensor(X_mask, dtype = torch.float32) # 为了和tgt_mask 保持一致 (-torch.inf)
        self.Y_mask = torch.tensor(Y_mask, dtype = torch.float32)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx): # 迭代
        return self.X[idx], self.Y[idx], self.Y_tgt[idx], self.X_mask[idx], self.Y_mask[idx]
    

## 2. 自定义seq2seq的交叉熵 ##
    #  弃用！！！！！！
class TransformerCrossEntropyLoss(nn.Module):
    '''
    用于处理所有时间内的输出值，然后输入CrossEntropy
    '''
    def __init__(self) -> None:
        super(TransformerCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index = 0)
    
    def forward(self, input : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        # 获取尺寸
        B, maxlen, voclen = input.shape

        # 调整输出和目标的形状
        input = input.view(-1, voclen)  # 调整为 [B * maxlen, voclen]
        target = target.view(-1)  # 调整为 [B * maxlen]

        return self.criterion.forward(input, target.long())
    
## 3. 反转map的方法 ##
def reverse_map (original_dict : dict) -> dict:
    return {value: key for key, value in original_dict.items()}