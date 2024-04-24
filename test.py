import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from LSTMs import LSTMAutoencoder
from transformer import TransformerChatBot
from utils import ChatDataset, TransformerCrossEntropyLoss, setup_seed, reverse_map
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.cuda.is_available())
     

with open("./src/voc.json",'r') as jsonfile: # 读取编码映射表
    word2id = json.load(jsonfile) 
id2word = reverse_map(word2id) # 反转

# 设置随机数种子
setup_seed(4595)


X : np.ndarray = np.load("./src/X.npy")# 加载数据集
X_mask: np.ndarray = np.load("./src/X_mask.npy")
Y : np.ndarray = np.load("./src/Y.npy") 
Y_mask : np.ndarray = np.load("./src/Y_mask.npy")


# 假设 X 和 Y 已经被定义为 list[list[int]]
dataset = ChatDataset(X, Y, X_mask, Y_mask)

# 划分数据集
train_size = int(0.6*len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# DataLoader
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16, prefetch_factor=1) # 多线程加载
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=16,prefetch_factor=1)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=16,prefetch_factor=1)


# 定义模型参数
vocab_size = X.max() + 1  # 假设词汇索引从0开始
embedding_dim = 256
hidden_dim = 256

# 实例化模型
model = TransformerChatBot(vocab_size, embedding_dim, hidden_dim, 10, 10)
model.load_state_dict(torch.load("./trained model/Transformer.pth"))
model.to(device)
model.eval()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001, weight_decay = 1e-3)

# 训练模型
num_epochs = 10
train_loss_lst = []
val_loss_lst = []
test_loss_lst = []
for epoch in range(num_epochs):

    # 创建 [maxlen, maxlen] 的训练掩码
    maxlen = X.shape[1]
    foresee_mask = torch.triu(torch.ones(maxlen, maxlen, dtype = torch.float32) * float('-inf'), diagonal=1).to(device)
    
    train_loss = 0
    count = 0 # 进度条
    for X, Y, X_mask, Y_mask in train_loader:
        X, Y, X_mask, Y_mask = X.to(device), Y.to(device), X_mask.to(device), Y_mask.to(device)
        optimizer.zero_grad()
        
        output = model.forward(X, Y, X_mask, Y_mask, foresee_mask)
        next_word = output.argmax(dim=2)
        
        loss = criterion.forward(output.reshape(-1, output.size(2)), Y.view(-1))
        
        clip_grad_value_(model.parameters(), clip_value = 10) # 梯度裁减，防止NAN

        optimizer.step() # 自动识别epoch
        train_loss += loss.item()
        if (count % 10 == 0):
            print(f"train: {count/len(train_loader) * 100:.4f}%, loss : {loss.item():.6f}")
        count += 1
    
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for X, Y, X_mask, Y_mask in val_loader:
            X, Y, X_mask, Y_mask = X.to(device), Y.to(device), X_mask.to(device), Y_mask.to(device)
            output = model.forward(X, Y, X_mask, Y_mask, foresee_mask)
            loss = criterion.forward(output.reshape(-1, output.size(2)), Y.view(-1))
            val_loss += loss.item()
    
    train_loss_lst.append(train_loss / len(train_loader))
    val_loss_lst.append(val_loss / len(val_loader))
    print(f"Epoch {epoch + 1}, Train Loss (Entropy): {train_loss / len(train_loader):.4f}, Validation Loss (Entropy): {val_loss / len(val_loader):.4f}")