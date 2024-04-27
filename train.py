import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from transformer import TransformerChatBot, MaskedCrossEntropy
from utils import ChatDataset, TransformerCrossEntropyLoss, setup_seed
import json
from torch.utils.tensorboard import SummaryWriter



with open("config.json", "r") as f:
    config = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.cuda.is_available())
     
# 设置随机数种子
setup_seed(4595)

writer = SummaryWriter('./log') # 实例化 tensorboard summary writer


X : np.ndarray = np.load("./src/X.npy")# 加载数据集
X_mask: np.ndarray = np.load("./src/X_mask.npy")
Y : np.ndarray = np.load("./src/Y.npy") 
Y_tgt : np.ndarray = np.load("./src/Y_target.npy") 
Y_mask : np.ndarray = np.load("./src/Y_mask.npy")


# 假设 X 和 Y 已经被定义为 list[list[int]]
dataset = ChatDataset(X, Y, Y_tgt, X_mask, Y_mask)

# 划分数据集
train_size = int(0.8*len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# DataLoader
batch_size = config["batch_size"]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16, prefetch_factor=1) # 多线程加载
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=16,prefetch_factor=1)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=16,prefetch_factor=1)


# 定义模型参数
vocab_size = X.max() + 1  # 假设词汇索引从0开始
embedding_dim = config["embedding_dim"]

# 实例化模型
model = TransformerChatBot(vocab_size, embedding_dim, X.shape[1], config["num_layers"], config["num_layers"])

model.to(device)

# 定义损失函数和优化器
criterion = MaskedCrossEntropy().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr =config["lr"], weight_decay = config["weight_decay"])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"], last_epoch = config["last_epoch"]) # schedule 让 lr 慢慢减小
# 训练模型
num_epochs = config["epochs"]
train_loss_lst = []
val_loss_lst = []
test_loss_lst = []
for epoch in range(num_epochs):
    model.train()

    # 创建 [maxlen, maxlen] 的训练掩码
    maxlen = X.shape[1]
    foresee_mask = torch.triu(torch.ones(maxlen, maxlen, dtype = torch.float32) * float('-inf'), diagonal=1).to(device)
    
    train_loss = 0
    count = 0 # 进度条
    for X, Y, Y_tgt, X_mask, Y_mask in train_loader:
        X, Y, Y_tgt, X_mask, Y_mask = X.to(device), Y.to(device), Y_tgt.to(device), X_mask.to(device), Y_mask.to(device)
        optimizer.zero_grad()
        
        output = model.forward(X, Y, X_mask, Y_mask, foresee_mask.clone()) 
        loss = criterion.forward(output.reshape(-1, output.size(2)), Y_tgt.view(-1), Y_mask)
        loss.backward()
         
        clip_grad_norm_(model.parameters(), max_norm = 1) # 梯度裁减，防止NAN

        optimizer.step() # 自动识别epoch
        scheduler.step() # schedular 的 step + 1， 调整学习率

        train_loss += loss.item()
        if (count % 10 == 0):
            print(f'Current LR: {scheduler.get_last_lr()}')
            print(f"train: {count/len(train_loader) * 100:.4f}%, loss : {loss.item():.6f}")

        writer.add_scalar('loss/train loss', loss.item(), count)
            
        count += 1
    
    
    
    
    
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for X, Y, Y_tgt, X_mask, Y_mask in val_loader:
            X, Y, Y_tgt, X_mask, Y_mask = X.to(device), Y.to(device), Y_tgt.to(device), X_mask.to(device), Y_mask.to(device)
            output = model.forward(X, Y, X_mask, Y_mask, foresee_mask.clone())
            loss = criterion.forward(output.reshape(-1, output.size(2)), Y_tgt.view(-1), Y_mask)
            val_loss += loss.item()
    
    train_loss_lst.append(train_loss / len(train_loader))
    val_loss_lst.append(val_loss / len(val_loader))
    print(f"Epoch {epoch + 1}, Train Loss (Entropy): {train_loss / len(train_loader):.4f}, Validation Loss (Entropy): {val_loss / len(val_loader):.4f}")

    writer.add_scalar('loss/train loss', train_loss / len(train_loader), epoch)
    


model.eval()
test_loss = 0
with torch.no_grad():
    for X, Y, Y_tgt, X_mask, Y_mask in test_loader:
        X, Y, Y_tgt, X_mask, Y_mask = X.to(device), Y.to(device), Y_tgt.to(device), X_mask.to(device), Y_mask.to(device)
        output = model.forward(X, Y, X_mask, Y_mask, foresee_mask)
        loss = criterion.forward(output.reshape(-1, output.size(2)), Y_tgt.view(-1), Y_mask)
        test_loss += loss.item()
print(f"Test Loss (Entropy): {test_loss / len(test_loader):.4f}")

torch.save(model.state_dict(), "./trained model/Transformer.pth")
print("///////////model saved////////////")




