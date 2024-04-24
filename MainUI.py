print("chatBot UI loading, plz wait ...")

'''
指令窗UI对话界面
'''
import torch
import numpy as np
import json
import re
from transformer import TransformerChatBot
from utils import setup_seed, reverse_map
import json

with open("config.json", "r") as f:
    config = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置随机数种子
setup_seed(4595)

# 加载数据
X : np.ndarray = np.load("./src/X.npy")# 加载数据集 (用于计算长度)

with open("./src/voc.json",'r') as jsonfile: # 读取编码映射表
    word2id = json.load(jsonfile) 
id2word = reverse_map(word2id) # 反转

# 加载模型
vocab_size = X.max() + 1
embedding_dim = config["embedding_dim"]
model = TransformerChatBot(vocab_size, embedding_dim, X.shape[1], config["num_layers"], config["num_layers"])
model.load_state_dict(torch.load("./trained model/Transformer.pth"))
model.to(device)
model.eval()

print("<--- chatBot UI, type Quit/quit to quit (input sentence no longer than 20 words) --->\n")
request = input("> You: ")
request = request.lower() # 保证输入的全是小写
while(request != "quit"):
    # 1. 读取用户输入
    reqlst = re.split(r' |\n|\.|\!|\?|\,|\s', request) 
    reqlst = [item for item in reqlst if item != ''] # 清除空字符串
    reqlst = ['<s>'] + reqlst + ['</s>'] # 加上 <s> 和 </s>
    if(len(reqlst) > 22): # 保证输入在 20 (+2) 个词以内
        print("> Error: The request must shorter than 20 word!")
        continue
    
    # 2. 用户输入预操作
    try:
        for i in range(len(reqlst)):
            reqlst[i] = word2id[reqlst[i]] # 改成id
        reqlen_original = len(reqlst) # 用于padding
        reqlst = reqlst + [0] * (config["max_len"] + 2 - len(reqlst)) # padding
        reqtensor = torch.tensor(reqlst).unsqueeze(0).to(device) # batch默认1: [1, word_nums] 
        reqlen = torch.tensor([[reqlen_original]], dtype = torch.int64) # y用于padding, 必须为int64的长度 (注意，不能送进GPU！！！！！)：[1, 1]
        # print(reqlst, reqlen)
    except:
        print("> Error: The request include some illegel words!")
        request = input("> You: ")
        request = request.lower() # 保证输入的全是小写
        continue

    # 2.5 生成X_mask

    req_mask = torch.tensor([0]*reqlen + [1] * (config["max_len"] + 2 - reqlen), dtype = torch.float32, device = device).unsqueeze(0)
    maxlen = config["max_len"] + 2
    # 3. 输出对话
    with torch.no_grad():
        outtensor : torch.Tensor = model.forward_prop(reqtensor, req_mask).cpu() # outtensor 是输出的概率tensor, 维度 [1, maxlen, voclen], 需要每个向量提取最大值
    outtensor.squeeze_(0)
    out = outtensor[outtensor != 0] # fliter掉所有等于0的
    print()
    print("> Bot: ", sep = "", end = "") # 输出
    for i in range(len(out)):
        if(out[i] == 1): # 开始符
            continue
        if (out[i] == 2): # 停止符
            break
        print(id2word[out[i].item()], " ", sep = "", end = "") # 一个一个转译输出
    print() # 换行
    request = input("> You: ") # 下一次输出
    request = request.lower() # 保证输入的全是小写