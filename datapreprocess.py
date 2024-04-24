import json
import re
import numpy as np
'''
本文件最后保存
X : 一行一个zero padding 后的语料
X_len : padding前的长度
Y : X 的回应
Y_len : 同理
'''

if (__name__ == "__main__"):

    with open("config.json", "r") as f:
        config = json.load(f)

    with open('./movie-corpus/utterances.jsonl', 'r', encoding="utf-8") as f:
        total_line = 0
        for line in f:
            total_line += 1 # 统计行数
    
    a = 0
    total_dialog = [] # 装对话 [[[]]], 分为 [N, M, T] 一共有N组对话,其中每组对话有M行（你一行我一行）每行话有 T 个词 （可能有多个句子）
    with open('./movie-corpus/utterances.jsonl', 'r', encoding="utf-8") as f_data:
    
        dialog_id = "L1044" # 最开始的对话
        dialog_contant = [] # 保存一整个对话中的句子
        for line in f_data:
            data = json.loads(line) # 一行一行读取

            ## 判断对话结束 ## 
            dialog_id_now = data["conversation_id"] #读取本行所属对话的id
            if (dialog_id_now != dialog_id):
                dialog_id = dialog_id_now # 更新对话
                total_dialog.append(dialog_contant) # 将上一个对话存入总对话中
                dialog_contant = [] # 清空对话contant 
                

            ## 如果对话没有结束 ##

            reply = []
            # 每个语素：
            for i in range(len(data['meta']['parsed'])):
                for tok_dic in data['meta']['parsed'][i]['toks']:
                    reply.append(tok_dic['tok'].lower()) # 全部化为小写然后计入strlst
                    if (a % 10000 == 0) :
                        print(f"{a / total_line:.3%}")
            a += 1

            dialog_contant.append(reply) # 把replay插入dialog contant中
            
    print(total_dialog)    
    
    # 处理成六个numpy
    '''
    实践重点：transformerDecoder每次输入一个[N, S, T]维度的tensor，输出也是，他会默认输出 **下一个位置** （自回归），所以训练的Y要分两个，输出和输入要shift一格 ！！！！
    '''
    X = []
    Y = []
    Y_target = [] # 向左shift一位，用于作为crossentropy loss的target
    X_len = []
    Y_len = []
    X_mask = []
    Y_mask = []

    max_len = 0
    a = 0
    max_len_limit = config["max_len"]
    for dialogi in range(len(total_dialog)):
        # 对于每个对话
        for sentencei in range(len(total_dialog[dialogi]) - 1): # 最后一组不算对话   
            if(len(total_dialog[dialogi][sentencei]) <= max_len_limit and len(total_dialog[dialogi][sentencei+1]) <= max_len_limit):
                X.append(total_dialog[dialogi][sentencei])
                X_len_now = len(total_dialog[dialogi][sentencei]) + 2
                X_len.append(X_len_now) # 注意还没有加上 <s></s>
                X_mask.append([0]*X_len_now + [-np.inf] * (max_len_limit + 2 - X_len_now))

                Y.append(total_dialog[dialogi][sentencei + 1])

                Y_target.append(total_dialog[dialogi][sentencei + 1])

                Y_len_now = len(total_dialog[dialogi][sentencei + 1]) + 2
                Y_len.append(Y_len_now)
                Y_mask.append([0]*Y_len_now + [-np.inf] * (max_len_limit + 2 - Y_len_now))

                max_len = len(total_dialog[dialogi][sentencei]) if len(total_dialog[dialogi][sentencei]) > max_len else max_len
                max_len = len(total_dialog[dialogi][sentencei+1]) if len(total_dialog[dialogi][sentencei+1]) > max_len else max_len
            
            if (a % 100 == 0):
                print(f"dialog : {a/len(total_dialog):.3%}, max len : {max_len}")
        a += 1
    
    with open("./src/voc.json", "r") as f_voc: # 读取映射表
        word2id = json.load(f_voc)

    # zero padding
    for i in range(len(X)):
        X[i] = ["<s>"] + X[i] + ["</s>"] + ["<pad>"] * (max_len - len(X[i]))
        Y[i] = ["<s>"] + Y[i] + ["</s>"] + ["<pad>"] * (max_len - len(Y[i]))
        Y_target[i] = Y_target[i] + ["</s>"] + ["<pad>"] * (1 + max_len - len(Y_target[i])) # 没有 <s>

        for j in range(len(X[i])):
            X[i][j] = word2id[X[i][j]] # 转为 id
        for j in range(len(Y[i])):    
            Y[i][j] = word2id[Y[i][j]]
        for j in range(len(Y_target[i])):    
            Y_target[i][j] = word2id[Y_target[i][j]]

    print(len(X), len(Y))
    for i in range(10):
        print()
        print("a:", X[i], "a_len:", X_len[i])
        print("a_mask:", X_mask[i])  
        print("b:", Y[i], "b_len:", Y_len[i])  
        print("b_mask:", Y_mask[i])  
        print("tgt:", Y_target[i], "b_len:", Y_len[i])  
        print("b_mask:", Y_mask[i])  
        print()

    X_np = np.array(X)
    Y_np = np.array(Y)
    Y_target_np = np.array(Y_target)
    X_len_np = np.array(X_len)
    Y_len_np = np.array(Y_len)
    X_mask_np = np.array(X_mask, dtype = np.float32)
    Y_mask_np = np.array(Y_mask, dtype = np.float32)

    print(X_np.shape, X_len_np.shape, Y_np.shape, Y_len_np.shape, Y_target_np.shape)

    np.save("./src/X.npy", X_np)
    np.save("./src/Y.npy", Y_np)
    np.save("./src/Y_target.npy", Y_target_np)
    np.save("./src/X_len.npy", X_len_np)
    np.save("./src/Y_len.npy", Y_len_np)
    np.save("./src/X_mask.npy", X_mask_np)
    np.save("./src/Y_mask.npy", Y_mask_np)

      