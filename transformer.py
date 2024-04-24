import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import numpy as np

with open("config.json", "r") as f:
    config = json.load(f)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## para seting
drop_out_rate = config["droupout"]

def infmask_to_zeromask(padding_mask : torch.Tensor,):
    one_mask = torch.ones_like(padding_mask)
    zero_mask = torch.zeros_like(padding_mask)
    return torch.where(padding_mask != 0, zero_mask, one_mask) # torch重写了逻辑运算符变为element wise的运算。true的话选取第一个对应位置的值，false的话选取第二个对应位置的值


class EmbeddingLayer(nn.Module):
    '''
    添加positional embedding, Transformer 和 LSTM 不一样，不提供positional embedding 他就无法学习到语序的影响
    '''
    def __init__(self, vocab_size, embed_dim, max_len = config["max_len"], learned_position_embedding = False) -> None:
        super().__init__()
        self.word_embed = nn.Embedding(vocab_size, embed_dim)
        self.learned_pos = learned_position_embedding
        self.position_embed = None
        self.embed_dim = embed_dim # 为了方便在embedding之后直接加和position embedding
        self.max_len = max_len
        self.dropout = nn.Dropout(drop_out_rate)
        if not learned_position_embedding:
            self.pos_embed_creater() # 创建 self.pe 并作为 常数记入 static_dict() 中， i.e. pytorch 负责保存这个pe, 在推理的时候就不需要再跑了，直接从 pth 文件中加载
        else:
            '''
            现代实现中一般使用 nn.Embedding 来执行位置编码
            '''
            self.position_embed = nn.Embedding(max_len, embed_dim) # 由于 nn.Embedding 可以把scaler映射到一个vecoter上
        
    def pos_embed_creater(self):
        # Compute the positional encodings once in log space.
        pe = torch.zeros(self.max_len, self.embed_dim) # 空白数组
        position = torch.arange(0, self.max_len).unsqueeze(1) # 构造迭代器
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2) *
                             -(np.log(10000.0) / self.embed_dim)) # 递增常量
        pe[:, 0::2] = torch.sin(position * div_term) # 偶数 sin
        pe[:, 1::2] = torch.cos(position * div_term) # 奇数 cos
        pe = pe.unsqueeze(0) # [1, max_len, embed_dim], 用于和 [B, max_len< embed_dim] 叠加
        self.register_buffer('pe', pe) # 这样 self.pe 会直接记录在模型中了

    def forward(self, input : torch.Tensor)->torch.Tensor:
        '''
        输入的 input 维度应该为 [B, seqlen]，需要 word embedding 与 pos embedding
        '''
        x = self.word_embed.forward(input)
        if not self.learned_pos:
            x = x + self.pe[:, :x.shape[1]] # 注意，可以只用pe的某一部分！
        else:
            positions = torch.arange(0, input.shape[1]).unsqueeze(0) # [1, seqlen], for broad casting
            x += self.position_embed.forward(positions)

        return self.dropout.forward(x) # 注意，需要dropout来防止神经网络对某个特定词语非常感兴趣！！！（embedding后也是如此）


class MaskedCrossEntropy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterian = nn.CrossEntropyLoss(reduction='none') # 自带softmax, none 用于方便后续处理mask，原本算出每个元素的loss之后会直接mean，现在现none，mask之后再mean

    def forward(self, input, target, Y_mask : torch.Tensor):
        '''
        src 维度 [B, seqlen, voclen]
        tgt_key_mask 维度: [B, seqlen, 1], [0 正常， -inf padding]
        mask : [1 正常, 0 padding]
        '''
        Y_mask_z = infmask_to_zeromask(Y_mask).flatten() # target shape [B*seqlen]
        loss = self.criterian.forward(input, target) * Y_mask_z
        return loss.mean()

class ResBlock(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, out_dim)
        self.bn1 = nn.LayerNorm(mid_dim)
        self.bn2 = nn.LayerNorm(out_dim)
        self.adapt_fc = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None # adjust the dimension
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        residual = x if self.adapt_fc is None else self.adapt_fc(x) # derectly jump two layers (need adjest / not needed)
        a1 = self.dropout(self.fc1.forward(x))
        a1 = self.bn1.forward(a1) 
        a1 = F.relu(a1) # compute a1
        a2 = self.fc2.forward(a1)
        a2 = self.bn2(a2)
        a2 = F.relu(a2) # compute a2
        return  residual + a2 # res connect


class FNN(nn.Module):
    '''
    处理decoder输出之后的
    '''
    def __init__(self, idim, odim) -> None:
        '''
        linear层会自动处理三位输入， [B, N, M] 输入之后会变成 [B*N, M], 相当于不同batch的不同向量合并到一块
        输出时会自动整理回三维格式
        '''
        super().__init__()
        self.idim = idim
        self.odim = odim
        self.res1 = ResBlock(idim, 2048, odim)

        

    def forward(self, input):
        a = self.res1.forward(input)
        return a
    

class TransformerChatBot(nn.Module):
    def __init__(self, vocab_size : int, embedding_dim : int, max_len : int, encode_layer_num : int, decode_layer_num : int) -> None:

        '''
        encoder和decoder的 d_model 要一样，就是输入输出维度要一样，然后接下来维度调整和softmax交给FNN来处理
        encoder 的 key 和 value 如何传递给 decoder？ 实际上pytroch默认只使用最上层dencoder的输出（每个输入向量只提供一个输出向量，同时作为这个输入对应的key和value）
        
        #encoder
        输入: [B, N_i, M]
        输出: [B, N_i, d_model]

        #decoder
        输入输出一样（seq2seq）: [B, N_O, d_model]

        #fnn
        输入 : [B, N_O, d_model]
        输出 : [B, N_O, 1] # softmax之后
        '''
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_size, embedding_dim, max_len = max_len) # 词嵌入
        self.vocab_size = vocab_size
        self.encoder = nn.TransformerEncoder(
            encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead = 16, batch_first = True), # 注意， TransformerEncoder只是把 encoderlayer多堆叠几层的方法，核心还是encoderlayer
            num_layers = encode_layer_num, # 堆叠多少层？                                  # batch first = true 可有可无
            norm = nn.LayerNorm(normalized_shape = embedding_dim), # transform更喜欢layernorm
            
        )
        
        self.decoder = nn.TransformerDecoder(
            decoder_layer = nn.TransformerDecoderLayer(d_model = embedding_dim, nhead = 16, batch_first = True), # 同理，这个也是多堆叠几层，核心还是decoderlayer
            num_layers = decode_layer_num, # encoder，decoder叠的层数可以不一样
            norm = nn.LayerNorm(normalized_shape = embedding_dim), # transform更喜欢layernorm
        )
        self.fnn = FNN(embedding_dim, vocab_size) # 输出

    def forward_prop(self, X: torch.Tensor, X_mask: torch.Tensor ):
        '''
        注意，nn.TransformerDecoder的输出是一致的
        '''
        batch_size, maxlen = X.shape
        self.eval()  # 确保模型处于评估模式

        # Embed the input sequence
        embed = self.embedding(X)  # 使用直接调用而非.forward，这是更常见的PyTorch做法

        # Encode the input sequence
        memory = self.encoder(embed, src_key_padding_mask=X_mask)  

        # Initialize the target sequence with the start-of-sequence token
        # 确保这个开始符号与您的词汇表设置匹配
        start_token_index = 1  # 假设1是开始符号的索引
        tgt = torch.full((batch_size, 1), start_token_index, dtype=torch.long, device='cuda')
        # Loop through the maximum length of the sequence
        for i in range(maxlen):
            tgt_embed = self.embedding(tgt)  # Embed the current target sequence
            output = self.decoder.forward(tgt_embed, memory, tgt_is_causal = True)  # Decode the sequence
            output : torch.Tensor = self.fnn(output)  # Apply the feedforward network to the last output
            # 用 multinnormal
            selected_val = torch.multinomial(F.softmax(output.squeeze(0)[-1] / config["Temperature"]), 1) # Temperature 在进softmax前应用，可以改变分布的集中程度

            # 获取被选中的值
            next_word = selected_val.reshape(1, -1)[-1].unsqueeze(0)

            # Update the target sequence (this is where you could add more sophisticated sampling strategies)
            tgt = torch.cat([tgt, next_word], dim=1)

        return tgt
    
    def forward(self, X : torch.Tensor, Y: torch.Tensor, X_mask : torch.Tensor, Y_mask : torch.Tensor, foresee_mask : torch.Tensor):
        '''
        训练用代码
        '''
        embed = self.embedding.forward(X)  # 词嵌入
        memory = self.encoder.forward(embed, src_key_padding_mask = X_mask)  # 编码
        tgt_embed = self.embedding.forward(Y)  # 目标序列嵌入
        output = self.decoder.forward(tgt_embed, memory, tgt_key_padding_mask = Y_mask, tgt_mask = foresee_mask, memory_key_padding_mask = X_mask, tgt_is_causal = True)  # 解码 (同时也要 mask memory中 的无效 value)
        output = self.fnn(output)  # 应用FNN
        return output

