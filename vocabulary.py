import json


class VocabCreater:
    '''
    Vocab 的目的是为每个词汇提供一个 unique 编号, 为embeding做准备
    '''
    def __init__(self) -> None:
        self.word2id = {} # 创建 Map

        ## 特殊标记 ##
        self.word2id['<pad>'] = 0  # Pad Token (128开始)
        self.word2id['<s>'] = 1  # Start Token
        self.word2id['</s>'] = 2  # End Token
        self.word2id['<unk>'] = 3  # Unknown Token

        self.wordset = {} # 创建set

        self.w_i = 4 # 从4开始记录
            
    
    def add_vocab(self, strlst : list[str]) -> None:
        '''
        在词汇表后加上 strlst 中出现的新词
        '''
            
        self.wordset = set(strlst) # 去重
        for word in self.wordset: # 标记
            self.word2id[word] = self.w_i #前四个被保留了
            self.w_i += 1

        for i in range(128): # 记下传统ascii码词
            self.word2id[chr(i)] = self.w_i
            self.w_i += 1

## main code ##


vocab = VocabCreater()
strlst = [] # 装所有可能的word

with open('./movie-corpus/utterances.jsonl', 'r', encoding="utf-8") as f:
    total_line = 0
    for line in f:
        total_line += 1 # 统计行数

with open('./movie-corpus/utterances.jsonl', 'r', encoding="utf-8") as f:
    a = 0
    for line in f:
        data = json.loads(line)
        for i in range(len(data['meta']['parsed'])):
            for tok_dic in data['meta']['parsed'][i]['toks']:
                strlst.append(tok_dic['tok'].lower()) # 全部化为小写然后计入strlst
        if (a % 10000 == 0) :
            print(f"{a / total_line:.3%}")
        a += 1

vocab.add_vocab(strlst)
print(vocab.word2id.__len__())

with open('./src/voc.json', 'w') as f:
    json.dump(vocab.word2id, f, indent=4)
