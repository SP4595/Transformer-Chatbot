# Transformer-Chatbot
-------------------------------

## How To Use

- Please download Cornell Movie Dialog Corpus and move every files to move-corups folder.

- Please use `config.json` to modify the settings.

- Then use `train.py` file to train the model.

- `MainUI.py` is the main UI for the Chatbot

------------------------------------

## 一点 nn.Transformer 的训练心得:

- padding mask 和 look-ahead mask 一个不能少

- 注意 nn.Transformer 输出本身就满足 teacher forcing， 输入输出 sequence 长度都是一样的

- 注意loss function也要mask padding, 因为 padding mask 的机制，所以我们只防止了其他position注意到padding的position，没有禁止 padding 的 position 注意其他 position

- batch 是一个非常有用的防止局部最优解的方法，建议最好就是 32 ~ 128 （我选择 64）

- Dropout层除了输出层外几乎都可以加（比如 FNN 解嵌入层的输入就可以加上Dropout）

- 如果 batch 选太大了，loss是降不下来的

- 要善用 lr_schedular, 这个的超参非常难调，但是非常重要！

- 现代实现中常常用 `nn.embedding` 来进行位置编码，而不是传统的三角函数方法

- transformer层堆叠的层数要和数据集规模匹配，否则叠再多也没什么用

- transformer输出解嵌入其实就一个nn.Linear就可以了，最多两层，不要再多了， 大头留给 Transformer 层

- 如无必要，不要改变隐藏层维度！

- 如果 pyTorch 无缘无故报什么 `replacement error`， 那么大概率是你用了一些 `Batch size` 敏感的操作，然后最后一个 batch 不能整除，size小了一些导致你的 reshape 函数中自动调用了某些 replace 操作