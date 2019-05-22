import torch
import torch.nn as nn
import torch.nn.functional as F

class textCNN(nn.Module):

    def __init__(self, config, device, pretrain_embedding=None):
        super(textCNN, self).__init__()
        self.config = config
        self.device = device
        self.label_num = 5
        if pretrain_embedding is None:
            pretrain_embedding = torch.nn.init.uniform(
                torch.FloatTensor(self.config.embedding_size,
                                  self.config.embedding_dim),
                a=-0.15, b=0.15)

        self.embedding = nn.Embedding(num_embeddings=self.config.embedding_size,
                                      embedding_dim=self.config.embedding_dim)
        self.embedding.weight.data.copy_(pretrain_embedding)
        self.embedding.weight.requires_grad = True

        self.convs = nn.ModuleList()
        self.linear = nn.Linear(self.config.embedding_dim, 5)
        self.to(self.device)



    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = F.dropout(h_embedding, p=self.config.dropout)

        out = self.linear(h_embedding)
        #[batch, label_num]
        out = torch.sum(out, dim=1)
        predict = torch.argmax(out, dim=-1)


        return out, predict


    def save_model(self):
        return None

    def load_model(self):
        return None
    def parameters_requires_grads(self):
        return list(filter(lambda p: p.requires_grad, self.parameters()))
#
#
# class RCNN(nn.Module):
#     def __init__(self, embedding_matrix):

# class TextCNN(nn.Module):
#     def __init__(self, embedding_matrix):
#         super(TextCNN, self).__init__()
#         #self.args = args
#
#         label_num = 2 # 标签的个数 2
#         filter_num = 100 # 卷积核的个数 100
#         filter_sizes = [3,4,5] #3 4 5
#
#         # vocab_size = args.vocab_size
#         # embedding_dim = 128  #128
#         embed_size = embedding_matrix.shape[1]
#
#
#         self.embedding = nn.Embedding(max_features, embed_size)
#         self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
#         self.embedding.weight.requires_grad = False
#
#        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
#
#
#         self.convs = nn.ModuleList(
#             [nn.Conv2d(1, filter_num, (fsz, embed_size)) for fsz in filter_sizes])
#         self.dropout = SpatialDropout(0.3)
#         self.linear = nn.Linear(len(filter_sizes)*filter_num, label_num)
#
#     def forward(self, x):
#         # 输入x的维度为(batch_size, max_len), max_len可以通过torchtext设置或自动获取为训练样本的最大=长度
#         x = self.embedding(x) # 经过embedding,x的维度为(batch_size, max_len, embedding_dim)
#
#         # 经过view函数x的维度变为(batch_size, input_chanel=1, w=max_len, h=embedding_dim)
#         x = x.view(x.size(0), 1, x.size(1), self.embed_size)
#
#         # 经过卷积运算,x中每个运算结果维度为(batch_size, out_chanel, w, h=1)
#         x = [F.relu(conv(x)) for conv in self.convs]
#
#         # 经过最大池化层,维度变为(batch_size, out_chanel, w=1, h=1)
#         x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]
#
#         # 将不同卷积核运算结果维度（batch，out_chanel,w,h=1）展平为（batch, outchanel*w*h）
#         x = [x_item.view(x_item.size(0), -1) for x_item in x]
#
#         # 将不同卷积核提取的特征组合起来,维度变为(batch, sum:outchanel*w*h)
#         x = torch.cat(x, 1)
#
#         # dropout层
#         x = self.dropout(x)
#
#         # 全连接层
#         logits = self.linear(x)
#         return logits