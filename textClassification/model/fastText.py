import torch
import torch.nn as nn
import torch.nn.functional as F


class fastText(nn.Module):

    def __init__(self, config, device, pretrain_embedding=None):
        super(fastText, self).__init__()
        self.config = config
        self.device = device
        if pretrain_embedding is None:
            pretrain_embedding = torch.nn.init.uniform(
                torch.FloatTensor(self.config.embedding_size,
                                  self.config.embedding_dim),
                a=-0.15, b=0.15)


        self.embedding = nn.Embedding(num_embeddings=self.config.embedding_size,
                                      embedding_dim=self.config.embedding_dim)
        self.embedding.weight.data.copy_(pretrain_embedding)
        self.embedding.weight.requires_grad = True

        self.linear = nn.Linear(self.config.embedding_dim, self.config.fast_text_hidden)
        self.linear_out = nn.Linear(self.config.fast_text_hidden, self.config.label_num)

        self.to(self.device)

    def forward(self, x):

        h_embedding = self.embedding(x)

        hidden = torch.mean(h_embedding, dim=1)

        hidden = F.relu(self.linear(hidden))

        out = self.linear_out(hidden)
        predict = torch.argmax(out, dim=-1)

        return out, predict

    def parameters_requires_grads(self):
        return list(filter(lambda p: p.requires_grad, self.parameters()))







