import torch
from torch import nn
input_dim = 10
embedding_dim = 2
embedding = nn.Embedding(input_dim, embedding_dim)
err = True
if err:
    #Any input more than input_dim - 1, here input_dim = 10
    #Any input less than zero
    input_to_embed = torch.tensor([10])
else:
    input_to_embed = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
embed = embedding(input_to_embed)
print(embed)