import torch 
import tiktoken 
from data_loader import create_dataloader_v1

tokenzier = tiktoken.get_encoding("cl100k_base")

output_dim = 256
vocab_size = tokenzier.n_vocab
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

with open("data/الأربعون النووية.txt", "r") as f:
    raw_text = f.read()

max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)


block_size = max_length
pos_embedding_layer = torch.nn.Embedding(block_size, output_dim)

pos_embeddings = pos_embedding_layer(torch.arange(block_size))
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)