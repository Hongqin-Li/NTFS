import copy
import math

import torch 
import torch.nn as nn
import torch.nn.functional as F

def clones(module, N):
    "Produce N identical layers"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    '''
    Attention(Q, K, V) = softmax((Q @ K')/sqrt(d_k)) @ V
    where K' is the transpose of K and @ is matrix multiplication
    For example, in self-attention, Q = K = V each of shape (batch_size, seq_len, embedding_dim), then Attention(Q, K, V) is also of shape (batch_size, seq_len, embedding_dim)

    Args:
        query/key/value: (..., seq_len, hidden_size)
        mask: (..., seq_len), 0 for word to be masked(i.e. won't be attented by other words)

    Return:
        attented word representation: (..., seq_len, hidden_size)
        attention matrix: (..., seq_len, seq_len)
    '''
    hidden_size = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(hidden_size)
    # (..., seq_len, seq_len)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        # (..., seq_len, seq_len)

    attn = F.softmax(scores, dim=-1)
    # (..., seq_len, seq_len)

    if dropout is not None:
        attn = dropout(attn)

    return torch.matmul(attn, value), attn


class MultiheadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super(MultiheadAttention, self).__init__()

        assert embedding_dim % num_heads == 0
        # Guarantee embedding_dim = num_heads * hidden_size

        self.hidden_size = embedding_dim // num_heads
        self.num_heads = num_heads

        self.dropout = nn.Dropout(p=dropout)
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        # self.linears[0:3] are weight for query, key and valu, i.e. WQ, WK, WV in the paper
        # self.linears[3] is the output weight, i.e. WO in the paper

    def forward(self, query, key, value, mask=None):
        '''
        Args:
            query/key/value: (..., seq_len, embedding_dim)
            mask: (..., seq_len), 0 for word to be masked(i.e. won't be attented by other words)
        
        Return: 
            attented word representation: (..., seq_len, embedding_dim)
        '''

        
        if mask is not None:
            # Same mask for all heads
            mask = mask.unsqueeze(1)
            # (batch_size, seq_len) -> (batch_size, 1, seq_len)

        batch_size, seq_len, embedding_dim = query.shape
        
        query, key, value = [linear(x).view(batch_size, seq_len, self.num_heads, self.hidden_size).transpose(1, 2) for linear, x in zip(self.linears[:3], (query, key, value))]
        # view: split the last dimension (embedding_dim -> num_heads * hidden_size)
        # transpose: since the next step is feeding to attention(), the last two dimensions of whose input are seq_len and hidden_size 
        # The fancy view and transpose are used to avoid concatenation of heads(see the paper)


        # query/key/value: (batch_size, num_heads, seq_len, hidden_size)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x: (batch_size, num_heads, seq_len, hidden_size)

        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.hidden_size)
        # (batch_size, seq_len, num_heads * hidden_size = embedding_dim)

        return self.linears[3](x)

# See paper "Layer Normalization"
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # x: (..., features)
        # return: (..., features)

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # (..., 1)
        return self.g * (x - mean) / (std + self.eps) + self.b


class AddAndNorm(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(AddAndNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(hidden_size)
        
    def forward(self, x, sublayer_out):
        # x: (..., hidden_size)
        # sublayer_out: (..., hidden_size)
        # return: (..., hidden_size)

        # Different from "The Annotated Transformer"!
        return self.norm(x + self.dropout(sublayer_out))
        
class EncoderLayer(nn.Module):

    def __init__(self, hidden_size, num_heads, feed_forward_hidden_size=2048):
        super(EncoderLayer, self).__init__()

        self.add_and_norm1 = AddAndNorm(hidden_size)
        self.add_and_norm2 = AddAndNorm(hidden_size)
        self.multihead_attention = MultiheadAttention(hidden_size, num_heads)
        # FIXME d_ff = 2048 in the paper maybe too large
        self.feed_forward = nn.Sequential(  nn.Linear(hidden_size, feed_forward_hidden_size), 
                                            nn.ReLU(), 
                                            nn.Dropout(), 
                                            nn.Linear(feed_forward_hidden_size, hidden_size)) 

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, hidden_size)
        # return: (batch_size, seq_len, hidden_size)

        x = self.add_and_norm1(x, self.multihead_attention(x, x, x, mask))
        x = self.add_and_norm2(x, self.feed_forward(x))
        return x

class Encoder(nn.Module):
    '''Transformer Encoder'''
    def __init__(self, hidden_size, num_layers, num_heads=8):
        super(Encoder, self).__init__()

        encoder_layer = EncoderLayer(hidden_size, num_heads)
        self.layers = clones(encoder_layer, num_layers)

    def forward(self, x, mask=None):

        for layer in self.layers:
            x = layer(x, mask)
        return x

class Bert(nn.Module):

    def __init__(self, vocab_size, token_type_vocab_size, max_seq_len=512, hidden_size=768, num_layers=12, num_heads=12, dropout=0.1):
        '''
        token_type_vocab_size: number of token type(e.g. 3 for [CLS], [SEP], normal word)
        '''
        super(Bert, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.token_type_embedding = nn.Embedding(token_type_vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

        self.layer_norm_and_dropout = nn.Sequential(LayerNorm(hidden_size), 
                                                    nn.Dropout(dropout))

        self.encoders = Encoder(hidden_size, num_layers, num_heads)

    def forward(self, token_idxs, position_idxs, token_type_idxs):
        '''
        Args:
            token_idxs: (batch_size, seq_len)
            position_idxs: (batch_size, seq_len)
            token_type_idxs: (batch_size, seq_len), used for segment embedding, e.g. 0 and 1 for first and second segment(sentence piece) repectively
        Return:
            (batch_size, seq_len, hidden_size)
        
        Example:
            Assming our vocabulary is {'Merry': 1, 'Christmas': 2, 'Mr': 3, 'Lawrence': 4, '[CLS]': 100, '[SEP]': 101}, 
            the batch_size is 1, and the input sequence is "[CLS] Merry Christmas [SEP] Mr Lawrence [SEP]"

            token_idxs is    [[100,  1 ,  2 , 101,  3 ,  4 , 101]]
            position_idxs is [[ 0 ,  1 ,  2 ,  3 ,  4 ,  5 ,  6 ]]
            token_type_idxs is  [[ 0 ,  0 ,  0 ,  0 ,  1 ,  1 ,  1 ]] (0 and 1 for first and second segment repectively)
        '''
        x = self.token_embedding(token_idxs) + self.position_embedding(position_idxs) + self.token_type_embedding(token_type_idxs)
        x = self.layer_norm_and_dropout(x)
        x = self.encoders(x)
        return x

if __name__ == '__main__':

    model = Bert(vocab_size=1000, token_type_vocab_size=2, max_seq_len=10, hidden_size=8, num_layers=1, num_heads=2)

    token_idxs = torch.LongTensor([[100, 1, 2, 101, 3, 4, 101]])
    position_idxs = torch.LongTensor([[ 0 ,  1 ,  2 ,  3 ,  4 ,  5 ,  6 ]])
    segment_idxs = torch.LongTensor([[ 0 ,  0 ,  0 ,  0 ,  1 ,  1 ,  1 ]])
    
    out = model(token_idxs, position_idxs, segment_idxs)

    print (out.shape)
    

