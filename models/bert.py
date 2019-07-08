import copy
import math
import json

import numpy as np
import tensorflow as tf

import torch 
import torch.nn as nn
import torch.nn.functional as F

def gelu(x):
    '''https://arxiv.org/abs/1606.08415'''
    cdf = 0.5 * (1.0 + torch.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))))
    return x * cdf

def linear(x):
    return x

def get_activation(act):
    if act == 'linear': return linear
    elif act == 'relu': return F.relu
    elif act == 'gelu': return gelu
    elif act == 'tanh': return F.tanh
    else:
        print (f'Error: Activation not supported: {act}')

class BertConfig(object):
    """Configuration for `BertModel`."""
    def __init__(self,
                vocab_size=1,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=16,
                initializer_range=0.02, 
                json_path=None):

        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.max_position_embeddings = max_position_embeddings

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout = attention_probs_dropout_prob
        self.initializer_range = initializer_range

        print ('Warning: dropout and activation functions should be manually set')
        if json_path is not None:
            self.from_json_file(json_path)

    def from_json_file(self, path):
        with open(path, 'r') as f:
            cfg = json.load(f)
            for key, value in cfg.items():
                setattr(self, key, value)

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
    hidden_size = query.shape[-1]
    seq_len = query.shape[-2]

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(hidden_size)
    # (..., seq_len, seq_len)

    if mask is not None:
        mask = torch.stack([mask] * seq_len, dim=-2).view(scores.shape)
        # (..., seq_len, seq_len)

        '''
        # Guarantee that values only varies on dimension 0 and 3, i.e. mask[i, j1, k1, :] == mask[i, j2, k2, :] for any i, j, k
        for i in range(mask.shape[-4]):
            x = mask[i, 0, 0, :]
            for j in range(mask.shape[-3]):
                for k in range(mask.shape[-2]):
                    for l in range(mask.shape[-1]):
                        
                        assert x[l] == mask[i, j, k, l]
        '''
                
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
        # self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        self.linear_q = nn.Linear(embedding_dim, embedding_dim)
        self.linear_k = nn.Linear(embedding_dim, embedding_dim)
        self.linear_v = nn.Linear(embedding_dim, embedding_dim)
        self.linear_out = nn.Linear(embedding_dim, embedding_dim)
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

        

        batch_size, seq_len, embedding_dim = query.shape
        num_heads = self.num_heads
        
        query, key, value = [linear(x).view(batch_size, seq_len, num_heads, self.hidden_size).transpose(1, 2) for linear, x in zip((self.linear_q, self.linear_k, self.linear_v), (query, key, value))]
        # view: split the last dimension (embedding_dim -> num_heads * hidden_size)
        # transpose: since the next step is feeding to attention(), the last two dimensions of whose input are seq_len and hidden_size 
        # The fancy view and transpose are used to avoid concatenation of heads(see the paper)


        if mask is not None:
            # Same mask for all heads
            mask = torch.stack([mask] * num_heads, dim=-2).view(batch_size, num_heads, seq_len)
            # (batch_size, seq_len) -> (batch_size, num_heads, seq_len)

        # query/key/value: (batch_size, num_heads, seq_len, hidden_size)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x: (batch_size, num_heads, seq_len, hidden_size)

        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * self.hidden_size)
        # (batch_size, seq_len, num_heads * hidden_size = embedding_dim)

        x = self.linear_out(x)

        return x
        # return self.linears[3](x)

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

    def __init__(self, hidden_size, num_heads, feed_forward_hidden_size):
        super(EncoderLayer, self).__init__()

        self.add_and_norm_attention = AddAndNorm(hidden_size)
        self.add_and_norm_feed_forward = AddAndNorm(hidden_size)
        self.multihead_attention = MultiheadAttention(hidden_size, num_heads)

        self.linear_ff = nn.Linear(hidden_size, feed_forward_hidden_size)
        self.linear_out = nn.Linear(feed_forward_hidden_size, hidden_size)
        self.dropout = nn.Dropout()

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, hidden_size)
        # return: (batch_size, seq_len, hidden_size)

        x = self.add_and_norm_attention(x, self.multihead_attention(x, x, x, mask))

        feed_forward = self.linear_out(self.dropout(gelu(self.linear_ff(x))))

        x = self.add_and_norm_feed_forward(x, feed_forward)
        return x

class Encoder(nn.Module):
    '''Transformer Encoder'''
    def __init__(self, hidden_size, num_layers, num_heads, feed_forward_hidden_size):
        super(Encoder, self).__init__()

        encoder_layer = EncoderLayer(hidden_size, num_heads, feed_forward_hidden_size)
        self.layers = clones(encoder_layer, num_layers)

    def forward(self, x, mask=None):

        for layer in self.layers:
            x = layer(x, mask)
        return x

class BertModel(nn.Module):

    def __init__(self, config, tf_checkpoint_path=None):
        '''
        token_type_vocab_size: number of token type(e.g. 3 for [CLS], [SEP], normal word)
        '''
        super(BertModel, self).__init__()
        
        self.hidden_size = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.layer_norm = LayerNorm(config.hidden_size)
        self.layer_norm_and_dropout = nn.Sequential(self.layer_norm, nn.Dropout(config.hidden_dropout_prob))

        self.encoders = Encoder(config.hidden_size, config.num_hidden_layers, config.num_attention_heads, config.intermediate_size)

        # tanh as activation 
        self.pooler = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.Tanh())



        if tf_checkpoint_path is not None:
            self.from_tf_checkpoint(tf_checkpoint_path)

    def forward(self, token_idxs, position_idxs, token_type_idxs, masks=None):
        '''
        Args:
            token_idxs: (batch_size, seq_len)
            position_idxs: (batch_size, seq_len)
            token_type_idxs: (batch_size, seq_len), used for segment embedding, e.g. 0 and 1 for first and second segment(sentence piece) repectively
            masks: (batch_size, seq_len), 0 for masked word(i.e. padding words when training or masks when pretraining), 1 otherwise
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

        sequence_output = self.encoders(x, mask=masks)
        # (batch_size, seq_len, hidden_size)

        first_token_output = sequence_output.narrow(dim=-2, start=0, length=1).squeeze(dim=-2)
        # Select at dimension -2(i.e. dim of seq_len), by range [start=0, start+length=1) (i.e. the first token CLS)] 
        # (batch_size, hidden_size)

        pooled_first_token_output = self.pooler(first_token_output)
        # (batch_size, hidden_size)

        return sequence_output, pooled_first_token_output

    def from_tf_checkpoint(self, path):

        # TODO Check this again
        para_map = {}
        para_map['bert/embeddings/LayerNorm/beta'] = self.layer_norm.b
        para_map['bert/embeddings/LayerNorm/gamma'] = self.layer_norm.g 

        para_map['bert/embeddings/position_embeddings'] = self.position_embedding.weight
        para_map['bert/embeddings/token_type_embeddings'] = self.token_type_embedding.weight
        para_map['bert/embeddings/word_embeddings'] = self.token_embedding.weight

        para_map['bert/pooler/dense/bias'] = self.pooler[0].bias
        para_map['bert/pooler/dense/kernel'] = self.pooler[0].weight

        for i, encoder_layer in enumerate(self.encoders.layers):
            para_map[f'bert/encoder/layer_{i}/attention/self/query/bias'] = encoder_layer.multihead_attention.linear_q.bias
            para_map[f'bert/encoder/layer_{i}/attention/self/query/kernel'] = encoder_layer.multihead_attention.linear_q.weight 
            para_map[f'bert/encoder/layer_{i}/attention/self/key/bias'] = encoder_layer.multihead_attention.linear_k.bias
            para_map[f'bert/encoder/layer_{i}/attention/self/key/kernel'] = encoder_layer.multihead_attention.linear_k.weight 
            para_map[f'bert/encoder/layer_{i}/attention/self/value/bias'] = encoder_layer.multihead_attention.linear_v.bias
            para_map[f'bert/encoder/layer_{i}/attention/self/value/kernel'] = encoder_layer.multihead_attention.linear_v.weight 

            para_map[f'bert/encoder/layer_{i}/attention/output/dense/bias'] = encoder_layer.multihead_attention.linear_out.bias
            para_map[f'bert/encoder/layer_{i}/attention/output/dense/kernel'] = encoder_layer.multihead_attention.linear_out.weight
            para_map[f'bert/encoder/layer_{i}/attention/output/LayerNorm/beta'] = encoder_layer.add_and_norm_attention.norm.b
            para_map[f'bert/encoder/layer_{i}/attention/output/LayerNorm/gamma'] = encoder_layer.add_and_norm_attention.norm.g

            para_map[f'bert/encoder/layer_{i}/intermediate/dense/bias'] = encoder_layer.linear_ff.bias
            para_map[f'bert/encoder/layer_{i}/intermediate/dense/kernel'] = encoder_layer.linear_ff.weight
            para_map[f'bert/encoder/layer_{i}/output/dense/bias'] = encoder_layer.linear_out.bias
            para_map[f'bert/encoder/layer_{i}/output/dense/kernel'] = encoder_layer.linear_out.weight

            para_map[f'bert/encoder/layer_{i}/output/LayerNorm/beta'] = encoder_layer.add_and_norm_feed_forward.norm.b
            para_map[f'bert/encoder/layer_{i}/output/LayerNorm/gamma'] = encoder_layer.add_and_norm_feed_forward.norm.g


        names, arrays = [], []
        parameters = para_map

        for name, shape in tf.train.list_variables(path):

            array = tf.train.load_variable(path, name)
            names.append(name)
            arrays.append(array)

        for name, array in zip(names, arrays):
            chunks = name.split('/')

            if chunks[0] == 'bert':
                # kernel is the weight of nn.Linear, whose shape is the transpose of tf (see pytorch doc for detail)
                if chunks[-1] == 'kernel': array = np.transpose(array)
                try:
                    assert parameters[name].shape == array.shape
                except:
                    print (f'Error: Shape unmatched: {name}, expect {array.shape}, but get {parameters[name].shape}')
                    raise
                parameters[name].data = torch.from_numpy(array)
                # print (f'Load {name}')
            else:
                print (f'Omit {name}')



if __name__ == '__main__':
    
    # Usage
    config = BertConfig(json_path='../../bert_checkpoints/chinese_L-12_H-768_A-12/bert_config.json')

    model = BertModel(config, tf_checkpoint_path='../../bert_checkpoints/chinese_L-12_H-768_A-12/bert_model.ckpt')

    # See the sample above
    token_idxs = torch.LongTensor([[100, 1, 2, 101, 3, 4, 101]])
    position_idxs = torch.LongTensor([[ 0 ,  1 ,  2 ,  3 ,  4 ,  5 ,  6 ]])
    token_type_idxs = torch.LongTensor([[ 0 ,  0 ,  0 ,  0 ,  1 ,  1 ,  1 ]])
    masks = torch.LongTensor([[1, 1, 1, 1, 0, 0, 1]])
    
    sequence_output, pooled_first_token_output = model(token_idxs, position_idxs, token_type_idxs, masks)

    print (sequence_output.shape, pooled_first_token_output.shape)
    

