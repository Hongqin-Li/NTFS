import copy
import math
import json

import numpy as np
import tensorflow as tf

import torch 
import torch.nn as nn
import torch.nn.functional as F

from .bert import BertConfig, LayerNorm, EncoderLayer

class CLPSEncoder(nn.Module):
    '''Cross-Layer Parameter Sharing Transformer Encoder'''
    def __init__(self, hidden_size, num_layers, num_heads, feed_forward_hidden_size):
        super(CLPSEncoder, self).__init__()

        self.layer = EncoderLayer(hidden_size, num_heads, feed_forward_hidden_size)
        self.num_layers = num_layers

    def forward(self, x, mask=None):

        # Cross-Layer Parameter Sharing
        for _ in range(self.num_layers):
            x = self.layer(x, mask)
        return x

class AlbertModel(nn.Module):

    def __init__(self, config, tf_checkpoint_path=None):
        '''
        token_type_vocab_size: number of token type(e.g. 3 for [CLS], [SEP], normal word)
        '''
        super(AlbertModel, self).__init__()
        
        self.hidden_size = config.hidden_size

        # factorized embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.token_embedding2 = nn.Linear(config.embedding_size, config.hidden_size, bias=False)

        self.token_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.layer_norm = LayerNorm(config.hidden_size)
        self.layer_norm_and_dropout = nn.Sequential(self.layer_norm, nn.Dropout(config.hidden_dropout_prob))

        self.encoder = CLPSEncoder(config.hidden_size, config.num_hidden_layers, config.num_attention_heads, config.intermediate_size)

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
            (batch_size, seq_len, hidden_size), (batch_size, hidden_size)
        
        Example:
            Assming our vocabulary is {'Merry': 1, 'Christmas': 2, 'Mr': 3, 'Lawrence': 4, '[CLS]': 100, '[SEP]': 101}, 
            the batch_size is 1, and the input sequence is "[CLS] Merry Christmas [SEP] Mr Lawrence [SEP]"

            token_idxs is    [[100,  1 ,  2 , 101,  3 ,  4 , 101]]
            position_idxs is [[ 0 ,  1 ,  2 ,  3 ,  4 ,  5 ,  6 ]]
            token_type_idxs is  [[ 0 ,  0 ,  0 ,  0 ,  1 ,  1 ,  1 ]] (0 and 1 for first and second segment repectively)
        '''
        x = self.token_embedding2(self.token_embedding(token_idxs)) + self.position_embedding(position_idxs) + self.token_type_embedding(token_type_idxs)
        x = self.layer_norm_and_dropout(x)

        sequence_output = self.encoder(x, mask=masks)
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
        para_map['bert/embeddings/word_embeddings_2'] = self.token_embedding2.weight

        para_map['bert/pooler/dense/bias'] = self.pooler[0].bias
        para_map['bert/pooler/dense/kernel'] = self.pooler[0].weight

        encoder_layer = self.encoder.layer

        para_map[f'bert/encoder/layer_shared/attention/self/query/bias'] = encoder_layer.multihead_attention.linear_q.bias
        para_map[f'bert/encoder/layer_shared/attention/self/query/kernel'] = encoder_layer.multihead_attention.linear_q.weight 
        para_map[f'bert/encoder/layer_shared/attention/self/key/bias'] = encoder_layer.multihead_attention.linear_k.bias
        para_map[f'bert/encoder/layer_shared/attention/self/key/kernel'] = encoder_layer.multihead_attention.linear_k.weight 
        para_map[f'bert/encoder/layer_shared/attention/self/value/bias'] = encoder_layer.multihead_attention.linear_v.bias
        para_map[f'bert/encoder/layer_shared/attention/self/value/kernel'] = encoder_layer.multihead_attention.linear_v.weight 

        para_map[f'bert/encoder/layer_shared/attention/output/dense/bias'] = encoder_layer.multihead_attention.linear_out.bias
        para_map[f'bert/encoder/layer_shared/attention/output/dense/kernel'] = encoder_layer.multihead_attention.linear_out.weight
        para_map[f'bert/encoder/layer_shared/attention/output/LayerNorm/beta'] = encoder_layer.add_and_norm_attention.norm.b
        para_map[f'bert/encoder/layer_shared/attention/output/LayerNorm/gamma'] = encoder_layer.add_and_norm_attention.norm.g

        para_map[f'bert/encoder/layer_shared/intermediate/dense/bias'] = encoder_layer.linear_ff.bias
        para_map[f'bert/encoder/layer_shared/intermediate/dense/kernel'] = encoder_layer.linear_ff.weight
        para_map[f'bert/encoder/layer_shared/output/dense/bias'] = encoder_layer.linear_out.bias
        para_map[f'bert/encoder/layer_shared/output/dense/kernel'] = encoder_layer.linear_out.weight

        para_map[f'bert/encoder/layer_shared/output/LayerNorm/beta'] = encoder_layer.add_and_norm_feed_forward.norm.b
        para_map[f'bert/encoder/layer_shared/output/LayerNorm/gamma'] = encoder_layer.add_and_norm_feed_forward.norm.g


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
                if chunks[-1] == 'word_embeddings_2': array = np.transpose(array)
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
    # config = BertConfig(json_path='../../bert_checkpoints/chinese_L-12_H-768_A-12/bert_config.json')
    config = BertConfig(json_path='../checkpoints/albert_tiny_zh/albert_config_tiny.json')
    model = AlbertModel(config, tf_checkpoint_path='../checkpoints/albert_tiny_zh/albert_model.ckpt')

    # See the sample above
    token_idxs = torch.LongTensor([[100, 1, 2, 101, 3, 4, 101]])
    position_idxs = torch.LongTensor([[ 0 ,  1 ,  2 ,  3 ,  4 ,  5 ,  6 ]])
    token_type_idxs = torch.LongTensor([[ 0 ,  0 ,  0 ,  0 ,  1 ,  1 ,  1 ]])
    masks = torch.LongTensor([[1, 1, 1, 1, 0, 0, 1]])
    
    sequence_output, pooled_first_token_output = model(token_idxs, position_idxs, token_type_idxs, masks)

    print (sequence_output.shape, pooled_first_token_output.shape)
    

