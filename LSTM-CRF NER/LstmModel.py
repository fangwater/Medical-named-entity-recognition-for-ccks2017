import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loader import CAP_DIM


class LSTMTagger(nn.Module):

    def __init__(self, parameter):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = parameter['hidden_dim']

        self.word_embeddings = nn.Embedding(parameter['vocab_size'],
                                            parameter['embedding_dim'])

        self.embedding_dim = parameter['embedding_dim']

        # The LSTM takes word embeddings and captical embedding as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.embedding_dim, parameter['hidden_dim'])

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(parameter['hidden_dim'], parameter['tagset_size'])
        self.hidden = self.init_hidden()
        self.loss_function = nn.NLLLoss()

    def init_word_embedding(self, init_matrix):
        self.word_embeddings.weight=nn.Parameter(torch.FloatTensor(init_matrix))

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.Tensor(1, 1, self.hidden_dim)),
                autograd.Variable(torch.Tensor(1, 1, self.hidden_dim)))

    def forward(self, **sentence):
        input_words = sentence['input_words']
        embeds = self.word_embeddings(input_words)
        lstm_out, self.hidden = self.lstm(embeds.view(len(input_words), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(input_words), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores

    def get_tags(self, **sentence):
        input_words = sentence['input_words']
        tag_scores = self.forward(input_words = input_words)
        _, tags = torch.max(tag_scores, dim=1)
        tags = tags.data.numpy().reshape((-1,))
        return tags

    def get_loss(self, tags, **sentence):
        input_words = sentence['input_words']
        tag_scores = self.forward(input_words = input_words)

        loss = self.loss_function(tag_scores, tags)
        return loss
