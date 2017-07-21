import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from CRF import CRF
DROP_OUT = 0.5


class BiLSTM_CRF(nn.Module):

    def __init__(self, parameter):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = parameter['embedding_dim']
        self.hidden_dim = parameter['hidden_dim']
        self.vocab_size = parameter['vocab_size']
        self.tagset_size = parameter['tagset_size']
        self.decode_method = parameter['decode_method']
        self.loss_function = parameter['loss_function']
        self.freeze = parameter['freeze']

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(p=DROP_OUT)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim/2, num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        # We add 2 here, because of START_TAG and STOP_TAG
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size+2)

        self.CRF = CRF(self.tagset_size)


    def init_word_embedding(self, init_matrix):
        self.word_embeds.weight=nn.Parameter(torch.FloatTensor(init_matrix))
        self.word_embeds.weight.requires_grad = not self.freeze


    def _get_lstm_features(self, dropout, **sentence):
        input_words = sentence['input_words']
        embeds = self.word_embeds(input_words)

        if dropout:
            embeds = self.dropout(embeds)

        lstm_out, _ = self.lstm(embeds.view(len(input_words), 1, -1))
        lstm_out = lstm_out.view(len(input_words), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats


    def get_loss(self, tags, **sentence):
        # Get the emission scores from the BiLSTM
        feats = self._get_lstm_features(dropout=False, **sentence)

        if self.loss_function == 'likelihood':
            return self.CRF._get_neg_log_likilihood_loss(feats, tags)
        elif self.loss_function == 'labelwise':
            return self.CRF._get_labelwise_loss(feats, tags)
        else:
            print("ERROR: The parameter of loss function is wrong")

    def forward(self, **sentence): # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        feats = self._get_lstm_features(dropout=False, **sentence)
        # Find the best path, given the features.
        if self.decode_method == 'marginal':
            score, tag_seq = self.CRF._marginal_decode(feats)
        elif self.decode_method == 'viterbi':
            score, tag_seq = self.CRF._viterbi_decode(feats)
        else:
            print("Error wrong decode method")

        return score, tag_seq


    def get_tags(self, **sentence):
        score, tag_seq = self.forward(**sentence)
        return np.asarray(tag_seq).reshape((-1,)),score
