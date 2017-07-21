# SequenceTagging model for ccks test

This is for ccks ner analysis project.

We want to implment LSTM-CRF autoencoder in PyTorch.

Split LSTMCRF into two models. Add CRF module and new loss functions.

The eval_script is the CONLL2000 scripts.

Local result get the :accuracy:  96.13%; precision:  90.73%; recall:  92.22%; FB1:  91.47
in test date.

The word-embedding is in /embedding/emb.txt
