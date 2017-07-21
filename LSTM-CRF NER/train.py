# -*- coding: utf-8 -*-
import optparse
import os
from collections import OrderedDict
from loader import prepare_dictionaries, load_dataset, get_word_embedding_matrix
import LstmCrfModel
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import evaluate, plot_result, save_model_dictionaries, load_parameters
import cPickle
import json
import codecs
np.random.seed(15213)
torch.manual_seed(15213)


optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default="data/train.txt",
    help="Train set location"
)
optparser.add_option(
    "-D", "--dev", default="data/test.txt",
    help="Development dataset"
)
optparser.add_option(
    "-z", "--zeros", default="0",
    type='int', help="Replace digits with 0"
)
optparser.add_option(
    "-p", "--pre_emb", default='embedding/emb.txt',
    help="Location of pretrained embeddings"
)
optparser.add_option(
    "-v", "--vocab_size", default="2000",
    type='int', help="vocab_size"
)
optparser.add_option(
    "-e", "--embedding_dim", default="100",
    type='int', help="words hidden dimension"
)
optparser.add_option(
    "-d", "--hidden_dim", default="200",
    type='int', help="LSTM hidden dimension"
)
optparser.add_option(
    "-t", "--decode_method", default="marginal",
    help="Choose viterbi or marginal to decode the output tag"
)
optparser.add_option(
    "-o", "--loss_function", default="labelwise",
    help="Choose likelihood or labelwise to determine the loss function"
)
optparser.add_option(
    "-c", "--clip", default=5.0,
    help="gradient clipping l2 norm"
)
optparser.add_option(
    "-f", "--freeze", default=False,
    help="Wheter freeze the embedding layer or not"
)
optparser.add_option(
    "-s", "--save", default='model',
    help="Model and dictionareis stored postition"
)
optparser.add_option(
    "--load", default=None,
    help="Load pre-trained Model and dictionaries"
)
opts = optparser.parse_args()[0]

# Parse parameters
Parse_parameters = OrderedDict()
Parse_parameters['zeros'] = opts.zeros == 1
Parse_parameters['pre_emb'] = opts.pre_emb
Parse_parameters['train'] = opts.train
Parse_parameters['development'] = opts.dev
Parse_parameters['vocab_size'] = opts.vocab_size

# Check parameters validity
assert os.path.isfile(opts.train)
assert os.path.isfile(opts.dev)
if opts.pre_emb:
    assert opts.embedding_dim in [50, 100, 200, 300]

# load datasets
if not opts.load:
    dictionaries = prepare_dictionaries(Parse_parameters)
else:
    # load dictionaries
    with open(opts.load+'/dictionaries.dic', 'rb') as f:
        dictionaries = cPickle.load(f)
    # load parameters
    opts = load_parameters(opts.load, opts)


tagset_size = len(dictionaries['tag_to_id'])

train_data = load_dataset(Parse_parameters, opts.train, dictionaries)
dev_data = load_dataset(Parse_parameters, opts.dev, dictionaries)


# Model parameters
Model_parameters = OrderedDict()
Model_parameters['vocab_size'] = opts.vocab_size
Model_parameters['embedding_dim'] = opts.embedding_dim
Model_parameters['hidden_dim'] = opts.hidden_dim
Model_parameters['tagset_size'] = tagset_size
Model_parameters['decode_method'] = opts.decode_method
Model_parameters['loss_function'] = opts.loss_function
Model_parameters['freeze'] = opts.freeze


#model = LstmModel.LSTMTagger(Model_parameters)
model = LstmCrfModel.BiLSTM_CRF(Model_parameters)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
# If using pre-train, we need to initialize word-embedding layer
if opts.pre_emb :
    print("Initialize the word-embedding layer")
    initial_matrix = get_word_embedding_matrix(dictionaries['word_to_id'],
                opts.pre_emb, opts.embedding_dim)
    model.init_word_embedding(initial_matrix)

# Load pre-trained model
if opts.load:
  model.load_state_dict(torch.load(opts.load+'/model.mdl'))
n_epochs = 20 # number of epochs over the training set
Division = 2
accuracys = []
precisions = []
recalls = []
FB1s =[]


for epoch in xrange(n_epochs):
    epoch_costs = []
    print("Starting epoch %i..." % (epoch))
    for i, index in enumerate(np.random.permutation(len(train_data))):
        if i %(len(train_data)/Division) == 0:
            # evaluate
            eval_result = evaluate(model, dev_data, dictionaries)
            accuracys.append(eval_result['accuracy'])
            precisions.append(eval_result['precision'])
            recalls.append(eval_result['recall'])
            FB1s.append(eval_result['FB1'])
            save_model_dictionaries('model', model, dictionaries, opts)

        # Step 1. Remember that Pytorch accumulates gradients.  We need to clear them out
        # before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into Variables
        # of word indices.
        input_words = autograd.Variable(torch.LongTensor(train_data[index]['words']))
        targets = autograd.Variable(torch.LongTensor(train_data[index]['tags']))

        # Step 3. Run our forward pass. We combine this step with get_loss function
        #tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by calling
        loss = model.get_loss(targets, input_words = input_words)

        epoch_costs.append(loss.data.numpy())
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), opts.clip)
        optimizer.step()


    print("Epoch %i, cost average: %f" % (epoch, np.mean(epoch_costs)))
    # Save model and dictionaries

# Final Evaluation after training
eval_result = evaluate(model, dev_data, dictionaries)
accuracys.append(eval_result['accuracy'])
precisions.append(eval_result['precision'])
recalls.append(eval_result['recall'])
FB1s.append(eval_result['FB1'])

# Save model and dictionaries
save_model_dictionaries('model', model, dictionaries, opts)

# Plot Result
print("Plot final result")
plot_result(accuracys, precisions, recalls, FB1s)
"""
def process(path,model,dictionaries):
    fns=[os.path.join(fn) for root,dirs,files in os.walk(path) for fn in files]
    for afile in fns:
        sentences = load_dataset(Parse_parameters,path+"/"+afile,dictionaries)
        with codecs.open("tmp/"+path+"/"+afile,'w','utf8') as f:
            for index in xrange(len(sentences)):
                #input sentence
                input_words = autograd.Variable(torch.LongTensor(sentences[index]['words']))

                #calculate the tag score
                tags,sen_score= model.get_tags(input_words = input_words)

                # get predict tags
                predict_tags = [dictionaries['id_to_tag'][tag] if (tag in dictionaries['id_to_tag']) else 'START_STOP' for tag in tags]
                # write words pos true_tag predict_tag into a file
                for word, pos,predict_tag in zip(sentences[index]['str_words'],
                                                    sentences[index]['pos'],
                                                    predict_tags):
                    f.write('%s\t%s\t%s\n' % (word,pos,predict_tag))
#do result
process("01",model,dictionaries)
process("02",model,dictionaries)
process("04",model,dictionaries)
process("05",model,dictionaries)
"""
