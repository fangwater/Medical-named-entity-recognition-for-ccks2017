import os
import re
import codecs
from utils import create_dico, create_mapping, zero_digits
from utils import read_pre_training
import numpy as np
import string


def load_sentences(path, zeros):
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        """
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        """
        line = line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences

def word_mapping(sentences,vocabulary_size, pre_train = None):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0] for x in s] for s in sentences]
    dico = create_dico(words)
    word_to_id, id_to_word = create_mapping(dico, vocabulary_size)
    print ("Found %i unique words (%i in total)" %
        (len(dico), sum(len(x) for x in words))
    )

    if pre_train:
        emb_dictionary = read_pre_training(pre_train)
        for word in dico.iterkeys():
        	  if word not in emb_dictionary:
        	  	  dico[word]=0

    dico['<UNK>'] = 10000000
    word_to_id, id_to_word = create_mapping(dico, vocabulary_size)
    return dico, word_to_id, id_to_word


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % (len(dico)))
    return dico, tag_to_id, id_to_tag

def prepare_dataset(sentences, word_to_id, tag_to_id,supervised = True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - tag indexes
    """
    def f(x): return x
    data = []
    for s in sentences:
        str_words = [w[0] for w in s]
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
                 for w in str_words]
        data.append({
            'str_words': str_words,
            'words': words,
        })
        if supervised:
            pos = [w[1] for w in s]
            tags = [tag_to_id[w[-1]] for w in s]
            data[-1]['pos']=pos;
            data[-1]['tags']=tags;
    return data

def prepare_dictionaries(parameters):
    zeros = parameters['zeros']
    train_path = parameters['train']
    dev_path = parameters['development']
    vocabulary_size = parameters['vocab_size']

    # Load sentences
    train_sentences = load_sentences(train_path, zeros)

    if parameters['pre_emb']:
        dev_sentences = load_sentences(dev_path,  zeros)
        sentences = train_sentences + dev_sentences
        dico_words, word_to_id,id_to_word = word_mapping(sentences,
                                   vocabulary_size, parameters['pre_emb'])
    else:
        dico_words, word_to_id, id_to_word = word_mapping(train_sentences,
                                    vocabulary_size, parameters['pre_emb'])
    dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

    dictionaries = {
        'word_to_id': word_to_id,
        'id_to_word': id_to_word,
        'tag_to_id': tag_to_id,
        'id_to_tag': id_to_tag,
    }

    return dictionaries

def load_dataset(parameters, path, dictionaries, supervised = True):
    # Data parameters
    zeros = parameters['zeros']

    # Load sentences
    sentences = load_sentences(path, zeros)
    dataset = prepare_dataset(
        sentences, dictionaries['word_to_id'], dictionaries['tag_to_id'],supervised
    )
    print("%i sentences in %s ."%(len(dataset), path))
    return dataset

def get_word_embedding_matrix(dictionary, pre_train, embedding_dim):
    emb_dictionary = read_pre_training(pre_train)
    dic_size = len(dictionary)
    initial_matrix = np.random.random(size=(dic_size, embedding_dim))
    for word, idx in dictionary.iteritems():
        if word != '<UNK>':
            try:
                initial_matrix[idx] = emb_dictionary[word]
            except:
                initial_matrix[idx] = np.random.random(size = (1,100))[0]

    return initial_matrix
