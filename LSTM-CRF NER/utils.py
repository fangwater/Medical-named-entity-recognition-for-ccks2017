import os
import re
import numpy as np
import torch
import torch.autograd as autograd
import codecs
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import json


def create_dico(item_list):
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico

def create_mapping(dico, vocabulary_size=2000):
    sorted_items = sorted(dico.items(),
            key=lambda x: (-x[1], x[0]))[:vocabulary_size]
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item

def read_pre_training(emb_path):
    print('Preparing pre-train dictionary')
    emb_dictionary={}
    for line in codecs.open(emb_path, 'r', 'utf-8'):
        temp = line.split()
        emb_dictionary[temp[0]] = np.asarray(temp[1:], dtype= np.float16)
    return emb_dictionary


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)

def evaluate(model, sentences, dictionaries):
    """
    Evaluate current model
    """
    output_path = 'tmp/evaluate.txt'
    scores_path = 'tmp/score.txt'
    eval_script = './tmp/conlleval'
    score_path = 'tmp/sen_score.txt'
    with codecs.open(output_path, 'w', 'utf8') as f:
       for index in xrange(len(sentences)):
            #input sentence
            input_words = autograd.Variable(torch.LongTensor(sentences[index]['words']))

            #calculate the tag score
            tags,sen_score= model.get_tags(input_words = input_words)

            #tags = model.get_tags(sentence_in)
            # get predict tags
            predict_tags = [dictionaries['id_to_tag'][tag] if (tag in dictionaries['id_to_tag']) else 'START_STOP' for tag in tags]

            # get true tags
            true_tags = [dictionaries['id_to_tag'][tag] for tag in sentences[index]['tags']]

            # write words pos true_tag predict_tag into a file
            for word, pos, true_tag, predict_tag in zip(sentences[index]['str_words'],
                                                        sentences[index]['pos'],
                                                        true_tags, predict_tags):
                f.write('%s %s %s %s\n' % (word, pos ,true_tag, predict_tag))
            f.write('\n')
            with codecs.open(score_path,"a","utf8") as sf:
                sf.write("%s\n %s\n %s\n" % (index,sen_score,len(tags)))

    os.system("%s < %s > %s" % (eval_script, output_path, scores_path))
    eval_lines = [l.rstrip() for l in codecs.open(scores_path, 'r', 'utf8')]
    result={
       'accuracy' : float(eval_lines[1].strip().split()[1][:-2]),
       'precision': float(eval_lines[1].strip().split()[3][:-2]),
       'recall': float(eval_lines[1].strip().split()[5][:-2]),
       'FB1': float(eval_lines[1].strip().split()[7])
    }
    print(eval_lines[1])
    return result

def plot_result(accuracys, precisions, recalls, FB1s):
    plt.figure()
    plt.plot(accuracys,"g-",label="accuracy")
    plt.plot(precisions,"r-.",label="precision")
    plt.plot(recalls,"m-.",label="recalls")
    plt.plot(FB1s,"k-.",label="FB1s")

    plt.xlabel("epoches")
    plt.ylabel("%")
    plt.title("CONLL2000 dataset")

    plt.grid(True)
    plt.legend()
    plt.show()

def save_model_dictionaries(path, model, dictionaries, opts):
    """
    We need to save the mappings if we want to use the model later.
    """
    print("Model is saved in:"+path)
    with open(path+'/dictionaries.dic', 'wb') as f:
        cPickle.dump(dictionaries, f)
    torch.save(model.state_dict(), path+'/model.mdl')
    with open(path+'/parameters.json', 'w') as outfile:
    	json.dump(vars(opts), outfile, sort_keys = True, indent = 4)
    print "save"

def load_parameters(path, opts):
    param_file = os.path.join(path, 'parameters.json')
    with open(param_file, 'r') as file:
        params = json.load(file)
        # Read network architecture parameters from previously saved
        # parameter file.
        opts.clip = params['clip']
        opts.decode_method = params['decode_method']
        opts.embedding_dim = params['embedding_dim']
        opts.freeze = params['freeze']
        opts.hidden_dim = params['hidden_dim']
        opts.loss_function = params['loss_function']
        opts.vocab_size = params['vocab_size']
        opts.zeros = params['zeros']
    return opts
