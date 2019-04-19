"""To test the model on new sentences"""

import argparse
import numpy as np
import torch
import utils
import model.net as net
import os
from nltk.tokenize import word_tokenize
# import nltk
# nltk.download('punkt')


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/small', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")

# Load the parameters
args = parser.parse_args()
json_path = os.path.join(args.model_dir, 'params.json')
params = utils.Params(json_path)

# loading dataset_params
json_path = os.path.join(args.data_dir, 'dataset_params.json')
assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)
dataset_params = utils.Params(json_path)

# loading vocab (we require this to map words to their indices)
vocab_path = os.path.join(args.data_dir, 'words.txt')
vocab = {}
with open(vocab_path) as f:
    for i, l in enumerate(f.read().splitlines()):
        vocab[l] = i

# setting the indices for UNKnown words and PADding symbols
unk_ind = vocab[dataset_params.unk_word]
pad_ind = vocab[dataset_params.pad_word]

# loading tags (we require this to map tags to their indices)
tags_path = os.path.join(args.data_dir, 'tags.txt')
tag_map = {}
with open(tags_path) as f:
    for i, t in enumerate(f.read().splitlines()):
        tag_map[t] = i

# adding dataset parameters to param (e.g. vocab size, )
params.update(json_path)

model = net.Net(params)

utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

# set model to evaluation mode
model.eval()
# sentence = input("Type a sentence.... ")
sentence = "breaking news"
sen_list = word_tokenize(sentence.lower())

batch_data = pad_ind*np.ones((1, len(sen_list)))


# sen_num_list = []
for i in range(len(sen_list)):
    word = sen_list[i]
    try:
        batch_data[0][i] = vocab[word]
    except:
        batch_data[0][i] = vocab['UNK']
batch_data = torch.LongTensor(batch_data)
# print(batch_data)
outputs = model(batch_data)
#
outputs = np.argmax(outputs.detach(), axis=1)
print(outputs)
