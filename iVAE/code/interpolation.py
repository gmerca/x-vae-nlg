import sys
import os
from preprocess import Indexer
import torch
import argparse
import logging
from data import Dataset
import numpy as np
import random
import pandas as pd
import itertools
import re
from tqdm.auto import tqdm
from nltk.corpus import stopwords
from concepts_extractor import Utils

parser = argparse.ArgumentParser()

run = "worldtree"
model_name = 'ivae'
print(f'Running {model_name} with {run} 0')

data_folder = '../data/worldtree/'
ds = 'wt'

# global parameters
parser.add_argument('--test_file', default=f'../data/{run}/{ds}-test.hdf5')
parser.add_argument('--results_folder_prefix', default=f'output/{ds}_results_')
parser.add_argument('--train_from_epo', default=40, type=int)
parser.add_argument('--seed', default=63, type=int)
parser.add_argument('--log_prefix', default='interpolation')
parser.add_argument('--model', default='mle', type=str, choices=['mle', 'mle_mi'])
parser.add_argument('--num_particles_eval', default=24, type=int)
parser.add_argument('--latent_dim', default=32, type=int)

# use GPU
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--no_gpu', action="store_true")

if sys.argv[1:] == ['0', '0']:
    args = parser.parse_args([])  # run in pycharm console
else:
    args = parser.parse_args()  # run in cmd

# parameters
test_data = Dataset(args.test_file)
test_sents = test_data.batch_size.sum()
vocab_size = int(test_data.vocab_size)
vocab = Indexer()
vocab.load_vocab(f'../data/{run}/{ds}.dict')

print('Test data: %d batches' % len(test_data))
print('Test data: %d sentences' % test_sents)
print('Word vocab size: %d' % vocab_size)

results_folder = args.results_folder_prefix + args.model + '/'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
logging.basicConfig(filename=os.path.join(results_folder, args.log_prefix + '.log'),
                    level=logging.INFO, format='%(asctime)s--- %(message)s')

if not torch.cuda.is_available(): args.no_gpu = True
gpu = not args.no_gpu
if gpu: torch.cuda.set_device(args.gpu)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if gpu: torch.cuda.manual_seed(args.seed)

train_from = results_folder + '%03d.pt' % args.train_from_epo

logging.info('load model from' + train_from)
checkpoint = torch.load(train_from, map_location="cuda:" + str(args.gpu) if gpu else 'cpu')

encoder = checkpoint['encoder']
decoder = checkpoint['decoder']

if gpu:
    encoder = encoder.cuda()
    decoder = decoder.cuda()


def get_text(sents):
    sampled_sents = []
    for i in range(sents.shape[0]):
        decoded_sentence = [vocab.idx2word[int(s)] for s in sents[i, :]]
        sampled_sents.append(decoded_sentence)
    for i, sent in enumerate(sampled_sents):
        logging.info('the %d-th sent: ' % i + ' '.join(sent))


def get_avg_code(encoder_, sents):
    if gpu:
        sents = sents.cuda()
    eps = torch.randn((sents.shape[0], args.latent_dim), device=sents.device)
    z_x, _ = encoder_(sents, eps)
    z_x = z_x.mean(dim=0, keepdim=True).data
    return z_x


def sample_text(decoder_, input_, z, eos):
    sentence = [input_.item()]
    max_index = 0
    input_word = input_
    batch_size, n_sample, _ = z.size()
    seq_len = 1
    z_ = z.expand(batch_size, seq_len, args.latent_dim)
    word_vecs = decoder_.dec_word_vecs(input_word)
    decoder_.h0 = torch.zeros((decoder_.dec_num_layers, word_vecs.size(0), decoder_.dec_h_dim), device=z.device)
    decoder_.c0 = torch.zeros((decoder_.dec_num_layers, word_vecs.size(0), decoder_.dec_h_dim), device=z.device)
    decoder_.h0[-1] = decoder_.latent_hidden_linear(z)
    hidden = None
    while max_index != eos and len(sentence) < 100:
        # (batch_size, seq_len, ni)
        word_embed = decoder_.dec_word_vecs(input_word)
        word_embed = torch.cat((word_embed, z_), -1)
        if len(sentence) == 1:
            output, hidden = decoder_.dec_rnn(word_embed, (decoder_.h0, decoder_.c0))
        else:
            output, hidden = decoder_.dec_rnn(word_embed, hidden)
        preds = decoder_.dec_linear[1:](output.view(word_vecs.size(0) * word_vecs.size(1), -1)).view(-1)
        max_index = torch.argmax(preds).item()
        input_word = torch.ones((), dtype=torch.long).new_tensor([[max_index]])
        if gpu: input_word = input_word.cuda()
        sentence.append(max_index)
    return sentence


def interpolate(s1_enc, s2_enc):
    global n_found
    global n_repetitions
    global refs

    z_x1 = get_avg_code(encoder, s1_enc)
    z_x2 = get_avg_code(encoder, s2_enc)
    z_x = z_x1
    for t_ in torch.arange(0.0, 1.01, 0.1):
        z_x = torch.cat((z_x, z_x1 * (1 - t_) + z_x2 * t_), dim=0)

    logging.info('---------------- Sample sentences: ----------------')
    sampled_sents = []

    for ix in range(len(z_x)):
        z = z_x[ix, :]
        z = z.view(1, 1, -1)

        start = vocab.convert('<s>')
        start_ = torch.ones((), dtype=torch.long).new_tensor([[start]])
        end = vocab.convert('</s>')
        if gpu: start_ = start_.cuda()
        sentence = sample_text(decoder, start_, z, end)
        decoded_sentence = [vocab.idx2word[st] for st in sentence]
        cleaned_sentence = " ".join(decoded_sentence[1:-1]).lower()
        cleaned_sentence = re.sub(r'\s([?,.!"](?:\s|$))', r'\1', cleaned_sentence)
        sampled_sents.append(cleaned_sentence)

    num_gen_sents = 0
    for sent in sampled_sents:

        no_stop_sent = ' '.join([el for el in sent.split() if el not in stopwords.words("english")])

        gen_entities = utils.recognize_entities(no_stop_sent)
        copy_df = ts.loc[ts['concepts'].apply(lambda x: gen_entities == x if isinstance(x, list) else False)]['fact']
        if len(copy_df) != 0: n_found += 1
        elif no_stop_sent in generated_non_stop: n_repetitions += 1
        generated_s.append(sent)
        generated_non_stop.append(no_stop_sent)
        num_gen_sents += 1


# ==================================================
utils = Utils()
utils.init_explanation_bank_lemmatizer()

tr = pd.read_json(f'{data_folder}train_set.json').transpose()
va = pd.read_json(f'{data_folder}dev_set.json').transpose()
te = pd.read_json(f'{data_folder}test_set.json').transpose()
ts = pd.read_json(f'{data_folder}table_store.json').transpose()


# ========================================================
#  Create dictionary with topic groups per question
# ========================================================
ex = pd.concat([tr, te, va])

ex['topic'] = ex['topic'].apply(lambda topics_: [topic.strip() for topic in topics_])

topic_list = set([el for sub in ex['topic'].tolist() for el in sub])

topic_dict = dict(zip(list(topic_list), [set() for _ in range(len(topic_list))]))
main_types = ['CENTRAL', 'LEXGLUE']

for i in range(len(ex)):
    q_topics = ex['topic'][i]
    ts_expls = [ts.loc[ts['id'] == k].iloc[0]['explanation'] for k in ex['explanation'][i].keys()]
    facts = [utils.clean_fact_for_overlaps(expl)  for expl in ts_expls] # clean fact
    topics = [utils.recognize_entities(f)  for f in facts] # extract topic with function
    topics = [item for sublist in topics for item in sublist]
    for el in q_topics:
        topic_dict[el].update(topics)

# ========================================================


# explanations with at least 2 facts
ex = ex.loc[(ex['explanation'].map(len) > 1)]

generated_s = []
generated_non_stop = []
refs = []

n_found = 0
n_repetitions = 0
n_eval = 0

e = 'explanation'

for i in tqdm(range(len(ex))):
    els = [ts.loc[ts['id'] == k][['fact', 'table_name']] for k in ex[e][i].keys() if ex[e][i].get(k) in main_types]
    if len(els) == 0: continue
    r_facts, r_types = map(list, zip(*[(el['fact'][0], el['table_name'][0]) for el in els if len(el) > 0]))
    qu_topics = ex['topic'][i]
    qu_concepts = set()
    for t in qu_topics:
        for el in topic_dict[t]:
            qu_concepts.add(el)

    fa = pd.DataFrame({'fact': r_facts, 'table_name': r_types})
    fa['len'] = fa['fact'].str.split(" ").map(len)
    same_types = fa.groupby(['table_name']).filter(lambda x: len(x) > 1)
    f_types = same_types['table_name'].unique()

    df_list = []
    for t in f_types: df_list.append(same_types[same_types['table_name'] == t])

    for j, df in enumerate(df_list):
        f_facts = df['fact'].unique().tolist()
        fact_type = f_types[j]

        for pair_of_sents in zip(*(iter(f_facts),) * 2):
            s1, s2 = pair_of_sents[0], pair_of_sents[-1]
            se1 = vocab.convert_sequence([vocab.BOS] + s1.strip().split() + [vocab.EOS])
            se2 = vocab.convert_sequence([vocab.BOS] + s2.strip().split() + [vocab.EOS])
            sents1 = torch.from_numpy(np.array([se1] * args.num_particles_eval))
            sents2 = torch.from_numpy(np.array([se2] * args.num_particles_eval))
            n_eval += 1
            refs += [r_facts] * 12
            interpolate(sents1, sents2)

print(f'Repetitions: {n_repetitions}\nFound:{n_found} \nTotal:{n_eval*12}')

dfs_dict = {'sentence': generated_s, 'ref':refs}
new_df = pd.DataFrame(dfs_dict)
new_df.to_json(f'{model_name}_generation.json')
