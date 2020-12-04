import nltk
from moverscore_v2 import get_idf_dict, word_mover_score
import logging
import transformers
import bert_score
from bert_score import score
from collections import defaultdict
import pandas as pd

# Evaluate text similarity between generated sentence and original explanation
data = pd.read_json('data/results/optimus_generation.json')

# Bleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
sf = SmoothingFunction()
fn = sf.method5
bleu_scores = data.apply(lambda x: sentence_bleu([el.split() for el in x['ref']], x['sentence'].split(), smoothing_function=fn ) , axis=1)
print(round(bleu_scores.mean(),2))

# Meteor
nltk.download('wordnet')
from nltk.translate.meteor_score import meteor_score
meteor_scores = data.apply(lambda x: meteor_score(x['ref'], x['sentence'] ) , axis=1)
print(round(meteor_scores.mean(),2))

# Mover score
idf_dict_hyp = get_idf_dict(data['sentence'])
idf_dict_ref = get_idf_dict(data['ref'])
mover_scores = word_mover_score(data['ref'], data['sentence'], idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=True)
print(round(mover_scores.mean(),2))

# Bert score
single_cands = data['sentence'].tolist()
multi_refs = data['ref'].tolist()
P_mul, R_mul, F_mul = score(single_cands, multi_refs, lang="en", rescale_with_baseline=True)
print(round(F_mul.mean(),2))