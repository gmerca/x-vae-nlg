import pandas as pd
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm.auto import tqdm
import numpy as np
from nltk.corpus import stopwords
import torch
from Optimus.code.big_ae import concepts_extractor

data_folder = 'data/worldtree/'

tr = pd.read_json(f'{data_folder}train_set.json').transpose()
va = pd.read_json(f'{data_folder}dev_set.json').transpose()
te = pd.read_json(f'{data_folder}test_set.json').transpose()
ts = pd.read_json(f'{data_folder}table_store.json').transpose()

stopwords = stopwords.words("english")
stopwords.remove('can')
stopwords.append('kind')

# for reproducing results
torch.manual_seed(0)


# ========================================================
#  Create dictionary with topic groups per question
# ========================================================
utils = concepts_extractor.Utils()
utils.init_explanation_bank_lemmatizer()

ex = pd.concat([tr, te, va])

ex['topic'] = ex['topic'].apply(lambda topics_: [topic.strip() for topic in topics_])
topic_list = set([el for sub in ex['topic'].tolist() for el in sub])
topic_dict = dict(zip(list(topic_list), [set() for _ in range(len(topic_list))]))

for i in range(len(ex)):
    q_topics = ex['topic'][i]
    ts_expls = [ts.loc[ts['id'] == k].iloc[0]['explanation'] for k in ex['explanation'][i].keys()]
    facts = [utils.clean_fact_for_overlaps(expl)  for expl in ts_expls] # clean fact
    topics = [utils.recognize_entities(f)  for f in facts] # extract topic with function
    topics = [item for sublist in topics for item in sublist]
    for el in q_topics:
        topic_dict[el].update(topics)


# explanations with at least 2 facts
ex = ex.loc[(ex['explanation'].map(len) > 1)]
e = 'explanation'
main_types = ['CENTRAL', 'LEXGLUE']

# =========  GPT2 settings   ===============================================
ckpt = 'gpt'
tokenizer = GPT2Tokenizer.from_pretrained(ckpt)
model = GPT2LMHeadModel.from_pretrained(ckpt, pad_token_id=tokenizer.eos_token_id)
model_name = 'ft_gpt2'

model.to('cuda')

temperatures=[0.3]
n_sentences_gpt = 6
max_l = 80

# ========================================================

n_found = 0
n_repetitions = 0
n_eval = 0

for temperature in tqdm(temperatures):
    generated_non_stop = []
    generated_s = []
    fact_list = []
    input_tokens = []
    output_tokens = []

    overlap_score_c = []
    excluded_words_c = []
    included_words_c = []
    rows_to_color = [1]
    refs = []

    for i in tqdm(range(len(ex))):
        els =  [ts.loc[ts['id'] == k][['fact', 'table_name']] for k in ex[e][i].keys() if ex[e][i].get(k) in main_types]
        if len(els) == 0: continue
        r_facts,r_types = map(list, zip(*[(el['fact'][0],el['table_name'][0]) for el in els if len(el) > 0]))
        fact_list.append(r_facts)

        qu_topics = ex['topic'][i]
        qu_concepts = set()
        for t in qu_topics:
            for el in topic_dict[t]:
                qu_concepts.add(el)


        fa = pd.DataFrame({'fact': r_facts, 'table_name': r_types})
        same_types = fa.groupby('table_name').filter(lambda x: len(x) > 1)
        f_types = same_types['table_name'].unique()


        df_list = []
        for t in f_types: df_list.append(same_types[same_types['table_name'] == t])

        for j, df in enumerate(df_list):
            f_facts = df['fact'].unique().tolist()
            fact_type = f_types[j]

            for pair_of_sents in zip(*(iter(f_facts),) * 2):
                s1, s2 = f'{pair_of_sents[0]}.', f'{pair_of_sents[-1]}.'
                num_gen_sents = 0
                n_eval += 1
                refs += [r_facts] * 12

                for s_i in [s1, s2]:
                    input_id_i = tokenizer.encode(s_i, return_tensors='pt')
                    input_id_i = input_id_i.to('cuda')

                    output_ids_i = model.generate(
                        input_id_i,
                        do_sample=True,
                        max_length=max_l,
                        top_k=0,
                        temperature=temperature,
                        num_return_sequences=n_sentences_gpt
                    )

                    for ix in range(n_sentences_gpt):
                        new_gen_sent = tokenizer.decode(output_ids_i[ix], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        new_gen_sent = new_gen_sent.replace('\n', 's_e_p_a').replace('.', 's_e_p_a').split(sep='s_e_p_a')
                        new_gen_sent = [s.split(' ') for s in new_gen_sent[1:] if len(s.split(' ')) > 1]
                        new_gen_sent.sort(key=len)
                        m_ix = int(len(new_gen_sent) / 2)
                        if len(new_gen_sent) > 1: new_gen_sent = ' '.join(new_gen_sent[m_ix])
                        else: continue
                        no_stop_sent = ' '.join([el for el in new_gen_sent.split() if el not in stopwords])
                        gen_entities = utils.recognize_entities(no_stop_sent)
                        copy_df = ts.loc[ts['concepts'].apply(lambda x: gen_entities == x if isinstance(x, list) else False )]['fact']
                        if len(copy_df)!= 0:n_found += 1
                        elif no_stop_sent in generated_non_stop: n_repetitions += 1
                        generated_s.append(new_gen_sent)
                        generated_non_stop.append(no_stop_sent)

                        num_gen_sents += 1


                l = rows_to_color[-1]
                rows_to_color.append(l + num_gen_sents + 1)


    dfs_dict = {'sentence': generated_s, 'ref':refs}
    new_df = pd.DataFrame(dfs_dict)
    new_df.to_json(f'{model_name}_{temperature}.json')
