from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
from collections import defaultdict
import json
import random
import os
import spacy
import editdistance
from snlp import StanfordNLP
from tqdm import tqdm
import corenlp as corenlp

CORENLP_LOG = 'corenlp.log'
CORENLP_PORT = 5050
import convert_queries as convert_queries

sNLP = StanfordNLP()

nlp = spacy.load("en_core_web_lg")

dataset_ents = defaultdict(list)

NEARBY_GLOVE_FILE = 'jh_out/SQuAD-nearest100_train.json'
SQUAD_TRAIN_PATH = 'datasetQA/SQuAD/dev.json'
AUG_DATA_DIR = 'SQuAD_addsent'
FILE_NAME = 'dev_SQuAD_attack.json'

with open(NEARBY_GLOVE_FILE) as json_file: 
        nearby_word_dict = json.load(json_file) 


def extract_ents(query):
    """
    Extract entities from quesry
    """
    doc = nlp(query)
    for ent in doc.ents:
        if ent.text.replace("||", "") not in dataset_ents[ent.label_]:
            dataset_ents.setdefault(ent.label_,[]).append(ent.text.replace("||", ""))

def load_squad_train():
    """
    Reads in an EQA data set in format of SQuAD. Returns list of dictionaries of samples.
    """
    datapoints = []
    context_set = {}
    with open(SQUAD_TRAIN_PATH) as f:
        obj = json.load(f)
        for data in obj['data']:
            for paragraph in data['paragraphs']:
                context = paragraph['context']

                # extract ents
                extract_ents(context)

                orig_context_char_length = len(context)
                context = context.lstrip()
                num_head_spaces = orig_context_char_length - len(context)

                if context not in context_set:
                    context_set[context] = len(context_set)
                context_id = context_set[context]

                qas = paragraph['qas']
                for qa in qas:
                    example_id = qa['id']
                    question = qa['question'].strip()
                    if 'plausible_answers' not in list(qa.keys()):
                        answer = qa['answers'][0]
                        answers = qa['answers']
                    else:
                        if len(qa['plausible_answers']) == 0:
                            continue
                        else:
                            answer = qa['plausible_answers'][0]
                            answers = qa['plausible_answers']
                    answer_texts = [answer['text'] for answer in answers]
                    start = answer['answer_start'] - num_head_spaces
                    end = answer['answer_start'] + len(answer['text']) - num_head_spaces

                    # TODO: refactor
                    sents = sent_tokenize(context)
                    context_ = ' '.join(sents)
                    if answer_texts[0] != context_[start:end]:
                        start += 1
                        end += 1
                        if answer_texts[0] != context_[start:end]:
                            start += 1
                            end += 1
                            if answer_texts[0] != context_[start:end]:
                                start += 1
                                end += 1
    
                    datapoints.append({
                        'example_id': example_id,
                        'context': context,
                        'sents': sents,
                        'question': question,
                        'start': start,
                        'end': end,
                        'answer_texts': answer_texts,
                        'context_id': context_id,
                    })
    print(len(datapoints))
    return datapoints

def addsent_attack(query):
    """
    Performs an AddOneSent adversarial attack on given query (antonyms[WordNet] + nearest words[GloVe]).
    """
    attacked_query = []
    orig_query = []

    query_ents = nlp(query).ents
    for ent in query_ents:
        replacement = random.choice(dataset_ents[ent.label_]) # random entity of same label
        query = query.replace(ent.text, replacement)

    parsed = nlp(query.lower())
    # 1. Replace nouns and adjectives w/ antonyms from WordNet
    for token in parsed:
        orig_query.append(token.text)
        antonyms = []
        if token.pos_ not in ('ADJ', 'NOUN', 'INTJ'):
            attacked_query.append(token.text)
        else:   # antonym replacement
            synsets = wn.synsets(token.text)
            for synset in synsets:
                for l in synset.lemmas():
                    if l.antonyms(): 
                        antonyms.append(l.antonyms()[0].name())
            if not antonyms:
                attacked_query.append(token.text)
            else:
                attacked_query.append(' '.join(antonyms[0].split('_')))

        # 2. Change named entities and numbers to nearest word in GloVe word vector space
    ch_query = []
    ents_lst = [e.text for e in parsed.ents]
    for token in attacked_query:
        if token in ents_lst and token in nearby_word_dict and token != "place":
            # replace with closest distance word 
            i = 1
            nearest_list = nearby_word_dict[token]
            nearest = nearest_list[i]["word"]
            while editdistance.eval(nearest, token) < 3 and i < len(nearest_list) - 1:
                i += 1
                nearest = nearby_word_dict[token][i]["word"]
            ch_query.append(nearest)
        else:
            ch_query.append(token)

    ch_query = ' '.join(ch_query)
    print(ch_query)

    if ch_query.lower().replace(" ", "") == query.lower().replace(" ", ""): 
        ch_query = replace_nearby_words(ch_query)

    return ch_query


def addsent_squad(query, answer):
    """
    Attack query and answer then convert back to statament.
    """
    changed_question = addsent_attack(query)
    changed_answer = addsent_attack(answer)
    print(f'orig: {query} {answer}')

    # 1. Did the answer change? 
    if (changed_answer.strip() == answer.lower().strip()):
        changed_answer = replace_nearby_words(changed_answer)

    # 2. Did the question change? 
    if (changed_question.strip() == query.lower().strip()): 
        changed_question = replace_nearby_words(changed_question)
    
    response = sNLP.annotate(changed_question)
    aug, miss = convert_queries.convert(changed_question, changed_answer, response['sentences'][0])
    print(f'aug: {aug}')
    return aug, changed_question
    


def replace_nearby_words(sent):
    changed = []
    parsed = nlp(sent.lower())
    for token in parsed:
        if token.pos_ in ('NOUN', 'VERB', 'NUM', 'PROPN', 'INTJ'):
            ## change
            if token.text in nearby_word_dict:
                i = 0
                nearest_list = nearby_word_dict[token.text]
                nearest = nearest_list[i]["word"]
                while editdistance.eval(nearest, token.text) < 3 and i < len(nearest_list)-1:
                    i += 1
                    nearest = nearby_word_dict[token.text][i]["word"]
                changed.append(nearest)
            elif token.pos_ in ('NOUN', 'PROPN'):
                rand_word = random.choice(list(nearby_word_dict))
                while nlp(rand_word)[0].pos_ not in ('NOUN', 'PROPN'):
                    rand_word = random.choice(list(nearby_word_dict))
                changed.append(rand_word)
            else:
                changed.append(token.text) # should never reach here
        else:
            changed.append(token.text)
    return ' '.join(changed).strip()

def generate_addsent_attacks(train_set, num_attacks):
    attack_examples = []
    for i, example in tqdm(enumerate(train_set[:])):
        attacks = []
        for i in range(num_attacks):
            question = example['question']
            answer = random.choice(example['answer_texts'])
            try:
                attack, question_fake = addsent_squad(question, answer)
            except:
                attack = ''
                question_fake = ''
            attacks.append(attack)

        attack_examples.append({
            'example_id': example['example_id'],
            'question': question,
            'attacks': attacks,
            'question_transform': question_fake
        })
    return attack_examples


# python -m rrtl.attacks.attack_squad
def main():
    train_set = load_squad_train()
    print("Load data done!")
    with corenlp.CoreNLPServer(port=CORENLP_PORT, logfile=CORENLP_LOG) as server:
        client = corenlp.CoreNLPClient(port=CORENLP_PORT)
        attacks = generate_addsent_attacks(train_set, 1)
    with open(os.path.join(AUG_DATA_DIR, FILE_NAME), 'w') as f:
        json.dump(attacks, f)
    print(f'Augmented input data saved under: {AUG_DATA_DIR}')


if __name__ == "__main__":
    main()