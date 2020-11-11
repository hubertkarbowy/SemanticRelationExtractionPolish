import re
import os, copy, pickle
import nltk
import itertools, random
from itertools import combinations
from collections import Counter
from bpemb import BPEmb
from transformers import BertTokenizer
from bs4 import BeautifulSoup
from readers_kpwr import *

def entity_id_sorter(x):
    if "." in x:
        x_index = int(x.split(".")[1])
        return int(x_index)
    else:
        return x

class WrappedTokenizer:
    """Tokenizes BPE, natively and for BERT (no special tokens added)"""
    def __init__(self, *, tokenizer_config):
        self.iface = tokenizer_config['iface']
        self.lang = tokenizer_config['lang']
        self.special_token_ids = {}
        self.tokenizer_config = tokenizer_config
        if self.iface == 'bpemb':
            bpemb_object  = BPEmb(lang=tokenizer_config['lang'], dim=tokenizer_config['dim'],
                                  vs=tokenizer_config['vs'])
            last_index = len(bpemb_object.words)
            if self.tokenizer_config.get('add_positional_tokens') is not None:
                if self.tokenizer_config['add_positional_tokens'] == "scheme_1":
                    self.ees_map = {'e1_beg': '<e1>', 'e1_end': '</e1>', 'e2_beg': '<e2>', 'e2_end': '</e2>'}
                    self.special_token_ids = {'<e1>': last_index, '</e1>': last_index+1,
                                              '<e2>': last_index+2, '</e2>': last_index+3}
                    bpemb_object.words.extend(['<e1>', '</e1>', '<e2>', '</e2>'])
                else:
                    raise ValueError(f"Unknown positional tokens scheme {self.tokenizer_config['add_positional_tokens']}")
            self.tokenizer_obj = bpemb_object
        elif self.iface == 'nltk':
            self.tokenizer_obj = nltk
        elif self.iface == 'transformers':
            self.tokenizer_obj = BertTokenizer.from_pretrained(self.tokenizer_config['pretrained'])

    def tokenize(self, doc, **kwargs):
        if self.iface == 'nltk':
            if kwargs.get('also_return_indices') is True:
                raise ValueError("Unconstrained vocabulary for NLTK word-based tokenization")
            return self.tokenizer_obj.word_tokenize(doc, language=self.lang)
        elif self.iface == 'transformers':
            retval = {}
            tmp = self.tokenizer_obj.encode_plus(doc, add_special_tokens=False, **kwargs)
            toks = self.tokenizer_obj.convert_ids_to_tokens(tmp['input_ids'])
            retval = {
                    'tokens': toks,
                    'token_ids': tmp['input_ids'],
                    }
            return retval
        elif self.iface == 'bpemb':
            retval = {
                    'tokens': self.tokenizer_obj.encode(doc),
                    'token_ids': self.tokenizer_obj.encode_ids(doc)
                    }
            # BPEmb adds a whitespace in front of every first token, resulting in '▁.' and '▁,' for 
            # documents containing just punctuation. This is a quick fix.
            if retval['tokens'] == ["▁."]:
                try:
                    retval['token_ids'] = [self.tokenizer_obj.words.index(".")]
                    retval['tokens'] = ["."]
                except:
                    pass # only fix punctuation if a "." subword is available
            if retval['tokens'] == ["▁,"]:
                try:
                    retval['token_ids'] = [self.tokenizer_obj.words.index(",")]
                    retval['tokens'] = [","]
                except:
                    pass
            if kwargs.get('enclose_e1') is True:
                retval['tokens'] = [self.ees_map['e1_beg']] + retval['tokens'] + [self.ees_map['e1_end']]
                retval['token_ids'] = [self.special_token_ids[self.ees_map['e1_beg']]] + retval['token_ids'] + [self.special_token_ids[self.ees_map['e1_end']]]
            if kwargs.get('enclose_e2') is True:
                retval['tokens'] = [self.ees_map['e2_beg']] + retval['tokens'] + [self.ees_map['e2_end']]
                retval['token_ids'] = [self.special_token_ids[self.ees_map['e2_beg']]] + retval['token_ids'] + [self.special_token_ids[self.ees_map['e2_end']]]
            return retval
        else:
            raise ValueError("Boo!")

    def detokenize(self, tokens, **kwargs):
        if self.iface == 'nltk':
            return " ".join(tokens)
        elif self.iface == 'transformers':
            raise NotImplementedError("Jeszcze nie")
        elif self.iface == 'bpemb':
            return self.tokenizer_obj.decode(tokens, **kwargs)
        else:
            raise NotImplementedError(f"Unknown iface {self.iface}!")

    def convert_ids_to_tokens(self, token_ids, **kwargs):
        if self.iface == 'nltk':
            return token_ids
        elif self.iface == 'transformers':
            raise NotImplementedError("Jeszcze nie")
        elif self.iface == 'bpemb':
            return [self.tokenizer_obj.words[token_id] for token_id in token_ids]
        else:
            raise NotImplementedError(f"Unknown iface {self.iface}!")

def calculate_positional_token_offsets(*, entity_labels, from_label_pure, to_label_pure, entity_encoding_scheme):
    """
    Given a list of `entity_labels` assigned to each token entities, `from` and `to` labels like 'city_nam-1'
    and 'country_nam-2' and an encoding scheme,
    calculates indices where <e1>, </e1>, <e2> and </e2> should be inserted.

    The tokens should be inserted as follows: <e1> before 'e1_beg', </e1> after 'e1_end',
                                              <e2> before 'e2_beg', </e2> after 'e2_end'.

    """
    insertable_positions = {'e1_beg': None, 'e1_end': None, 'e2_beg': None, 'e2_beg': None}
    if entity_encoding_scheme is None:
        insertable_positions['e1_beg'] = entity_labels.index(from_label_pure)
        insertable_positions['e2_beg'] = entity_labels.index(to_label_pure)
        insertable_positions['e1_end'] = max([idx for idx, val in enumerate(entity_labels) if val == from_label_pure])
        insertable_positions['e2_end'] = max([idx for idx, val in enumerate(entity_labels) if val == to_label_pure])
    elif entity_encoding_scheme == 'iob':
        print(entity_labels)
        insertable_positions['e1_beg'] = entity_labels.index(f"B-{from_label_pure}")
        insertable_positions['e2_beg'] = entity_labels.index(f"B-{to_label_pure}")
        insertable_positions['e1_end'] = max([idx for idx, val in enumerate(entity_labels) if val in [f"B-{from_label_pure}", f"I-{from_label_pure}"]])
        insertable_positions['e2_end'] = max([idx for idx, val in enumerate(entity_labels) if val in [f"B-{to_label_pure}", f"I-{to_label_pure}"]])
    else:
        raise ValueError(f"Unknown entity encoding scheme {entity_encoding_scheme}")
    if insertable_positions['e2_beg'] < insertable_positions['e1_beg']: # swap e1 and e2
        tmp_e1_beg = insertable_positions['e1_beg']
        tmp_e1_end = insertable_positions['e1_end']
        insertable_positions['e1_beg'] = insertable_positions['e2_beg']
        insertable_positions['e1_end'] = insertable_positions['e2_end']
        insertable_positions['e2_beg'] = tmp_e1_beg
        insertable_positions['e2_end'] = tmp_e1_end
    return insertable_positions

def insert_positional_tokens(*, tokens, token_ids, entities, entity_ids, positions, entity_encoding_scheme, \
                             labels_map, positional_tokens, special_token_ids):
    if entity_encoding_scheme not in [None, "iob"]:
        raise ValueError(f"Unknown entity encoding scheme {entity_encoding_scheme}")
    if positional_tokens == 'scheme_1':
        e1_beg = "<e1>"; e1_end = "</e1>"; e2_beg = "<e2>"; e2_end = "</e2>"
    else:
        raise ValueError(f"Unknown positional tokens scheme {positional_tokens}")

    ############## </e2> ############
    tokens.insert(positions['e2_end'] + 1, e2_end)
    token_ids.insert(positions['e2_end'] + 1, special_token_ids[e2_end])
    if entity_encoding_scheme == None:
        entities.insert(positions['e2_end'] + 1, entities[positions['e2_end']])
        entity_ids.insert(positions['e2_end'] + 1, entity_ids[positions['e2_end']])
    elif entity_encoding_scheme == 'iob':
        expected_end_label = re.sub("^B-", "I-", entities[positions['e2_end']]) # B-city_nam-1 => I-city_nam-1 if singleton
        entities[positions['e2_end']] = expected_end_label
        entity_ids[positions['e2_end']] = get_kpwr_entity_id_from_indexed_label(expected_end_label, labels_map, entity_encoding_scheme)
        entities.insert(positions['e2_end'] + 1, entities[positions['e2_end']])
        entity_ids.insert(positions['e2_end'] + 1, entity_ids[positions['e2_end']])
    else:
        raise ValueError(f"Unknown ees {entitity_encoding_scheme}")

    ############## <e2> ############
    tokens.insert(positions['e2_beg'], e2_beg)
    token_ids.insert(positions['e2_beg'], special_token_ids[e2_beg])
    if entity_encoding_scheme is None:
        entities.insert(positions['e2_beg'], entities[positions['e2_beg']])
        entity_ids.insert(positions['e2_beg'], entity_ids[positions['e2_beg']])
    elif entity_encoding_scheme == 'iob':
        i_beg_label = re.sub("^B-", "I-", entities[positions['e2_beg']]) # B-city_nam-1 => I-city_nam-1 before <e1>
        b_beg_label = re.sub("^I-", "B-", i_beg_label) # I-city_nam-1 => B-city_nam-1 for <e1>
        entities[positions['e2_beg']] = i_beg_label
        entity_ids[positions['e2_beg']] = get_kpwr_entity_id_from_indexed_label(i_beg_label, labels_map, entity_encoding_scheme)
        entities.insert(positions['e2_beg'], b_beg_label)
        entity_ids.insert(positions['e2_beg'], get_kpwr_entity_id_from_indexed_label(b_beg_label, labels_map, entity_encoding_scheme))
    else:
        raise ValueError(f"Unknown ees {entitity_encoding_scheme}")

    ############## </e1> ############
    tokens.insert(positions['e1_end'] + 1, e1_end)
    token_ids.insert(positions['e1_end'] + 1, special_token_ids[e1_end])
    if entity_encoding_scheme == None:
        entities.insert(positions['e1_end'] + 1, entities[positions['e1_end']])
        entity_ids.insert(positions['e1_end'] + 1, entity_ids[positions['e1_end']])
    elif entity_encoding_scheme == 'iob':
        expected_end_label = re.sub("^B-", "I-", entities[positions['e1_end']]) # B-city_nam-1 => I-city_nam-1 if singleton
        entities[positions['e1_end']] = expected_end_label
        entity_ids[positions['e1_end']] = get_kpwr_entity_id_from_indexed_label(expected_end_label, labels_map, entity_encoding_scheme)
        entities.insert(positions['e1_end'] + 1, entities[positions['e1_end']])
        entity_ids.insert(positions['e1_end'] + 1, entity_ids[positions['e1_end']])
    else:
        raise ValueError(f"Unknown ees {entitity_encoding_scheme}")

    ############## <e1> ############
    tokens.insert(positions['e1_beg'], e1_beg)
    token_ids.insert(positions['e1_beg'], special_token_ids[e1_beg])
    if entity_encoding_scheme is None:
        entities.insert(positions['e1_beg'], entities[positions['e1_beg']])
        entity_ids.insert(positions['e1_beg'], entity_ids[positions['e1_beg']])
    elif entity_encoding_scheme == 'iob':
        i_beg_label = re.sub("^B-", "I-", entities[positions['e1_beg']]) # B-city_nam-1 => I-city_nam-1 before <e1>
        b_beg_label = re.sub("^I-", "B-", i_beg_label) # I-city_nam-1 => B-city_nam-1 for <e1>
        entities[positions['e1_beg']] = i_beg_label
        entity_ids[positions['e1_beg']] = get_kpwr_entity_id_from_indexed_label(i_beg_label, labels_map, entity_encoding_scheme)
        entities.insert(positions['e1_beg'], b_beg_label)
        entity_ids.insert(positions['e1_beg'], get_kpwr_entity_id_from_indexed_label(b_beg_label, labels_map, entity_encoding_scheme))
    else:
        raise ValueError(f"Unknown ees {entitity_encoding_scheme}")
    inserted_locations = {"e1_beg": tokens.index(e1_beg), \
                          "e1_end": tokens.index(e1_end), \
                          "e2_beg": tokens.index(e2_beg), \
                          "e2_end": tokens.index(e2_end)}
    return tokens, token_ids, entities, entity_ids, inserted_locations

# TODO - wersje tokenizacji:
# 1) This is <e1>aaaaa</e1> for <e2>bbbbbb</e2>
#    Entity-Origin(e1,e2) / Member-Collection(e2,e1) / ------

def tokenize_encoded_xml(*, doc_id, doc_text, tokenizer_obj, lang='english',
                        entity_encoding_scheme=None, use_entity_classes=False,
                        sentence_tokenize=True,
                        raw_relations=None, positional_tokens=None,
                        add_no_relations=False,
                        retain_natural_no_rels=True): # WARNING WARNING WARNING: raw_relations gets mutated!
    """ `doc_text` should look like this:
        An extension to the <entity id="P83-1003.1">GPSG grammatical formalism</entity> is proposed,
        allowing <entity id="P83-1003.2">non-terminals</entity> to consist of finite sequences of...

        It's best not to have dangling punctuation on the edges of <entity>, e.g. this
        is known to be problematic: <entity id="zzz">Hello!</entity> or even worse
        <entity id="zzz"> Hello !</entity>
    """
    # 0. tokens = []; entities = []
    # 1. Find offsets of all <entity> ... </entity> tags
    # 2. i=0; j=0
    # 3. For each entity_tag:
    #         - j=entity_tag.begin
    #         - tokenize everything from i to j: add to tokens, add False to entities
    #         - tokenize everything from j to entity_tag.end, add True to entities
    #         - i=entity_tag.end
    # 4. j = endpos
    #    tokenize from i to j, add False to entities
    #    ex)
    sents = nltk.sent_tokenize(doc_text) if sentence_tokenize is True else [doc_text] 
    # regexp = '<entity id\b?="(.*?)">(.*?)</entity>'
    regexp = '<entity id\b?="(.*?)"( category="(.*?)")?>(.*?)</entity>' 
    tokens = []; token_ids = []; entities = []; entity_ids = []; annotated_relations = []
    potential_rels = raw_relations.get(doc_id) or {}
    for sent in sents:
        num_inserted_sents = 0
        i=0; j=0; curr_tokens = []; curr_token_ids = []; curr_entities = []; curr_entity_ids = []; curr_relations = []
        uniq_entity_ids = set()
        ent_iter = re.finditer(regexp, sent)
        for entity in ent_iter:
            j = entity_beg = entity.span()[0]; entity_end = entity.span()[1]
            entity_id = entity.groups()[0]   
            uniq_entity_ids.add(entity_id)
            if use_entity_classes is False:
                entity_class = "ENT"
            else:
                entity_class = entity.groups()[2]
                # raise NotImplementedError("Please implement entity classes!")
            res = tokenizer_obj.tokenize(sent[i:j])
            curr_tokens.extend(res['tokens'])
            curr_token_ids.extend(res['token_ids'])
            curr_entities.extend([None]*len(res['tokens']))
            curr_entity_ids.extend([None]*len(res['tokens']))

            entity_content = entity.groups()[3]
            res = tokenizer_obj.tokenize(entity_content)
            curr_tokens.extend(res['tokens'])
            curr_token_ids.extend(res['token_ids'])
            if entity_encoding_scheme is None:
                curr_entities.extend([entity_class]*len(res['tokens']))
                curr_entity_ids.extend([entity_id]*len(res['tokens']))
            elif entity_encoding_scheme == 'iob':
                if len(res['tokens']) == 1:
                    curr_entities.extend([f"B-{entity_class}"])
                    curr_entity_ids.extend([entity_id])
                else:
                    iob_ents = [f"I-{entity_class}"]*len(res['tokens'])
                    iob_ents[0] = f"B-{entity_class}"
                    curr_entities.extend(iob_ents)
                    curr_entity_ids.extend([entity_id]*len(iob_ents))
            i = entity_end
        res = tokenizer_obj.tokenize(sent[i:])
        curr_tokens.extend(res['tokens'])
        curr_token_ids.extend(res['token_ids'])
        curr_entities.extend([None]*len(res['tokens']))
        curr_entity_ids.extend([None]*len(res['tokens']))
        # Add "NO_RELATION" between unmarked entities:
        if add_no_relations:
            if 1 < len(uniq_entity_ids) < 15:
                candidate_relation_pairs = set(combinations(sorted(uniq_entity_ids, key=entity_id_sorter), 2))
                undefined_pairs = candidate_relation_pairs - set(potential_rels.keys())
                for undefined_pair in undefined_pairs:
                    potential_rels[undefined_pair] = "NO_RELATION"
                # print(f">>>> UNDEFINED RELATIONS FOR DOCUMENT {doc_id}: {undefined_pairs} <<<<<")
            else:
                pass
                # candidate_relation_pairs = {}
        ############# STAD maglowanie encji parami.... - TYLKO DLA TOKENOW POZYCYJNYCH, DLA INNYCH NAPISAC COS INNEGO ###################
        for relevant_entity_tuple, relation_class in potential_rels.items():
            first_id = relevant_entity_tuple[0]
            second_id = relevant_entity_tuple[1]
            # print(f"Curr entity ids {curr_entity_ids}")
            if first_id in curr_entity_ids and second_id in curr_entity_ids:
                curtoks_copy = copy.deepcopy(curr_tokens); curtokids_copy = copy.deepcopy(curr_token_ids);
                curent_copy = copy.deepcopy(curr_entities); curentids_copy = copy.deepcopy(curr_entity_ids);
                first_id_begin = curentids_copy.index(first_id)
                first_id_end = len(curentids_copy) - curentids_copy[::-1].index(first_id) - 1
                second_id_begin = curentids_copy.index(second_id)
                second_id_end = len(curentids_copy) - curentids_copy[::-1].index(second_id) - 1

                # Wkładamy tylko jedną relację do zdania.
                entity_class_first_marker = curent_copy[second_id_begin]; # f"B-{entity_class}" if entity_encoding_scheme == "iob" else entity_class
                # entity_class_end = curent_copy[second_id_end]; # f"I-{entity_class}" if entity_encoding_scheme == "iob" else entity_class
                if entity_encoding_scheme == "iob":
                    generic_entity_class = entity_class_first_marker.strip("B-").strip("I-")
                    inside_entity_class = "I-" + generic_entity_class
                    begin_entity_class = "B-" + generic_entity_class
                else:
                    generic_entity_class = entity_class_first_marker
                    inside_entity_class = entity_class_first_marker
                    begin_entity_class = entity_class_first_marker
                curent_copy[second_id_begin] = curent_copy[second_id_end] = inside_entity_class # B-ENT ==> I-ENT
                curtoks_copy = curtoks_copy[0:second_id_begin] + ['<e2>'] + curtoks_copy[second_id_begin:second_id_end+1] + ['</e2>'] + curtoks_copy[second_id_end+1:]
                curtokids_copy = curtokids_copy[0:second_id_begin] + [tokenizer_obj.special_token_ids['<e2>']] + curtokids_copy[second_id_begin:second_id_end+1] \
                                                                   + [tokenizer_obj.special_token_ids['</e2>']] + curtokids_copy[second_id_end+1:]
                curent_copy = curent_copy[0:second_id_begin] + [begin_entity_class] + curent_copy[second_id_begin:second_id_end+1] \
                                                             + [inside_entity_class] + curent_copy[second_id_end+1:]
                curentids_copy = curentids_copy[0:second_id_begin] + [second_id] + curentids_copy[second_id_begin:second_id_end+1] \
                                                                   + [second_id] + curentids_copy[second_id_end+1:]

                entity_class_first_marker = curent_copy[first_id_begin]; # f"B-{entity_class}" if entity_encoding_scheme == "iob" else entity_class
                # entity_class_end = curent_copy[second_id_end]; # f"I-{entity_class}" if entity_encoding_scheme == "iob" else entity_class
                if entity_encoding_scheme == "iob":
                    generic_entity_class = entity_class_first_marker.strip("B-").strip("I-")
                    inside_entity_class = "I-" + generic_entity_class
                    begin_entity_class = "B-" + generic_entity_class
                else:
                    generic_entity_class = entity_class_first_marker
                    inside_entity_class = entity_class_first_marker
                    begin_entity_class = entity_class_first_marker

                curent_copy[first_id_begin] = curent_copy[first_id_end] = inside_entity_class # B-ENT ==> I-ENT
                curtoks_copy = curtoks_copy[0:first_id_begin] + ['<e1>'] + curtoks_copy[first_id_begin:first_id_end+1] + ['</e1>'] + curtoks_copy[first_id_end+1:] # FIXME: nie tylko scheme_1
                curtokids_copy = curtokids_copy[0:first_id_begin] + [tokenizer_obj.special_token_ids['<e1>']] + curtokids_copy[first_id_begin:first_id_end+1] \
                                                                   + [tokenizer_obj.special_token_ids['</e1>']] + curtokids_copy[first_id_end+1:]
                curent_copy = curent_copy[0:first_id_begin] + [begin_entity_class] + curent_copy[first_id_begin:first_id_end+1] \
                                                            + [inside_entity_class] + curent_copy[first_id_end+1:]
                curentids_copy = curentids_copy[0:first_id_begin] + [first_id] + curentids_copy[first_id_begin:first_id_end+1] \
                                                                   + [first_id] + curentids_copy[first_id_end+1:]
                tokens.append(curtoks_copy)
                token_ids.append(curtokids_copy)
                entities.append(curent_copy)
                entity_ids.append(curentids_copy)
                if all([marker in curtoks_copy for marker in ['<e1>', '</e1>', '<e2>', '</e2>']]):
                    relation_info = {'e1_beg': curtoks_copy.index('<e1>'), 'e1_end': curtoks_copy.index('</e1>'),
                                     'e2_beg': curtoks_copy.index('<e2>'), 'e2_end': curtoks_copy.index('</e2>'),
                                     'relation_class': relation_class, 'is_reversed': 'FIXME'}
                else:
                    relation_info = {}
                annotated_relations.append(relation_info)
                num_inserted_sents += 1
            else:
                continue
        # if num_inserted_sents == 0:
        #     tokens.append(curr_tokens)
        #     token_ids.append(curr_token_ids)
        #     entities.append(curr_entities)
        #     entity_ids.append(curr_entity_ids)
        #     if retain_natural_no_rels is True:
        #         annotated_relations.append({'comment': 'global', 'relation_class': 'NO_RELATION'})
        #     else:
        #         annotated_relations.append({})
        if num_inserted_sents == 0:
            if retain_natural_no_rels is True:
                annotated_relations.append({'comment': 'global', 'relation_class': 'NO_RELATION'})
                tokens.append(curr_tokens)
                token_ids.append(curr_token_ids)
                entities.append(curr_entities)
                entity_ids.append(curr_entity_ids)
            else:
                print(f"Skipping sentence {sent[0:80]}... - no rels")
    if sentence_tokenize is True:
        return tokens, token_ids, entities, entity_ids, annotated_relations
    else:
        return tokens[0], token_ids[0], entities[0], entity_ids[0], annotated_relations[0]

def tokenize_from_kpwr(*, doc_id, doc_text, tokenizer_obj, lang='polish',
                          entity_encoding_scheme=None, entity_labels_map, relations_map, sentence_tokenize=True,
                          raw_relations=None, positional_tokens=None,
                          add_no_relations=False, retain_natural_no_rels=True):
    """ Returns:
        "tokens":    ['martin', 'luther', 'day'],
        "token_ids": [120, 355],
        "multi_entities_with_sentence_index": [["B-firstname_nam-1", "B-name_nam-1"], ["I-name_nam-1"], ["O"]],
        "multi_entity_ids": [[10, 12], [13], [0]],
        "annotated_relations_all": [] # empty!!! This will be generated by a different function
        OR:
        "annotated_relations_all": [{'begin_e1': 10, 'end_e1': 12, 'entity_class_e1': 'facility_nam-1',
                                     'begin_e2': 20, 'end_e2': 21, 'entity_class_e2': 'city_nam-1',
                                     'relation_class': 'PART-WHOLE', is_reversed: False}]
    """
    if sentence_tokenize is False:
        raise ValueError("KPWr corpus reader operates on a sentence level")
    soup = BeautifulSoup(doc_text, "xml")
    msents = soup.find_all('sentence')
    mrels = BeautifulSoup(raw_relations, "lxml")
    tokens = []; token_ids = []; multientities = []; multientity_ids = []; annotated_relations = []
    # potential_rels = raw_relations.get(doc_id) or {}
    for sent in msents:
        num_inserted_sents = 0
        curr_tokens = []; curr_token_ids = []; curr_multientities = []; curr_multientity_ids = []; curr_relations = []
        running_entities = set() # entity annotations seen on the previous token (for IOB)
        orig_tokens = sent.find_all('tok')
        for orig_token in orig_tokens:
            res = tokenizer_obj.tokenize(orig_token.orth.text) # straszny kod... opisac co to wlasciwie robi
            subwords = res['tokens']
            subword_ids = res['token_ids']
            subword_multientities = []
            subword_multientity_ids = []
            relevant_ann_tags = orig_token.find_all(kpwr_is_running_entity) # nieoczywistość tu się może dziać.
            heretag_names = set([t['chan']+"-"+t.text for t in relevant_ann_tags])
            tags_which_begin_here = heretag_names - running_entities
            tags_which_continue_here = heretag_names.intersection(running_entities)
            if entity_encoding_scheme == 'iob':
                # tmp_subword_tags = []; tmp_subword_tag_ids = []
                for subword_idx in range(len(subwords)):
                    tmp_sub_subword_tags = []; tmp_sub_subword_tag_ids = []
                    if subword_idx == 0:
                        for begin_tag in tags_which_begin_here:
                            # readable_label = f"B-{ann_tag['chan']}-{ann_tag.text}"
                            readable_label = f"B-{begin_tag}" # B-person_nam-1
                            label_id = get_kpwr_entity_id_from_indexed_label(readable_label,
                                                                             entity_labels_map,
                                                                             entity_encoding_scheme=entity_encoding_scheme)
                            tmp_sub_subword_tags.append(readable_label)
                            tmp_sub_subword_tag_ids.append(label_id)
                        for continue_tag in tags_which_continue_here:
                            # readable_label = f"I-{ann_tag['chan']}-{ann_tag.text}"
                            readable_label = f"I-{continue_tag}" # I-person_nam-1
                            label_id = get_kpwr_entity_id_from_indexed_label(readable_label,
                                                                             entity_labels_map,
                                                                             entity_encoding_scheme=entity_encoding_scheme)
                            tmp_sub_subword_tags.append(readable_label)
                            tmp_sub_subword_tag_ids.append(label_id)
                    else:
                        for any_tag in tags_which_begin_here.union(tags_which_continue_here):
                            # readable_label = f"I-{ann_tag['chan']}-{ann_tag.text}"
                            readable_label = f"I-{any_tag}"
                            label_id = get_kpwr_entity_id_from_indexed_label(readable_label,
                                                                             entity_labels_map,
                                                                             entity_encoding_scheme=entity_encoding_scheme)
                            tmp_sub_subword_tags.append(readable_label)
                            tmp_sub_subword_tag_ids.append(label_id)
                    assert len(tmp_sub_subword_tags) == len(tmp_sub_subword_tag_ids), "IOB tag misaligned!"
                    if len(tmp_sub_subword_tags) == 0: # if there were no labels on this token, we assign "O" / 0
                        tmp_sub_subword_tags.append("O")
                        tmp_sub_subword_tag_ids.append(0)
                    subword_multientities.append(tmp_sub_subword_tags) # append ["B-person_nam-1", "I-city_nam-1"]
                    subword_multientity_ids.append(tmp_sub_subword_tag_ids)
            elif entity_encoding_scheme is None:
                all_tags = list(tags_which_begin_here.union(tags_which_continue_here))
                all_tag_ids = [get_kpwr_entity_id_from_indexed_label(readable_label, \
                                                                     entity_labels_map, \
                                                                     entity_encoding_scheme=entity_encoding_scheme) \
                               for readable_label in all_tags]
                assert len(all_tags) == len(all_tag_ids)
                if len(all_tags) == 0:
                    all_tags = ["O"]
                    all_tag_ids = [0]
                subword_multientities.extend([all_tags]*len(subwords))
                subword_multientity_ids.extend([all_tag_ids]*len(subword_ids))
                # print(subword_multientities) # naprawić tutej
            else:
                raise ValueError(f"Unknown entity encoding scheme {entity_encoding_scheme}")
            assert len(subword_multientities) == len(subword_multientity_ids) == len(subwords) == len(subword_ids), \
                   f"subwords: {len(subwords)}, ids: {len(subword_ids)}, multients: {len(subword_multientities)}, multient_ids: {len(subword_multientity_ids)}"
            curr_tokens.extend(subwords)
            curr_token_ids.extend(subword_ids)
            curr_multientities.extend(subword_multientities)
            curr_multientity_ids.extend(subword_multientity_ids)
            running_entities = heretag_names
        # ;STĄÐ - poniżej wpiąć "wypłaszczanie" multiencji i generacje zdań z relacjami
        if raw_relations is not None: # relation extraction on entity pairs
            try:
                pairwise_tokens, pairwise_token_ids, pairwise_entities, pairwise_entity_ids, pairwise_relations = \
                    kpwr_multientities_to_relations(sentence_id=sent.attrs['id'],
                                                    rels_xml=mrels,
                                                    tokens=curr_tokens,
                                                    token_ids=curr_token_ids,
                                                    multientities=curr_multientities,
                                                    multientity_ids=curr_multientity_ids,
                                                    retain_natural_no_rels=retain_natural_no_rels,
                                                    add_no_relations=add_no_relations,
                                                    entity_encoding_scheme=entity_encoding_scheme,
                                                    entity_labels_map=entity_labels_map,
                                                    relations_map=relations_map,
                                                    special_token_ids=tokenizer_obj.special_token_ids,
                                                    positional_tokens=positional_tokens)
                tokens.extend(pairwise_tokens)
                token_ids.extend(pairwise_token_ids)
                multientities.extend(pairwise_entities)
                multientity_ids.extend(pairwise_entity_ids)
                annotated_relations.extend(pairwise_relations)
            except NoRelevantRelationsException as e:
                print(f"No relevant relations found in sentence {sent.attrs['id']} and directed to skip")
                continue
        else: # only named entity recognition (with multientities)
            tokens.append(curr_tokens)
            token_ids.append(curr_token_ids)
            multientities.append(curr_multientities)
            multientity_ids.append(curr_multientity_ids)
            annotated_relations.append({})
    return tokens, token_ids, multientities, multientity_ids, annotated_relations


def tokenize_from_kbp37(*, doc_id, doc_text, tokenizer_obj, lang='english',
                           entity_encoding_scheme=None, entity_labels_map, relations_map, sentence_tokenize=True,
                           raw_relations=None, positional_tokens=None,
                           add_no_relations=False, retain_natural_no_rels=True):
 

    return tokens, token_ids, multientities, multientity_ids, annotated_relations

def flatten(l):
    flattened = [val for sublist in l for val in sublist]
    return flattened

def shuf_and_split_list(*, trainset_list, valid_frac: float, test_frac: float):
    copy_list = copy.deepcopy(trainset_list)
    train_list = valid_list = test_list = None
    last_valid_idx = 0
    random.shuffle(copy_list)
    if valid_frac is not None:
        last_valid_idx = int(valid_frac * len(trainset_list))
        valid_list = copy_list[0:last_valid_idx]
        copy_list = copy_list[last_valid_idx:]
    if test_frac is not None:
        last_valid_idx = int(test_frac * len(trainset_list))
        test_list = copy_list[0:last_valid_idx]
        copy_list = copy_list[last_valid_idx:]
    train_list = copy_list
    return copy_list, valid_list, test_list
        



    
