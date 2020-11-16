import re
import os, copy, pickle
import itertools, random
from itertools import combinations
from collections import Counter
from bs4 import BeautifulSoup

class OverlappingRelationException(Exception):
    pass

class NoRelevantRelationsException(Exception):
    pass

def mk_kpwr_labels(*, corpus_path, from_pickle=None, entity_encoding_scheme, add_no_rels=True):
    """ Reads the whole corpus and returns labels_map and rev_labels_map.
        It's rather inefficient to scan everything just for labels and then slurp the contents,
        but the corpus isn't very big and the label to label_id mapping can then be done
        at the same time.

        "O" ("no label") is always mapped to 0.
    """
    print ("Starting...")
    msg = ""
    if from_pickle is not None:
        labels_map, rev_labels_map, rels_map, rev_rels_map, label_counts = pickle.load(open(from_pickle, 'rb'))
        return labels_map, rev_labels_map, rels_map, rev_rels_map, label_counts
    label_counts = Counter()
    relations_set = set()
    for root, dirs, files in os.walk(corpus_path):
        for f in files:
           if f.endswith(".rel.xml"):
               msg2 = f"Scanning relation labels in {os.path.join(root, f)}"
               print(msg2)
               msg += msg2 + "\n"
               soup = BeautifulSoup(open(os.path.join(root, f), "r", encoding="utf-8"), "xml")
               rels = [r.attrs['name'] for r in soup.find_all('rel', {'set': 'Semantic relations'})]
               relations_set.update(set(rels))
           if f.endswith(".xml"):
               msg2 = f"Scanning labels in {os.path.join(root, f)}"
               print(msg2)
               msg += msg2 + "\n"
               soup = BeautifulSoup(open(os.path.join(root, f), "r", encoding="utf-8"), "xml")
               all_annotations = [str(e.attrs.get('chan')) for e in soup.find_all('ann')]
               all_annotations = {a for a in all_annotations if a.endswith('_nam')}
               for annotation in all_annotations:
                   num_labels = len(soup.find_all("ann", {"chan": annotation}))
                   label_counts.update({annotation: num_labels})
    all_labels = sorted(label_counts.keys())
    labels_map = {"O": 0}; rev_labels_map = {0: "O"}
    cnt = 1
    for label in all_labels:
        if entity_encoding_scheme is None:
            labels_map[label] = cnt
            rev_labels_map[cnt] = label
            cnt += 1
        elif entity_encoding_scheme == 'iob':
            labels_map[f"B-{label}"] = cnt
            rev_labels_map[cnt] = f"B-{label}"
            cnt += 1
            labels_map[f"I-{label}"] = cnt
            rev_labels_map[cnt] = f"I-{label}"
            cnt += 1
        elif entity_encoding_scheme == 'bilou':
            labels_map[f"B-{label}"] = cnt
            rev_labels_map[cnt] = f"B-{label}"
            cnt += 1
            labels_map[f"I-{label}"] = cnt
            rev_labels_map[cnt] = f"I-{label}"
            cnt += 1
            labels_map[f"L-{label}"] = cnt
            rev_labels_map[cnt] = f"L-{label}"
            cnt += 1
            labels_map[f"U-{label}"] = cnt
            rev_labels_map[cnt] = f"U-{label}"
            cnt += 1
        else:
            raise ValueError(f"Unknown entity encoding scheme {entity_encoding_scheme}")
    label_counts.update({"O": 0})
    rels_map = {}
    if add_no_rels is True:
        rels_cnt = 0
        rels_map['NO_RELATION'] = 0
        rels_cnt = 1
    else:
        rels_cnt = 0
    for rel in sorted(list(relations_set)):
        rels_map[rel] = rels_cnt
        rels_cnt += 1
    rev_rels_map = {v:k for k,v in rels_map.items()}
    fname = "kpwr_labels_noencodingscheme.p" if entity_encoding_scheme is None else f"kpwr_labels_{entity_encoding_scheme}.p"

    pickle.dump((labels_map, rev_labels_map, rels_map, rev_rels_map, label_counts), open(fname, "wb"))
    with open(fname.strip(".p") + "_log.txt", 'w') as f:
        f.write(msg)
    return labels_map, rev_labels_map, rels_map, rev_rels_map, label_counts

def restore_kpwr_labels(*, path, entity_encoding_scheme):
    if entity_encoding_scheme is None:
        labels_map, rev_labels_map, rels_map, rev_rels_map, label_counts = pickle.load(open(os.path.join(path, "kpwr_labels_noencodingscheme."), "rb"))
    elif entity_encoding_scheme == 'iob':
        labels_map, rev_labels_map, rels_map, rev_rels_map, label_counts = pickle.load(open(os.path.join(path, "kpwr_labels_iob.p"), "rb"))
    else:
        raise ValueError(f"Unknown entity encoding scheme {entity_encoding_scheme}")
    return labels_map, rev_labels_map, rels_map, rev_rels_map, label_counts

def kpwr_xml2dataprovider_entity_label(*, xml_relation_fromto_node, entity_encoding_scheme):
    """ Given an XML label node like this: <from sent="sent2" chan="district_nam">1</from>
        generate a set of possible KPWr entity labels according to the entity_encoding_scheme.

        Ex) when ees is 'iob' - return ['B-district_nam-1', 'I-district_nam-1']
    """
    ret_labels = []
    label_text = xml_relation_fromto_node.attrs['chan']
    label_seq_number = xml_relation_fromto_node.text
    if entity_encoding_scheme is None:
        ret_labels.append(f"{label_text}-{label_seq_number}")
    elif entity_encoding_scheme == 'iob':
        ret_labels.append(f"B-{label_text}-{label_seq_number}")
        ret_labels.append(f"I-{label_text}-{label_seq_number}")
    elif entity_encoding_scheme == 'bilou':
        ret_labels.append(f"B-{label_text}-{label_seq_number}")
        ret_labels.append(f"I-{label_text}-{label_seq_number}")
        ret_labels.append(f"L-{label_text}-{label_seq_number}")
        ret_labels.append(f"U-{label_text}-{label_seq_number}")
    else:
        raise ValueError(f"Unknown entity encoding scheme {entity_encoding_scheme}")
    return ret_labels

def print_kpwr_tuples(i, tokens, token_ids, multientities, multientity_ids, rels=None):
    print("{: >4} {: >15} {: >10} {: >40} {: >20}".format("idx", "subword", "subword_id", "multients", "multient_ids"))
    for j in range(len(tokens[i])): 
        print("{: >4} {: >15} {: >10} {: >40} {: >20}".format(j, tokens[i][j], token_ids[i][j], str(multientities[i][j]),
                                                       str(multientity_ids[i][j])))
    if rels is not None:
        print(f"Relations: {rels[i]}")

def generic_entity_id_from_label(raw_label, labels_map, entity_encoding_scheme=None):
    """
    "O" => 0                                              # all schemes
    "B-firstname_nam-1" => labels_map['B-firstname_nam-1']   # ees = 'iob'/'bilou'
    """
    return labels_map[raw_label]

def get_kpwr_entity_id_from_indexed_label(indexed_label, kpwr_labels_map, entity_encoding_scheme=None):
    """
    "O" => 0                                              # all schemes
    "B-firstname_nam-1" => kpwr_labels_map['B-firstname']   # ees = 'iob'/'bilou'
    "firstname_nam-1"   => kpwr_labels_map['firstname']     # ees = None
    """
    if indexed_label == 'O':
        return kpwr_labels_map['O']
    split_label = indexed_label.split("-")
    if entity_encoding_scheme == None:
        assert len(split_label) == 2, f"With entity encoding scheme set to {entity_encoding_scheme}, " \
                                      f"expected label to look like 'person_nam-1' but found {indexed_label}"
        key = split_label[0]
    elif entity_encoding_scheme == 'iob':
        assert len(split_label) == 3, f"With entity encoding scheme set to {entity_encoding_scheme}, " \
                                      f"expected label to look like 'B-person_nam-1' but found {indexed_label}"
        key = "-".join(split_label[0:2])
    else:
        raise ValueError(f"Unsupported entity encoding scheme {entity_encoding_scheme}")
    return kpwr_labels_map[key]

def kpwr_is_running_entity(tag):
    """ Return true if `tag` is <ann> and contains a running entity (to be used with BeautifulSoup's find_all """
    if tag.name != 'ann': return False
    if str(tag.attrs.get('chan')).endswith('_nam'):
        if int(tag.text) > 0:
            return True
        else:
            return False
    else:
        return False

def kpwr_multientities_to_relations(*, sentence_id, rels_xml, tokens, token_ids, multientities, multientity_ids,
                                    retain_natural_no_rels=True, add_no_relations=False,
                                    entity_encoding_scheme=None, entity_labels_map, relations_map, special_token_ids=None,
                                    positional_tokens=None):
    """ Input:  the output of tokenize_from_kpwr
        Output: [tokens], [token_ids], [entities], [entity_ids], [annotated_relations]
        where [entities] and [entity_ids] are decided as follows:

        if retain_natural_no_rels is False and len(relevant_relations) == 0:
            return empty set
        sents = []
        1. For each relation in annotated_relations_all:
           1.1. Remove entity annotations irrevelant for the relation:
              - only one pair <e1>..</e1>, <e2>..</e2> should remain
           1.2. Insert positional tokens if required
           1.3. Append the result to sents
        2. If add_no_relations is True:
           2.1. Generate all possible relation pairs
           2.2. From the set in .1 remove relations where entities overlap
           2.3. Randomly shuffle the pairs
           2.4. Choose no more than 5 or the same number or NO_RELATION pairs (whichever is higher)
                2.4.1. Insert positional tokens if required
           2.5. Append to sents
        return sents 
    """ 
    if retain_natural_no_rels is False and add_no_relations is True:
        raise ValueError("Conflicting parameters - cannot set `retain_natural_no_rels` to False " \
                         "and `add_no_relations` to True at the same time")
    
    if retain_natural_no_rels is True or add_no_relations is True:
        if relations_map.get('NO_RELATION') != 0:
            raise ValueError("The relation labels map doesn't contain a NO_RELATION key or its value isn't set to 0")
    annotated_relations = rels_xml.find_all("rel", {"set": "Semantic relations"})
    relevant_relations = [r for r in annotated_relations if r.find("from").attrs['sent'] == sentence_id \
                                                         and r.find("to").attrs['sent'] == sentence_id \
                                                         and r.find("from").attrs['chan'].endswith('_nam') \
                                                         and r.find("to").attrs['chan'].endswith('_nam')]
    all_tokens = []; all_token_ids = []; all_entities = []; all_entity_ids = []; all_rels = []
    
    # Add NO_RELATION class on up to five unrelated entities (steps 2.1 to 2.4)
    # Note: Steps 2.4.1 and 2.5 are accomplished through adding a bogus "NO_RELATION" xml node to
    # `relevant_relations`
    if add_no_relations is True:
        nonoverlapping_pairs = kpwr_find_all_non_overlapping_entity_pairs(multients=multientities, \
                entity_encoding_scheme=entity_encoding_scheme,
                pre_existing_relations_xml=relevant_relations)
        #print(f"There are {len(relevant_relations)} predefined relations in sentence {sentence_id}. " \
        #      f"Additionally found {len(nonoverlapping_pairs)} nonoverlapping entity pairs " \
        #      f"which can be used to build NO_RELATION: {nonoverlapping_pairs}")
        if len(nonoverlapping_pairs) > 0:
            rels_to_add = random.sample(nonoverlapping_pairs, min(len(nonoverlapping_pairs), len(relevant_relations), 5))
            rels_to_add_xml = ""
            for bogus_rel in rels_to_add:
                first_chan = bogus_rel[0].split("-")[0]; first_val = bogus_rel[0].split("-")[1]
                second_chan = bogus_rel[1].split("-")[0]; second_val = bogus_rel[1].split("-")[1]
                rels_to_add_xml += f'<rel name="NO_RELATION" set="Semantic relations">'
                rels_to_add_xml += f'<from sent="{sentence_id}" chan="{first_chan}">{first_val}</from>'
                rels_to_add_xml += f'<to sent="{sentence_id}" chan="{second_chan}">{second_val}</to>'
                rels_to_add_xml += f'</rel>'
            #print(rels_to_add_xml)
            rels_to_add_xml = BeautifulSoup(rels_to_add_xml, "lxml").find_all('rel')
            relevant_relations.extend(rels_to_add_xml)
            #print(f"Relevant relations now contain {len(relevant_relations)} elements, including NO_RELATION.")
        else:
            print(f"Info: In sentence {sentence_id} all entities are in some relation - cannot generate additional NO_RELS")

    if len(relevant_relations) == 0:
        if retain_natural_no_rels is True:
            all_tokens.append(tokens)
            all_token_ids.append(token_ids)
            all_entities.append([ent[0] for ent in multientities])
            all_entity_ids.append([ent[0] for ent in multientity_ids])
            all_rels.append({'e1_beg': None, 'e1_end': None, 'e2_beg': None, 'e2_end': None, 'relation_class': 'NO_RELATION', 'relation_class_id': 0})
        else:
            raise NoRelevantRelationsException(f"No relations found in sentence {sentence_id}")

    for rel in relevant_relations:
        # 0.1. Copy the arrays, since we'll be operating on their indices many times:
        tokens_ = copy.deepcopy(tokens);               token_ids_ = copy.deepcopy(token_ids)
        multientities_ = copy.deepcopy(multientities); multientity_ids_ = copy.deepcopy(multientity_ids)
        rels_ = {}
        # 1.1. Remove entity annotations irrevelant for the relation
        relevant_entity_labels = []
        relevant_entity_labels.extend(kpwr_xml2dataprovider_entity_label(xml_relation_fromto_node=rel.find('from'),
                                      entity_encoding_scheme=entity_encoding_scheme))
        relevant_entity_labels.extend(kpwr_xml2dataprovider_entity_label(xml_relation_fromto_node=rel.find('to'),
                                      entity_encoding_scheme=entity_encoding_scheme))
        relevant_entity_labels = set(relevant_entity_labels).union({'O'})
        #print(f"Relevant entity labels are {relevant_entity_labels}")
        #print(f"Multientities are {multientities_}")
        try:
            for i in range(len(tokens_)):
                multientities_[i] = [l for l in multientities_[i] if l in relevant_entity_labels]
                if len(multientities_[i]) == 0:
                    multientities_[i].append('O')
                multientity_ids_[i] = [get_kpwr_entity_id_from_indexed_label(indexed_label, \
                                                                             entity_labels_map, \
                                                                             entity_encoding_scheme=entity_encoding_scheme) \
                                       for indexed_label in multientities_[i]]
                #print(f"at {i}: multients - {multientities_[i]}, ids - {multientity_ids_[i]}")
                #print(f"at {i} lens: multients - {len(multientities_[i])}, ids - {len(multientity_ids_[i])}")
                if not len(multientities_[i]) == len(multientity_ids_[i]) == 1:
                    raise OverlappingRelationException(f"There should be exactly 1 named entity label (or 'O') per token, " \
                             f"but at index {i} found multients={multientities_[i]} and ids={multientity_ids_[i]}. " \
                             f"\nRel:\n{rel}\nRelevant_labels: {relevant_entity_labels}")
                # assert len(multientities_[i]) == len(multientity_ids_[i]) == 1, "There should be exactly 1 named entity label (or 'O') per token, " \
                #         f"but at index {i} found multients={multientities_[i]} and ids={multientity_ids_[i]}.\nRel:\n{rel}\nRelevant_labels: {relevant_entity_labels}"
                multientities_[i] = multientities_[i][0] # ['O'] => 'O' after making sure that only one entity remains
                multientity_ids_[i] = multientity_ids_[i][0]
        except OverlappingRelationException as e:
            print(">>>>>>>>>>>>>>>>>>>> OVERLAPPING EXCEPTION WARNING <<<<<<<<<<<<<<<<<<<<")
            print(str(e))
            continue
        # 1.2. Insert positional tokens if required
        if positional_tokens is not None:
            if positional_tokens == 'scheme_1':
                from_label_pure = f"{rel.find('from').attrs['chan']}-{rel.find('from').text}"
                to_label_pure = f"{rel.find('to').attrs['chan']}-{rel.find('to').text}"
                positional_token_locations_to_insert = calculate_positional_token_offsets(entity_labels=multientities_,
                                                                                          from_label_pure=from_label_pure,
                                                                                          to_label_pure=to_label_pure,
                                                                                          entity_encoding_scheme=entity_encoding_scheme,
                                                                                          dbg_rel=rel)
                tokens_, token_ids_, multientities_, multientity_ids_, positional_token_locations_inserted = \
                    insert_positional_tokens(tokens=tokens_, token_ids=token_ids_, entities=multientities_, entity_ids=multientity_ids_, \
                                             labels_map=entity_labels_map, \
                                             positions=positional_token_locations_to_insert, \
                                             entity_encoding_scheme=entity_encoding_scheme, \
                                             positional_tokens=positional_tokens, \
                                             special_token_ids=special_token_ids)
                rels_ = positional_token_locations_inserted
                rels_['relation_class'] = rel.attrs['name']
                rels_['relation_class_id'] = relations_map[rel.attrs['name']]
            else:
                raise ValueError(f"Unknown positional tokens scheme {positional_tokens}")
        else: # If ees is None, the tokens_ and other arrays already contain what is needed - no further action required
            pass
        all_tokens.append(tokens_)
        all_token_ids.append(token_ids_)
        all_entities.append(multientities_)
        all_entity_ids.append(multientity_ids_)
        all_rels.append(rels_)

    return all_tokens, all_token_ids, all_entities, all_entity_ids, all_rels

def kpwr_find_all_non_overlapping_entity_pairs(*, multients, entity_encoding_scheme, pre_existing_relations_xml=None, max_nchoosek=45):
    """ Generates pairs of entities, e.g. ('city_nam-1', 'facility_nam-1') that do not overlap
        in multients.
    """
    all_pure_entities_in_sent = set()
    overlapping_pairs = set()
    pre_existing_pairs = set()

    ########## PAIRS ALREADY USED UP BY PREDEFINED RELATIONS #########
    for rel in pre_existing_relations_xml or []:
        from_label_pure = f"{rel.find('from').attrs['chan']}-{rel.find('from').text}"
        to_label_pure = f"{rel.find('to').attrs['chan']}-{rel.find('to').text}"
        ents = tuple(sorted([from_label_pure, to_label_pure]))
        pre_existing_pairs.add(ents)
    ########### OVERLAPPING PAIRS ##########
    for ents in multients:
        combinations_on_this_token = None
        if entity_encoding_scheme is None:
            pure_ents_here = ents
        elif entity_encoding_scheme == 'iob':
            pure_ents_here = [re.sub("^[BI]-", "", ent) for ent in ents]
        else:
            raise ValueError(f"Unknown entity encoding scheme {entity_encoding_scheme}")
        if len(pure_ents_here) == 1: all_pure_entities_in_sent.add(pure_ents_here[0])
        elif len(pure_ents_here) > 1:
            for pure_ent in pure_ents_here: all_pure_entities_in_sent.add(pure_ent)
            x = list(combinations(sorted(pure_ents_here), 2))
            for local_ent_pair in x: overlapping_pairs.add(local_ent_pair)
        else:
            raise ValueError("Oops, we didn't expect a multient list to be of length zero. Something's bronek.")
    all_pure_entities_in_sent -= {"O"}
    ############ ALL PAIRS #############
    entity_pairs_tmp = combinations(sorted(all_pure_entities_in_sent), 2)
    all_pairs = set(itertools.islice(entity_pairs_tmp, max_nchoosek))
    nonoverlapping_pairs = all_pairs - overlapping_pairs - pre_existing_pairs
    return nonoverlapping_pairs

def kpwr_find_overlapping_entities(*, xml_relation_node, multients, entity_encoding_scheme):
    """ For each entity in entity_classes find entities that overlap.

        Ex) xml_relation_node:
            <rel name="location" set="Semantic relations">
                <from sent="sent2" chan="facility_nam">1</from>
                <to sent="sent2" chan="district_nam">1</to>
            </rel>
            and multients as below:

            subword subword_id                                multients         multient_ids
            ▁to        222     ['B-city_nam-1', 'B-facility_nam-1']             [21, 41]
            Returns {B-facility_nam-1}
    """
    raise NotImplementedError("Niepotrzebne?")
    encoded_entity_labels = []
    for direction in ['from', 'to']:
        partial_labels = kpwr_xml2dataprovider_entity_label(xml_relation_fromto_node=xml_relation_node.find(direction),
                                                            entity_encoding_scheme=entity_encoding_scheme)
        encoded_entity_labels.extend(partial_labels)
    encoded_entity_labels = set(encoded_entity_labels)
 
def calculate_positional_token_offsets(*, entity_labels, from_label_pure, to_label_pure, entity_encoding_scheme, dbg_rel):
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
        #print(dbg_rel)
        #print(entity_labels)
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
                             labels_map, positional_tokens, special_token_ids, corpus='kpwr'):
    if entity_encoding_scheme not in [None, "iob"]:
        raise ValueError(f"Unknown entity encoding scheme {entity_encoding_scheme}")
    if positional_tokens == 'scheme_1':
        e1_beg = "<e1>"; e1_end = "</e1>"; e2_beg = "<e2>"; e2_end = "</e2>"
    elif positional_tokens == 'scheme_2':
        e1_beg = "$"; e1_end = "$"; e2_beg = "#"; e2_end = "#"
    else:
        raise ValueError(f"Unknown positional tokens scheme {positional_tokens}")

    if corpus == 'kpwr':
        entity_id_getter_fn = get_kpwr_entity_id_from_indexed_label
    else:
        entity_id_getter_fn = generic_entity_id_from_label
    ############## </e2> ############
    tokens.insert(positions['e2_end'] + 1, e2_end)
    token_ids.insert(positions['e2_end'] + 1, special_token_ids[e2_end])
    if entity_encoding_scheme == None:
        entities.insert(positions['e2_end'] + 1, entities[positions['e2_end']])
        entity_ids.insert(positions['e2_end'] + 1, entity_ids[positions['e2_end']])
    elif entity_encoding_scheme == 'iob':
        expected_end_label = re.sub("^B-", "I-", entities[positions['e2_end']]) # B-city_nam-1 => I-city_nam-1 if singleton
        entities[positions['e2_end']] = expected_end_label
        entity_ids[positions['e2_end']] = entity_id_getter_fn(expected_end_label, labels_map, entity_encoding_scheme)
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
        entity_ids[positions['e2_beg']] = entity_id_getter_fn(i_beg_label, labels_map, entity_encoding_scheme)
        entities.insert(positions['e2_beg'], b_beg_label)
        entity_ids.insert(positions['e2_beg'], entity_id_getter_fn(b_beg_label, labels_map, entity_encoding_scheme))
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
        entity_ids[positions['e1_end']] = entity_id_getter_fn(expected_end_label, labels_map, entity_encoding_scheme)
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
        entity_ids[positions['e1_beg']] = entity_id_getter_fn(i_beg_label, labels_map, entity_encoding_scheme)
        entities.insert(positions['e1_beg'], b_beg_label)
        entity_ids.insert(positions['e1_beg'], entity_id_getter_fn(b_beg_label, labels_map, entity_encoding_scheme))
    else:
        raise ValueError(f"Unknown ees {entitity_encoding_scheme}")
    #inserted_locations = {"e1_beg": tokens.index(e1_beg), \
    #                      "e1_end": tokens.index(e1_end), \
    #                      "e2_beg": tokens.index(e2_beg), \
    #                      "e2_end": tokens.index(e2_end)}
    inserted_locations = {"e1_beg": positions['e1_beg'], \
                          "e1_end": positions['e1_end'] + 2, \
                          "e2_beg": positions['e2_beg'] + 2, \
                          "e2_end": positions['e2_end'] + 4}
    return tokens, token_ids, entities, entity_ids, inserted_locations
