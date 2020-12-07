from abc import ABC, abstractmethod
from readers import WrappedTokenizer, tokenize_encoded_xml_v2, tokenize_from_kpwr, tokenize_from_kbp37, shuf_and_split_list
from readers_kpwr import mk_kpwr_labels, restore_kpwr_labels
from bs4 import BeautifulSoup
import pickle
import re
import os, random, pathlib
import xml.etree.ElementTree as ET
from sundry_exceptions import *

class DataProviderInterface(ABC):
    def __init__(self, *, config):
        self.config = config
        self.trainset = None
        self.validset = None
        self.testset = None
        self.valid_split = None
        if self.config.get('train_params') is not None:
            self.valid_split = self.config['train_params'].get('valid_split')
        self.tokenizer = None

    @abstractmethod
    def slurp(self):
        pass

    def serialize(self):
        pickle.dump(self.trainset, open("dump_trainset.p", "wb"))
        pickle.dump(self.validset, open("dump_validset.p", "wb"))
        pickle.dump(self.testset, open("dump_testset.p", "wb"))

    def deserialize(self):
        print(">>>>>>>>>>>>>>>> DESERIALIZING DATA FROM PICKLES <<<<<<<<<<<<<<<<<")
        self.trainset = pickle.load(open("dump_trainset.p", "rb"))
        self.validset = pickle.load(open("dump_validset.p", "rb"))
        self.testset = pickle.load(open("dump_testset.p", "rb"))

    def get_dataset(self, which):
        if which in ['train', 'trainset']: return self.trainset
        elif which in ['valid', 'eval', 'validset', 'evalset']: return self.validset
        elif which in ['test', 'testset']: return self.testset
        else:
            raise ValueError(f"Unknown dataset {which}")

    @abstractmethod
    def get_entity_labels(self):
        """ Return a dictionary of {int: label} """
        pass

    @abstractmethod
    def get_relations_labels(self):
        """ Return a dictionary of {int: relation} """
        pass

class SemEval2018Task7Provider(DataProviderInterface):

    def __init__(self, *, config):
        super().__init__(config=config)
        self.tokenizer = WrappedTokenizer(tokenizer_config=config['tokenizer'])
        self.append_titles = self.maxlen = None
        if self.config.get('task_specific') is not None:
            self.append_titles = self.config['task_specific'].get('append_title')
            self.maxlen = self.config['task_specific'].get('maxlen')
        print(f"appender {self.append_titles}")
        self._clazzez = None

    def _read_relations_v2(self, fname_rels, ignore_directionality=True):
        """ Read relations per document, not per class
            ret = {'doc1': {('ent_id1', 'ent_id2'): 'PART_WHOLE'}
        """
        all_clazzez = set()
        all_rels = {}
        with open(fname_rels, 'r', encoding='utf-8') as f:
            for line in f:
                rt = re.match(r"^([A-Z_-]+)\((.*?),(.*?)[,\)](REVERSE)?", line).groups()
                clazz = rt[0]; first_entity = rt[1]; second_entity = rt[2]; is_reversed = True if rt[3] is not None else False
                doc_first = re.match("(.*?)\.", first_entity).groups()[0]
                doc_second = re.match("(.*?)\.", second_entity).groups()[0]
                #if ignore_directionality is False and is_reversed is True:
                #    all_rels[(first_entity, second_entity)] = "REV_" + clazz
                #else:
                #    all_rels[(first_entity, second_entity)] = clazz
                if all_rels.get(doc_first) is None: all_rels[doc_first] = {} 
                if ignore_directionality is False and is_reversed is True:
                    all_rels[doc_first][(first_entity, second_entity)] = "REV_" + clazz
                    all_clazzez.add(f"REV_" + clazz)
                else:
                    all_rels[doc_first][(first_entity, second_entity)] = clazz
                    all_clazzez.add(clazz)
        # todo: add NONE relation
        self._clazzez = all_clazzez
        return all_rels

    def _read_corpus(self, fname_doc, fname_rels, ignore_directionality=True):
        """ Read 1.1.text.xml and return a dictionary with keys=document ids,
            values=documents with annotated entities.
        """
        documents = {}
        relations = {}
        ######### Step 1 - get relations ##############
        if fname_rels is not None:
            relations = self._read_relations_v2(fname_rels, ignore_directionality)
        ######### Step 2 - get documents ##############
        tree = ET.parse(fname_doc)
        root = tree.getroot()
        all_docs = root.findall("text")
        for doc in all_docs:
            text_id = doc.attrib['id']
            if self.append_titles is True:
                title = ET.tostring(doc.find('title'))
                title = "" if title is None else title.decode('utf-8')
                title = re.sub("</?title>", "", title).strip()
                title += ". "
            else:
                title = ""
            abstract = ET.tostring(doc.find('abstract')).decode('utf-8')
            abstract = re.sub("</?abstract>", "", abstract).strip()
            abstract = title + abstract
            documents[text_id] = abstract
        return documents, relations

    def slurp(self):
        tokenizer_obj = WrappedTokenizer(tokenizer_config=self.config['tokenizer'])
        entity_labels = self.get_entity_labels()
        ############### Trainset #############
        trainset_path = os.path.join(self.config['input_data']['source_files'], '1.1.text.xml')
        rels_path = None
        if self.config['task_specific'].get('with_relations') is True:
            rels_path = os.path.join(self.config['input_data']['source_files'], '1.1.relations.txt')
        raw_data, raw_rels = self._read_corpus(trainset_path, rels_path,
                                               ignore_directionality=self.config['task_specific'].get('ignore_directionality'))
        rels_cnt = 0
        for raw_rel in raw_rels.values(): rels_cnt += len(raw_rel)
        print(f"There are {rels_cnt} relations defined in {len(raw_rels)} documents of the training file")
        self.trainset = {}
        for doc_id, doc_txt, in raw_data.items():
            all_sent_tokens, all_sent_token_ids, all_sent_entities, all_entity_ids, all_relations_info = \
                tokenize_encoded_xml_v2(doc_id=doc_id, doc_text=doc_txt, tokenizer_obj=tokenizer_obj, sentence_tokenize=True, \
                                     entity_encoding_scheme=self.config['tokenizer'].get('entity_encoding'), \
                                     entity_labels_map=entity_labels, \
                                     raw_relations=raw_rels, positional_tokens=self.config['tokenizer'].get('add_positional_tokens'),
                                     add_no_relations=self.config['input_data'].get('add_no_relations_clazz'),
                                     retain_natural_no_rels=self.config['input_data'].get('retain_natural_no_rels')
                                     )
            for sent_num, tokens in enumerate(all_sent_tokens):
                self.trainset[f"{doc_id}_sent{sent_num}"] = {}
                self.trainset[f"{doc_id}_sent{sent_num}"]['tokens'] = tokens
                self.trainset[f"{doc_id}_sent{sent_num}"]['token_ids'] = all_sent_token_ids[sent_num]
                self.trainset[f"{doc_id}_sent{sent_num}"]['entities'] = all_sent_entities[sent_num]
                self.trainset[f"{doc_id}_sent{sent_num}"]['entity_ids'] = all_entity_ids[sent_num]
                self.trainset[f"{doc_id}_sent{sent_num}"]['relations'] = all_relations_info[sent_num]
        ############## Testset ################
        testset_path = os.path.join(self.config['input_data']['source_files'], '1.1.test.text.xml')
        rels_path = None
        if self.config['task_specific'].get('with_relations') is True:
            rels_path = os.path.join(self.config['input_data']['source_files'], 'keys.test.1.1.txt')
        raw_data, raw_rels = self._read_corpus(testset_path, rels_path,
                                               ignore_directionality=self.config['task_specific'].get('ignore_directionality'))
        self.testset = {}
        for doc_id, doc_txt, in raw_data.items():
            all_sent_tokens, all_sent_token_ids, all_sent_entities, all_entity_ids, all_relations_info = \
                    tokenize_encoded_xml_v2(doc_id=doc_id, doc_text=doc_txt, tokenizer_obj=tokenizer_obj, sentence_tokenize=True, \
                    entity_encoding_scheme=self.config['tokenizer'].get('entity_encoding'), \
                    entity_labels_map=entity_labels, \
                    raw_relations=raw_rels, positional_tokens=self.config['tokenizer'].get('add_positional_tokens'),
                    add_no_relations=self.config['input_data'].get('add_no_relations_clazz'),
                    retain_natural_no_rels=self.config['input_data'].get('retain_natural_no_rels'))
            for sent_num, tokens in enumerate(all_sent_tokens):
                self.testset[f"{doc_id}_sent{sent_num}"] = {}
                self.testset[f"{doc_id}_sent{sent_num}"]['tokens'] = tokens
                self.testset[f"{doc_id}_sent{sent_num}"]['token_ids'] = all_sent_token_ids[sent_num]
                self.testset[f"{doc_id}_sent{sent_num}"]['entities'] = all_sent_entities[sent_num]
                self.testset[f"{doc_id}_sent{sent_num}"]['entity_ids'] = all_entity_ids[sent_num]
                self.testset[f"{doc_id}_sent{sent_num}"]['relations'] = all_relations_info[sent_num]
        ############## Validset ############### 
        if self.valid_split is not None:
            rnd_doc_ids = random.sample(list(self.trainset), int(self.valid_split*len(self.trainset)))
            print(f"Moving {len(rnd_doc_ids)} documents to the validset")
            self.validset = {}
            for doc_id in rnd_doc_ids:
                self.validset[doc_id] = self.trainset[doc_id]
                del(self.trainset[doc_id])
            print(f"Validset contains {len(self.validset)} sentences")
        print(f"Trainset contains {len(self.trainset)} sentences")
        print(f"Testset contains {len(self.testset)} sentences")

    def get_entity_labels(self):
        print(">>>>>> WARNING: For SemEval2018-Task7 reader we have *hardcoded* entity labels <<<<<<<<")
        if self.config['tokenizer'].get('entity_encoding') is None:
            return {'O': 0, 'ENT': 1}
        elif self.config['tokenizer'].get('entity_encoding') == 'iob':
            return {'O': 0, 'B-ENT': 1, 'I-ENT': 2}
        else:
            raise ValueError(f"Unknown entity encoding scheme {entity_encoding}")

        # uniq_labels = set()
        # for dataset in [self.trainset, self.validset, self.testset]:
        #     if dataset is None: continue
        #     for ent_series in dataset.values():
        #         uniq_labels.update(set(ent_series['entities']))
        # uniq_labels_dict = {}
        # idx = 0 
        # if None in uniq_labels:
        #     uniq_labels_dict[None] = 0
        #     uniq_labels.remove(None)
        #     idx = 1
        # for label in sorted(uniq_labels):
        #     uniq_labels_dict[label] = idx
        #     idx += 1
        # return uniq_labels_dict

    def get_relations_labels(self):
        """ NOTE: a special class "0" = "NO_RELATION" is added if the config says so """
        uniq_clazzez = set()
        for dataset in [self.trainset, self.validset, self.testset]:
            if dataset is None: continue
            for entry in dataset.values():
                clazz = entry['relations'].get('relations_class')
                if clazz is not None: uniq_clazzez.add(clazz)
        has_no_rels_clazz = self.config['input_data'].get('add_no_relations_clazz') or \
                            self.config['input_data'].get('retain_natural_no_rels')
        offset = 1 if has_no_rels_clazz else 0
        clazzez_dict = {clazz:i+offset for i, clazz in enumerate(sorted(list(self._clazzez)))}
        if has_no_rels_clazz:
            clazzez_dict['NO_RELATION'] = 0
        return clazzez_dict

class KPWrProvider(DataProviderInterface):
    def __init__(self, *, config):
        super().__init__(config=config)
        self.tokenizer = WrappedTokenizer(tokenizer_config=config['tokenizer'])
        self.append_titles = self.maxlen = None
        if self.config.get('task_specific') is not None:
            self.maxlen = self.config['task_specific'].get('maxlen')
        print(f"appender {self.append_titles}")

    def get_entity_labels(self, ret='entities'):
        if self.config['input_data'].get('precomputed_labels_path') is None:
            labels_map, rev_labels_map, rels_map, rev_rels_map, _ = \
                mk_kpwr_labels(corpus_path=self.config['input_data']['source_files'],
                               entity_encoding_scheme=self.config['tokenizer'].get('entity_encoding'),
                               add_no_rels=self.config['input_data'].get('add_no_relations_clazz'))
            return rels_map if ret=='relations' else labels_map
        else:
            labels_map, rev_labels_map, rels_map, rev_rels_map, _ = \
                restore_kpwr_labels(path=self.config['input_data']['precomputed_labels_path'], entity_encoding_scheme=self.config['tokenizer'].get('entity_encoding'))
            return rels_map if ret=='relations' else labels_map

    def get_relations_labels(self):
        return self.get_entity_labels(ret='relations')

    def _get_relevant_paths(self):
        """ March through the KPWr corpus and select only those documents where at least one semantic relation is defined """
        relevant_paths = []; rejected_paths = []
        for root, dirs, files in os.walk(self.config['input_data']['source_files']):
            candidates = [f for f in files if ".rel." in f]
            for candidate in candidates:
                candidate_path = os.path.join(root, candidate)
                smoke_test = BeautifulSoup(open(candidate_path, "r"), "lxml")
                if smoke_test.find_all("rel", {'set': 'Semantic relations'}) == []:
                    print (f"Rejecting {candidate_path} outright - no relations defined")
                    rejected_paths.append((candidate_path.replace(".rel.xml", ".xml"), candidate_path))
                    continue
                relevant_paths.append((candidate_path.replace(".rel.xml", ".xml"), candidate_path))
        return relevant_paths, rejected_paths

    def slurp(self):
        relevant_paths, _ = self._get_relevant_paths() # STEP 0: Get only docs with rels, skip the rest
        tokenizer_obj = WrappedTokenizer(tokenizer_config=self.config['tokenizer'])

        # STEP 1: Generate labels from corpus or restore them from a pickle
        if self.config['input_data'].get('precomputed_labels_path') is None:
            labels_map, rev_labels_map, rels_map, rev_rels_map, _ = \
                mk_kpwr_labels(corpus_path=self.config['input_data']['source_files'],
                               entity_encoding_scheme=self.config['tokenizer'].get('entity_encoding'),
                               add_no_rels=self.config['input_data'].get('add_no_relations_clazz'))
        else:
            labels_map, rev_labels_map, rels_map, rev_rels_map, _ = \
                restore_kpwr_labels(path=self.config['input_data']['precomputed_labels_path'], entity_encoding_scheme=self.config['tokenizer'].get('entity_encoding'))

        self.trainset = {}; self.validset = {}; self.testset = {}
        train_paths, valid_paths, test_paths = shuf_and_split_list(trainset_list=relevant_paths,
                                                                   valid_frac=self.config['train_params'].get('valid_split'),
                                                                   test_frac=self.config['train_params'].get('test_split'))
        relevant_paths = zip([train_paths, valid_paths, test_paths], [self.trainset, self.validset, self.testset])
        # return relevant_paths
        for data_bundle, dest_dataset in relevant_paths:
            for rpt in data_bundle:
                print(rpt)
                doc_path = pathlib.Path(rpt[0]); rels_path = rpt[1]
                print(f"Processing file {doc_path} ...")
                corpus_category = doc_path.parent.parts[-1] # /wymiana/kpwr-1.1/stenogramy/00101581.xml => stenogramy
                basename_noextension = re.sub(".xml$", "", doc_path.name)
                doc_id = f"{corpus_category}_{basename_noextension}"
                with open(doc_path, 'r', encoding="utf-8") as f:
                    doc_text = f.read()
                raw_relations = None
                if self.config['task_specific'].get('with_relations') is True:
                    with open(rels_path, 'r', encoding='utf-8') as f:
                        raw_relations = f.read()
                all_sent_tokens, all_sent_token_ids, all_sent_multientities, all_sent_multientity_ids, all_sent_relations = \
                    tokenize_from_kpwr(doc_id=doc_id, doc_text=doc_text, tokenizer_obj=tokenizer_obj,
                                       entity_encoding_scheme=self.config['tokenizer'].get('entity_encoding'),
                                       entity_labels_map=labels_map,
                                       relations_map=rels_map,
                                       raw_relations=raw_relations,
                                       positional_tokens=self.config['tokenizer'].get('add_positional_tokens'),
                                       add_no_relations=self.config['input_data'].get('add_no_relations_clazz'),
                                       retain_natural_no_rels=self.config['input_data'].get('retain_natural_no_rels'))
                assert len(all_sent_tokens) == len(all_sent_token_ids) == \
                       len(all_sent_multientities) == len(all_sent_multientity_ids) == \
                       len(all_sent_relations), "Oops, sth went wrong when processing multientities"
                for fake_sent_num, tokens in enumerate(all_sent_tokens):
                    dest_dataset[f"{doc_id}_fsent{fake_sent_num}"] = {}
                    dest_dataset[f"{doc_id}_fsent{fake_sent_num}"]['tokens'] = tokens
                    dest_dataset[f"{doc_id}_fsent{fake_sent_num}"]['token_ids'] = all_sent_token_ids[fake_sent_num]
                    dest_dataset[f"{doc_id}_fsent{fake_sent_num}"]['entities'] = all_sent_multientities[fake_sent_num]
                    dest_dataset[f"{doc_id}_fsent{fake_sent_num}"]['entity_ids'] = all_sent_multientity_ids[fake_sent_num]
                    dest_dataset[f"{doc_id}_fsent{fake_sent_num}"]['relations'] = all_sent_relations[fake_sent_num]

class KBP37DataProvider(DataProviderInterface):
    def __init__(self, *, config):
        super().__init__(config=config)
        self.tokenizer = WrappedTokenizer(tokenizer_config=config['tokenizer'])

    def get_entity_labels(self):
        """ Entity types cannot be assigned reliably, so skipping them """
        if self.config['tokenizer'].get('entity_encoding') == 'iob':
            ent_labels = {'O': 0, 'B-ENT': 1, 'I-ENT': 2}
        elif self.config['tokenizer'].get('entity_encoding') is None:
            ent_labels = {'O': 0, 'ENT': 1}
        else:
            raise ValueError(f"Unknown entity encoding scheme {self.config['tokenizer'].get('entity_encoding')}")
        return ent_labels

    def get_relations_labels(self):
        """ This is fixed, so no need to scan the corpus """
        rel_list = ['NO_RELATION', 'per:alternate_names', 'per:origin', 'per:spouse', 'per:title', 'per:employee_of', \
                    'per:countries_of_residence', 'per:stateorprovinces_of_residence', 'per:cities_of_residence',
                    'per:country_of_birth', \
                    'org:alternate_names', 'org:subsidiaries', 'org:top_members/employees', 'org:founded', \
                    'org:founded_by', 'org:country_of_headquarters', 'org:stateorprovince_of_headquarters', \
                    'org:city_of_headquarters', 'org:members']
        rel_map = {}
        cnt = 0
        if self.config['input_data'].get('retain_natural_no_rels') is True:
            rel_map['NO_RELATION'] = 0
            cnt += 1
            rel_list = rel_list[1:]
        else:
            rel_list = rel_list[1:]
        for rel in rel_list:
            rel_map[rel] = cnt
            if self.config['input_data'].get('ignore_directionality') is True:
                cnt += 1
            else:
                rel_map[rel + "_rev"] = cnt + 1
                cnt += 2
        return rel_map

    def slurp(self):
        self.trainset = {}; self.validset = {}; self.testset = {}
        for dataset_obj, fname in zip([self.trainset, self.validset, self.testset], ['train.txt', 'dev.txt', 'test.txt']):
            with open(os.path.join(self.config['input_data']['source_files'], fname), 'r', encoding='utf-8') as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    doc_line = re.search("^([0-9]+)\t(.*)", line.strip())
                    if doc_line is not None:
                        doc_id = doc_line.groups()[0]
                        doc_text = doc_line.groups()[1]
                        rel_line = f.readline().strip()
                        try: 
                            tokens, token_ids, entities, entity_ids, relations = \
                                tokenize_from_kbp37(doc_id=doc_id, doc_text=doc_text, tokenizer_obj=self.tokenizer, \
                                                    entity_encoding_scheme=self.config['tokenizer'].get('entity_encoding'), \
                                                    raw_relations=rel_line, relations_map=self.get_relations_labels(), \
                                                    positional_tokens=self.config['tokenizer'].get('add_positional_tokens'), \
                                                    retain_natural_no_rels=self.config['input_data'].get('retain_natural_no_rels'), \
                                                    ignore_directionality=self.config['input_data'].get('ignore_directionality')
                                                    )
                        except MalformedEntityException as e:
                            print(str(e))
                        dataset_obj[f"sent{doc_id}"] = {}
                        dataset_obj[f"sent{doc_id}"]['tokens'] = tokens
                        dataset_obj[f"sent{doc_id}"]['token_ids'] = token_ids
                        dataset_obj[f"sent{doc_id}"]['entities'] = entities
                        dataset_obj[f"sent{doc_id}"]['entity_ids'] = entity_ids
                        dataset_obj[f"sent{doc_id}"]['relations'] = relations

class DataProviderFactory:
    @staticmethod
    def get_instance(task, config) -> DataProviderInterface:
        providers_map = {
            'semeval2018_task7': SemEval2018Task7Provider,
            'kpwr': KPWrProvider,
            'kbp37': KBP37DataProvider
        }
        ret = providers_map.get(task)
        if ret is None:
            raise ValueError(f"Unknown task type {task}. Valid tasks are {providers_map.keys()}")
        instance = ret(config=config)
        return instance
