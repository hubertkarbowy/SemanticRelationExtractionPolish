import numpy as np
import tensorflow as tf
import copy, pickle
import colors
import re
import os
from transformers import AutoTokenizer, TFAutoModel, AutoConfig
from tensorflow.keras.preprocessing.sequence import pad_sequences as pad_seq
from tensorflow.keras.layers import Input
from EncjoSzukaczLSTM import EncjoSzukaczLSTM
from DataProvider import DataProviderFactory
from readers import tokenize_encoded_xml_v2
from sklearn.metrics import classification_report

class RelacjoSzukaczBERT(EncjoSzukaczLSTM):
    def __init__(self, *, config):
        # super().__init__(config=config) # TODO: fix inheritance
        self.config = config
        if self.config['tokenizer']['iface'] != 'transformers':
            raise ValueError("The bert-based relations finder requires a BERT-compatible tokenizer iface")
        self.data_provider = DataProviderFactory.get_instance(config['input_data']['reader'], config)
        if config['input_data'].get('deserialize') is True:
            self.data_provider.deserialize()
        else:
            self.data_provider.slurp()
        self.model = None
        self.mangled_pickles = None

    def _print_from_dataset(self, dataset, relation_infos, i):
        pieces = self.data_provider.tokenizer.convert_ids_to_tokens(dataset['input_ids'][i])
        print(pieces)
        print("{: >4} {: >20} {: >15} {: >8} {: >8} {: >8}".format("idx", "subword", "subword_id", "attn_mask", "e1_mask", "e2_mask"))
        for j in range(len(pieces)): 
            print("{: >4} {: >20} {: >15} {: >8} {: >8} {: >8}".format(j, pieces[j], dataset['input_ids'][i][j],
                                                                dataset['attention_mask'][i][j], dataset['e1_mask'][i][j][0],
                                                                dataset['e2_mask'][i][j][0]))
        print(f"Relation infos (orig): {relation_infos[i]}")

    def _mangle_rels(self, dataset, as_generator=False):
        self.labels_map = self.data_provider.get_relations_labels()
        self.rev_labels_map = {v:k for k,v in self.labels_map.items()}
        all_rel_labels = [r['relations'].get('relation_class') for r in dataset.values()]
        if self.config['input_data'].get('add_no_relations_clazz'):
            all_rel_labels = [r or "NO_RELATION" for r in all_rel_labels]
        if not all(all_rel_labels):
            raise ValueError("There are some sentences without relations in your input sets. " \
                             "Please fix it in one of these ways: 1) Annotate the sentences explicitly as NO_RELATION " \
                             "2) Remove the sentences from the input data " \
                             "3) Set the 'add_no_relations_clazz' flag to `True` in config['input_data']")
        all_rel_labels = [self.labels_map[l] for l in all_rel_labels]
        labels_categorical = tf.keras.utils.to_categorical(all_rel_labels, num_classes=len(self.labels_map))
        return labels_categorical

    def _get_entity_mask(self, record, which_entity, override_max_seq_len=None):
        seqlen = override_max_seq_len or self.config['max_seq_len']
        ret = [0]*seqlen
        # first, second = (0, 1) if which_entity == 'e1' else (2, 3) if which_entity == 'e2' else (None,None)
        first = record[f"{which_entity}_beg"]; second = record[f"{which_entity}_end"]
        if second in [0, None]:
            return ret
        for i in range(first, second+1):
            ret[i] = 1
        return ret

    def _mangle_inputs(self, which, as_generator=False):
        # if self.mangled_pickles is not None:
        #     p = pickle.load(open(os.path.join(self.mangled_pickles, f"{which}_RBert.p"), 'rb'))
        #     return p
        print("Copying dataset...")
        dataset = copy.deepcopy(self.data_provider.get_dataset(which))
        too_long_sents = set()
        print("Appending special tokens...")
        for sent_key in dataset.keys():
            dataset[sent_key]['tokens'].append('[SEP]')
            dataset[sent_key]['token_ids'].append(self.data_provider.tokenizer.special_token_ids['[SEP]'])
            dataset[sent_key]['entities'].append(None)
            dataset[sent_key]['entity_ids'].append(None)

            dataset[sent_key]['tokens'].insert(0, '[CLS]')
            dataset[sent_key]['token_ids'].insert(0, self.data_provider.tokenizer.special_token_ids['[CLS]'])
            dataset[sent_key]['entities'].insert(0, None)
            dataset[sent_key]['entity_ids'].insert(0, None)

            if dataset[sent_key]['relations'].get('e1_beg') is not None: # shift by 1 because CLS
                dataset[sent_key]['relations']['e1_beg'] += 1
                dataset[sent_key]['relations']['e1_end'] += 1
                dataset[sent_key]['relations']['e2_beg'] += 1
                dataset[sent_key]['relations']['e2_end'] += 1
                if any([marker_pos > self.config['max_seq_len'] - 1 for marker_pos in [dataset[sent_key]['relations']['e1_beg'],
                                                                                       dataset[sent_key]['relations']['e1_end'],
                                                                                       dataset[sent_key]['relations']['e2_beg'],
                                                                                       dataset[sent_key]['relations']['e2_end']]]):
                    print(f"Warning! Sentence {sent_key} contains entities that fall outside the padding boundary and will be removed.")
                    too_long_sents.add(sent_key)
            if dataset[sent_key]['token_ids'][-1] != 0: # SEP falls outside the max_len boundary, so it has to be reinstated on the final position
                dataset[sent_key]['token_ids'][-1] = self.data_provider.tokenizer.special_token_ids['[SEP]']
        print("Removing longish sents...")
        for sent_key in too_long_sents: del(dataset[sent_key])
        print("Extracting relation infos...")
        relation_infos = [dataset[sent_key]['relations'] for sent_key in dataset.keys()]
        print("Padding...")
        padded_tokens = self._mangle_token_ids(dataset)
        token_types = tf.zeros(padded_tokens.shape, dtype=tf.int32)
        print("Attention masks...")
        #attn_mask = tf.map_fn(fn=lambda row: tf.map_fn(fn=lambda tok_id: 0 if tok_id == 0 else 1, elems=row), elems=padded_tokens, dtype=tf.int32) # FIXME: extremely slow
        attn_mask = tf.cast(tf.cast(padded_tokens, dtype=tf.bool), dtype=tf.int32)
        print("One-hot encoding relation classes...")
        relation_classes_tensor = self._mangle_rels(dataset)
        print("Masks for <e1>...")
        e1_mask_tensor = tf.constant([self._get_entity_mask(sg_rel_info, 'e1') for sg_rel_info in relation_infos], dtype=tf.float32)
        e1_mask_tensor = tf.expand_dims(e1_mask_tensor, axis=2) # to enable broadcasting, put the mask in the last axis so that it can propagate into the vector of 768 outputs on each timestep
        print("Masks for <e2>...")
        e2_mask_tensor = tf.constant([self._get_entity_mask(sg_rel_info, 'e2') for sg_rel_info in relation_infos], dtype=tf.float32)
        e2_mask_tensor = tf.expand_dims(e2_mask_tensor, axis=2)
        return {
                'input_ids': padded_tokens,
                'token_type_ids': token_types,
                'attention_mask': attn_mask,
                'e1_mask': e1_mask_tensor,
                'e2_mask': e2_mask_tensor,
                }, relation_classes_tensor, relation_infos

    def _build_model(self):
        # TODO: 1) [?potrzebne?] Wylaczyc gradient przy zmiennych ng_
        # 2) [OK] Pozbyc sie nanow przy dzieleniu przez zero -> ;TUTEJ
        # 3) [OK] Detalicznie przesledzic czy tensory sa dobrze obliczane przy kazdym z krokow
        # 4) [OK] Przerobic trening
        # 5) Od-hardkodowac 768 poza tą funkcją
        input_ids_layer = Input(shape=(self.config['max_seq_len'],), name='input_ids', dtype=tf.int32)
        input_toktypeids_layer = Input(shape=(self.config['max_seq_len'],), name='token_type_ids', dtype=tf.int32)
        input_attnmask_layer = Input(shape=(self.config['max_seq_len'],), name='attention_mask', dtype=tf.int32)
        input_e1_mask_layer = Input(shape=(self.config['max_seq_len'], 1, ), name='e1_mask', dtype=tf.float32)
        input_e2_mask_layer = Input(shape=(self.config['max_seq_len'], 1, ), name='e2_mask', dtype=tf.float32)
        bert_config = AutoConfig.from_pretrained(self.config['tokenizer']['pretrained'])
        hidden_size = bert_config.hidden_size # 768 by default

        bert_layer = TFAutoModel.from_pretrained(self.config['tokenizer']['pretrained'])

        bert_layer.trainable = self.config['task_specific'].get('finetune_pretrained')
        bert_out_hidden = bert_layer({'input_ids': input_ids_layer, 'token_type_ids': input_toktypeids_layer,
                                      'attention_mask': input_attnmask_layer})[0] # only the hidden states
        
        ng_pierwsza = tf.math.multiply(bert_out_hidden, input_e1_mask_layer, name="e1_mask")
        ng_druga    = tf.math.multiply(bert_out_hidden, input_e2_mask_layer, name="e2_mask")
        ng_len_pierwsza = tf.reduce_sum(input_e1_mask_layer, axis=1, name="len_of_e1")
        ng_len_druga    = tf.reduce_sum(input_e2_mask_layer, axis=1, name="len_of_e2")
        
        H_ij_avg = tf.math.divide_no_nan(tf.reduce_sum(ng_pierwsza, axis=1), ng_len_pierwsza, name="e1_avg") # who would have thought TF has a special op for dealing with 0 / 0 nans...
        H_km_avg = tf.math.divide_no_nan(tf.reduce_sum(ng_druga, axis=1), ng_len_druga, name="e2_avg")

        cls = bert_out_hidden[:, 0, :]
        dense_cls = tf.keras.layers.Dense(hidden_size, activation='tanh', name="dense_cls")(cls)
        dense_cls_dropout = tf.keras.layers.Dropout(0.1)(dense_cls)

        dense_shared = tf.keras.layers.Dense(hidden_size, activation='tanh', name="dense_e1_e2")
        H_ij_dense = dense_shared(H_ij_avg)
        H_ij_dense_dropout = tf.keras.layers.Dropout(0.1)(H_ij_dense)
        H_km_dense = dense_shared(H_km_avg)
        H_km_dense_dropout = tf.keras.layers.Dropout(0.1)(H_km_dense)

        polaczone = tf.concat([dense_cls_dropout, H_ij_dense_dropout, H_km_dense_dropout], axis=1, name="concatenated")
        # przedostatnia = tf.keras.layers.Dense(3*len(self.data_provider.get_relations_labels()), activation='relu')(polaczone)
        # ostatnia = tf.keras.layers.Dense(len(self.data_provider.get_relations_labels()), activation='softmax')(przedostatnia)
        ostatnia = tf.keras.layers.Dense(len(self.data_provider.get_relations_labels()), activation='softmax', name='final_classif')(polaczone)
        
        model = tf.keras.Model(inputs=[input_ids_layer, input_toktypeids_layer, input_attnmask_layer, input_e1_mask_layer, input_e2_mask_layer],
                               outputs=[ostatnia])
        model.summary()
        return model

    def train(self):
        x_train, y_train, _ = self._mangle_inputs('trainset')
        x_valid, y_valid, _ = self._mangle_inputs('validset')
        x_train_bundle = [x_train['input_ids'], x_train['token_type_ids'],
                          x_train['attention_mask'], x_train['e1_mask'],
                          x_train['e2_mask']]
        x_valid_bundle = [x_valid['input_ids'], x_valid['token_type_ids'],
                          x_valid['attention_mask'], x_valid['e1_mask'],
                          x_valid['e2_mask']]
        self.model = self.model or self._build_model()
        opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(x_train_bundle, y_train, epochs=self.config['train_params'].get('num_epochs') or 1,
                  batch_size=self.config['train_params'].get('batch_size'),
                  validation_data=(x_valid_bundle, y_valid))

    def prepare_pred_input(self, mangled_dataset, from_idx, to_idx):
        x_pred_bundle = [mangled_dataset['input_ids'][from_idx:to_idx], mangled_dataset['token_type_ids'][from_idx:to_idx],
                         mangled_dataset['attention_mask'][from_idx:to_idx], mangled_dataset['e1_mask'][from_idx:to_idx],
                         mangled_dataset['e2_mask'][from_idx:to_idx]]
        ret = self.model.predict(x_pred_bundle)
        return ret


    def evaluate(self, which=None):
        which=which or 'validset'
        x_test, y_test = self._mangle_inputs(which)
        res = self.model.predict(x_test)
        y_preds = np.argmax(res, axis=1)
        y_golds = np.argmax(y_test, axis=1)
        readable_labels = []
        for label_index in sorted(self.rev_labels_map): readable_labels.append(self.rev_labels_map[label_index])
        print(classification_report(y_golds, y_preds, labels=sorted(self.rev_labels_map.keys()), target_names=readable_labels)) 

    def predict(self):
        raise NotImplementedError

    def _predict_from_token_ids(self, token_ids, golds=None):
        #padded_token_ids = pad_seq([token_ids], maxlen=self.config['max_seq_len'],
        #                           padding='pre', truncating='post', value=0)[0].tolist()
        y_preds = self.model.predict(token_ids)
        y_preds = [self.rev_labels_map[l] for l in np.argmax(y_preds, axis=1).tolist()]
        recovered_sequences = []
        for sequence in token_ids:
            sequence = [t for t in sequence if t != 0]
            toks = self.data_provider.tokenizer.convert_ids_to_tokens(sequence)
            recovered_sequences.append(self.data_provider.tokenizer.detokenize(toks))
        # recovered_tokens = self.data_provider.tokenizer.convert_ids_to_tokens(padded_token_ids)
        if golds is not None:
            gold_labels = [self.rev_labels_map[l] for l in np.argmax(golds, axis=1).tolist()]
        else:
            gold_labels = ["?"]*token_ids.shape[0]
        for sequence, gold_label, pred_label in list(zip(recovered_sequences, gold_labels, y_preds)):
            seq_colored = re.sub("(<e[12]>.*?</e[12]>)", colors.color("\g<1>", fg='yellow'), sequence)
            labels_color = 'blue' if gold_label == pred_label == 'NO_RELATION' \
                     else 'green' if gold_label == pred_label \
                     else 'red'
            lab_colored = colors.color(f"{gold_label}/{pred_label}", fg=labels_color)
            print(f"{seq_colored} -> {lab_colored}\n")
        return
        print("{:15}{:5}\t {}\n".format("Token", "Gold", "Pred"))
        print("-"*30)
        for i, (recovered_token, y_pred) in enumerate(zip(recovered_tokens, y_preds[0].tolist())):
            #print("{:15}{}\t{}".format(words[w-1], self.rev_labels_map(golds[i]), self.rev_labels_map(y_pred)))
            print("{:15}{}\t{}".format(recovered_token, padded_golds[i] if padded_golds is not None else "---", self.rev_labels_map[y_pred]))

    def predict_single(self, input_text):
        # token_ids = self.data_provider.tokenizer.convert_ids_to_tokens(padded_token_ids)
        #padded_token_ids = pad_seq([token_ids], maxlen=self.config['max_seq_len'],
        #                           padding='pre', truncating='post', value=0)[0].tolist()
        # Input text: Hellow <e>pierwsza</e>encja<e>druga</e>encja
        # Output text: the output of tokenize_encoded_xml
        # Output data: .....
        input_text = re.sub("<e[1-9].*?>", "<ENT>", input_text)
        input_text = re.sub("</e[1-9]>", "</entity>", input_text)
        ent_iter = re.finditer("<ENT>", input_text)
        spans = []
        for ent_marker in ent_iter:
            spans.append(ent_marker.span())
        spans.reverse()
        cnt = len(spans)
        for span in spans:
            input_text = input_text[:span[0]] + f'<entity id="T01-1001.{cnt}">' + input_text[span[1]:]
            cnt -=1 
        print(input_text)
        return input_text # this is half baked

    def predict_cli(self, input_text):
        # while True:
        #     break
        raise NotImplementedError

    def save(self, path):
        self.model.save(path, overwrite=True)
        pickle.dump({'labels': self.labels_map, 'rev_labels': self.rev_labels_map},
                open(os.path.join(path, 'bert_labels.p'), 'wb'))
        print(f"Saved to {path}")

    def restore(self, path):
        self.model = tf.keras.models.load_model(path)
        rest_labels = pickle.load(open(os.path.join(path, 'bert_labels.p'), 'rb'))
        self.labels_map = rest_labels['labels']
        self.rev_labels_map = rest_labels['rev_labels']
        print(f"Restored from {path}")
