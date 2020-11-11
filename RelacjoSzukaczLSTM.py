import bpemb
import numpy as np
import tensorflow as tf
import pickle
import colors
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences as pad_seq
from EncjoSzukaczLSTM import EncjoSzukaczLSTM
from DataProvider import DataProviderFactory
from readers import tokenize_encoded_xml
from sklearn.metrics import classification_report

class RelacjoSzukaczLSTM(EncjoSzukaczLSTM):
    def __init__(self, *, config):
        super().__init__(config=config)

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

    def _mangle_inputs(self, which, as_generator=False):
        dataset = self.data_provider.get_dataset(which)
        padded_tokens = self._mangle_token_ids(dataset)
        relation_classes = self._mangle_rels(dataset)
        return padded_tokens, relation_classes

    def _build_model(self):
        int_inputs = tf.keras.layers.Input(shape=(self.config['task_specific']['maxlen'],))
        embs = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.emb_dim,
                                         input_length=self.config['task_specific']['maxlen'],
                                         embeddings_initializer=tf.keras.initializers.Constant(self.emb_matrix),
                                         mask_zero=True,
                                         trainable=True)(int_inputs)
        lstm_conf = self.config['engine_params']['layer_defs']['lstm_layer']
        sequ = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_conf['num_units'],
                                             dropout=lstm_conf.get('dropout') or 0.0,
                                             activation = lstm_conf.get('activation') or 'tanh',
                                             return_sequences=True))(embs)
        max_each_dim_op = tf.reduce_max(sequ, axis=1, name="choose_max_dim")
        dense_conf = self.config['engine_params']['layer_defs']['dense_layer']
        dropouted = tf.keras.layers.Dropout(dense_conf.get('dropout') or 0.0)(max_each_dim_op)
        final_dense = tf.keras.layers.Dense(len(self.data_provider.get_relations_labels()),
                                            activation='softmax')(dropouted)
        model = tf.keras.Model(inputs=int_inputs, outputs=final_dense)
        model.summary()
        return model

    def train(self):
        self.model = self.model or self._build_model()
        opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        x_train, y_train = self._mangle_inputs('trainset')
        x_valid, y_valid = self._mangle_inputs('validset')
        self.model.fit(x_train, y_train, epochs=self.config['train_params'].get('num_epochs') or 1,
                  batch_size=self.config['train_params'].get('batch_size'),
                  validation_data=(x_valid, y_valid))

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
        pickle.dump({'labels': self.labels_map, 'rev_labels': self.rev_labels_map}, open('lstm_labels.p', 'wb')) # FIXME: path
        print(f"Saved to {path}")

    def restore(self, path):
        self.model = tf.keras.models.load_model(path)
        rest_labels = pickle.load(open('lstm_labels.p', 'rb')) # FIXME: path
        self.labels_map = rest_labels['labels']
        self.rev_labels_map = rest_labels['rev_labels']
        print(f"Restored from {path}")
