import bpemb
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences as pad_seq
from EncjoSzukacz import EncjoSzukacz
from DataProvider import DataProviderFactory

class EncjoSzukaczLSTM(EncjoSzukacz):
    def __init__(self, *, config):
        super().__init__(config=config)
        assert type(self.data_provider.tokenizer.tokenizer_obj) == bpemb.bpemb.BPEmb, \
               "Please use SentencePiece tokens and BPEmb embeddings with EncjoSzukaczLSTM"
        assert 'engine_params' in config, "Please set 'engine_params' in the configuration file"
        assert 'layer_defs' in config['engine_params'], "Please provide layer definitions in the config file"
        self.emb_matrix = self.data_provider.tokenizer.tokenizer_obj.vectors
        self.emb_dim = self.data_provider.tokenizer.tokenizer_obj.dim
        self.word_list = self.data_provider.tokenizer.tokenizer_obj.words
        self.vocab_size = len(self.word_list)
        self.word_list[0] = "<pad>" # redefine the zeroeth index for padding (no <pad> token in BPEmb)
        self.emb_matrix[0] = np.zeros(self.emb_dim) # set padding to all zeros
        if self.config['engine_params'].get('add_positional_embeddings') is True:
            init_pos_embs = np.random.rand(len(self.data_provider.tokenizer.special_token_ids), self.emb_matrix.shape[1])
            self.emb_matrix = np.append(self.emb_matrix, init_pos_embs, axis=0)
            print(f"Appended {init_pos_embs.shape[0]} embeddings for special tokens and " \
                    "initialized them randomly. Be sure to restore them from pickle!")

        #### populated by train: ###
        self.labels_map = None
        self.rev_labels_map = None
        self.model = None

    def _mangle_token_ids(self, dataset, as_generator=False):
        """ Generates input for keras train. Also does padding and other preprocessing magic. """
        # dataset = self.data_provider.get_dataset(which)
        # self.labels_map = self.data_provider.get_entity_labels()
        # self.rev_labels_map = {v: k for k, v in self.labels_map.items()}
        all_token_ids = [ts['token_ids'] for ts in dataset.values()]
        # all_entities = [[self.labels_map[j] for j in ts['entities']] for ts in dataset.values()]
        padded_tokens = pad_seq(all_token_ids, maxlen=self.config['max_seq_len'], \
                                padding='pre', truncating='post', value=0)
        #padded_entities = pad_seq(all_entities, maxlen=self.config['max_seq_len'],
        #                        padding='pre', truncating='post', value=0)
        # Tensorflow versions earlier than 1.14 (or 2.0?) have issues with sparse_categorical_crossentropy
        # so it's safe to convert the labels into one-hot vectors for compatibility
        # padded_entities = tf.keras.utils.to_categorical(padded_entities, num_classes=len(self.labels_map))
        # return padded_tokens, padded_entities
        return padded_tokens
    
    def _mangle_ners(self, dataset, as_generator=False):
        self.labels_map = self.data_provider.get_entity_labels()
        self.rev_labels_map = {v: k for k, v in self.labels_map.items()}
        all_entities = [[self.labels_map[j] for j in ts['entities']] for ts in dataset.values()]
        padded_entities = pad_seq(all_entities, maxlen=self.config['max_seq_len'],
                                padding='pre', truncating='post', value=0)
        padded_entities = tf.keras.utils.to_categorical(padded_entities, num_classes=len(self.labels_map))
        return padded_entities

    def _mangle_inputs(self, which, as_generator=False):
        dataset = self.data_provider.get_dataset(which)
        padded_tokens = self._mangle_token_ids(dataset)
        padded_entities = self._mangle_ners(dataset)
        return padded_tokens, padded_entities

    def _build_model(self):
        model = tf.keras.Sequential()
        int_inputs = tf.keras.layers.Input(shape=(self.config['task_specific']['maxlen'],))
        model.add(int_inputs)
        model.add(tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.emb_dim,
                                            input_length=self.config['task_specific']['maxlen'],
                                            embeddings_initializer=tf.keras.initializers.Constant(self.emb_matrix),
                                            mask_zero=True,
                                            trainable=False))
        for layer in self.config['engine_params']['layer_defs']:
            if layer['layer_type'] == 'lstm':
                if layer.get('bidirectional') is True:
                    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(layer['num_units'], dropout=layer.get('dropout') or 0.0, activation = layer.get('activation') or 'tanh', return_sequences=True)))
                else:
                    model.add(tf.keras.layers.LSTM(layer['num_units'], dropout=layer.get('dropout') or 0.0, activation = layer.get('activation') or 'tanh', return_sequences=True))

            elif layer['layer_type'] == 'dense':
                l = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(layer['num_units'], activation="softmax")) 
                model.add(l)
                if layer.get('dropout') is not None:
                    model.add(tf.keras.layers.Dropout(layer['dropout']))
            else:
                raise ValueError(f"Unknown layer type {layer['layer_type']}")
        #out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(3, activation='softmax'))(model)
        #model = tf.keras.Model(int_inputs, out)
        model.summary()
        return model

    def train(self):
        self.model = self.model or self._build_model()
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        x_train, y_train = self._mangle_inputs('trainset')
        x_valid, y_valid = self._mangle_inputs('validset')
        self.model.fit(x_train, y_train, epochs=self.config['train_params'].get('num_epochs') or 1,
                  batch_size=self.config['train_params'].get('batch_size'),
                  validation_data=(x_valid, y_valid))

    def evaluate(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def _predict_from_token_ids(self, token_ids: list, golds=None):
        padded_token_ids = pad_seq([token_ids], maxlen=self.config['max_seq_len'],
                                   padding='pre', truncating='post', value=0)[0].tolist()
        padded_golds = None if golds is None else pad_seq([golds], maxlen=self.config['max_seq_len'],
                                                          padding='pre', truncating='post', value="None", dtype=object)[0].tolist()
        print(padded_token_ids)
        y_preds = self.model.predict_classes([padded_token_ids])
        recovered_tokens = self.data_provider.tokenizer.convert_ids_to_tokens(padded_token_ids)
        print("{:15}{:5}\t {}\n".format("Token", "Gold", "Pred"))
        print("-"*30)
        for i, (recovered_token, y_pred) in enumerate(zip(recovered_tokens, y_preds[0].tolist())):
            #print("{:15}{}\t{}".format(words[w-1], self.rev_labels_map(golds[i]), self.rev_labels_map(y_pred)))
            print("{:15}{}\t{}".format(recovered_token, padded_golds[i] if padded_golds is not None else "---", self.rev_labels_map[y_pred]))


    def predict_cli(self, input_text):
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
