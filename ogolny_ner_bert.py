import os
import tensorflow as tf
from transformers import *
from conll_helpers import get_examples as read_conll2003

BERT_MODEL="bert-base-cased" # make sure it's TF 2.0
MAX_LEN=256
CONLL2003_PATH=os.path.join(os.environ.get('ML_DATA'), 'CoNLL2003', 'NER')

def batch_encode_plus_and_labels_tf(bert_tokenizer, whitespace_tokenized_sentences, text_labels_per_sentence, outside_label_id=0):
    """
    Adapted from: https://github.com/chambliss/Multilingual_NER/blob/master/python/utils/main_utils.py#L11
    See this blog for original concept: https://gab41.lab41.org/lessons-learned-fine-tuning-bert-for-named-entity-recognition-4022a53c0d90

    text_labels_per_sentence should contain label ids as ints, not as text. However, no checking is done here.

    Also, from https://huggingface.co/transformers/model_doc/bert.html#tfbertmodel:
    TF 2.0 models accepts two formats as inputs:
       - having all inputs as keyword arguments (like PyTorch models), or
       - having all inputs as a list, tuple or dict in the first positional arguments.

    If you choose this second option, there are three possibilities:
       - ...
       - ...
       - a dictionary with one or several input Tensors associated to the input names given
         in the docstring: model({'input_ids': input_ids, 'token_type_ids': token_type_ids})
                                 ^^^^^^^^^^ this dict is returned in this function

    """
    tokenized = [] # text representation, not used
    input_ids = [] # dtype=int32
    token_type_ids = [] # dtype=int32, all zeros
    attn_mask = [] # dtype=int32, ones where word/token exists, zeros elsewhere
    all_labels = []

    for sentence, text_labels in zip(whitespace_tokenized_sentences, text_labels_per_sentence):
        tokens = ['[CLS]']
        labels = [outside_label_id]
        for word, label in zip(sentence, text_labels):
            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = bert_tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)
    
            # Add the tokenized word to the final tokenized word list
            tokens.extend(tokenized_word)
    
            # Add the same label to the new list of labels `n_subwords` times
            labels.extend([label] * n_subwords)
        tokens.append('[SEP]')
        labels.append(outside_label_id)
        assert len(tokens) == len(labels), f'Assertion failed. Tokens={tokens}, Labels={labels}'
        # ;STAD
        token_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
        input_ids.append(token_ids)
        token_type_ids.append([0]*len(token_ids))
        attn_mask.append([1]*len(token_ids))
        all_labels.append(labels)
        tokenized.append(tokens)

    input_ids = tf.ragged.constant(input_ids).to_tensor(default_value=0, shape=[None, MAX_LEN]) # maybe Huggingface will support ragged tensors in the puture, for now we have to convert them to dense tensors
    token_type_ids = tf.zeros(input_ids.shape, dtype=tf.int32)
    attn_mask = tf.ragged.constant(attn_mask).to_tensor(default_value=0, shape=[None, MAX_LEN])
    all_labels = tf.ragged.constant(all_labels).to_tensor(default_value=outside_label_id, shape=[None, MAX_LEN])
    
    # change last token id to SEP=102 if there is an overflow because of padding
    input_ids_last = tf.reshape(tf.map_fn(lambda p: 0 if p==0 else bert_tokenizer.sep_token_id, input_ids[:, -1:]), shape=[input_ids.shape[0], 1])
    input_ids = tf.concat([input_ids[:, :-1], input_ids_last], axis=1) #
    # nie wierze, ze powyzsze dwie linijki sam wymyslilem i napisalem...
    ret_input = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attn_mask}
    ret_labels = all_labels
    return tokenized, ret_input, ret_labels

def import_conll_datasets():
    trainset = read_conll2003(os.path.join(CONLL2003_PATH, 'train.txt'))
    validset = read_conll2003(os.path.join(CONLL2003_PATH, 'dev.txt'))
    testset  = read_conll2003(os.path.join(CONLL2003_PATH, 'test.txt'))
    train_tokens = [sent.split() for sent in trainset[0]]
    train_labels = [sent.split() for sent in trainset[1]]
    valid_tokens = [sent.split() for sent in validset[0]]
    valid_labels = [sent.split() for sent in validset[1]]
    test_tokens = [sent.split() for sent in testset[0]]
    test_labels = [sent.split() for sent in testset[1]]
    return (train_tokens,train_labels), (valid_tokens,valid_labels), (test_tokens,test_labels)

def get_labels_dict(dataset_labels):
    labels_dict = {'O': 0}
    all_data_labels = [label for sent in dataset_labels for label in sent]
    all_data_labels = set(all_data_labels)
    all_data_labels = all_data_labels - {'O'}
    for idx, label in enumerate(sorted(all_data_labels)):
        labels_dict[label] = idx+1
    return labels_dict

def main(args=None):
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    # model = BertModel.from_pretrained(BERT_MODEL)
    tf_model = TFBertModel.from_pretrained(BERT_MODEL, from_pt=True)
    (trainset, train_labels), (validset, valid_labels), (testset, test_labels) = import_conll_datasets()
    labels_dict = get_labels_dict(train_labels)
    train_labels = [[labels_dict[l] for l in sent] for sent in train_labels]
    valid_labels = [[labels_dict[l] for l in sent] for sent in valid_labels]
    test_labels = [[labels_dict[l] for l in sent] for sent in test_labels]
    print("Realigning BIO tags to subwords and generating TF train input. This can take some time")
    _, tf_train, tf_train_gold = batch_encode_plus_and_labels_tf(tokenizer, trainset, train_labels, outside_label_id=0)
    # tf_train is a dictionary containing 3 key-value pairs because TFBertModel is a tf.keras.Model with 3 inputs
    # ff_ner = tf.nn.
    bert_output = tf_model.outputs[0] # sequence of hidden states of shape (batch_size, seq_len, 768)
    # teraz mozna nadbudowac warstwe dense nad bertem:
    # wyj = tf.keras.layers.Dense(8)(tf_model.outputs[0])
    # todo: 1) wylaczyc trainable w tf_model
    #       2) zawinac *jakos* tf_model z nadbudowanym densem tak, zeby jako input mozna bylo przekazac slownik z linijki 26 (czyli to, co w drugim elemencie zwraca funkcja batch_encode_plus_and_labels_tf)
    # ALBO MOZE BEZ KERASA
    # po prostu with tf.GradientTape i gradienty tylko aplikowac do densa. Chyba prosciej bedzie niz na sile wciskac do Kerasa, zwlaszcza ze wtedy mamy kontrole nad tym co jest trainable a co nie
    return tokenizer, tf_model, tf_train, tf_train_gold



# PRZYKLAD UCZENIA Z GRADIENTTAPE:
# def step(X, y):
# 	# keep track of our gradients
# 	with tf.GradientTape() as tape:
# 		# make a prediction using the model and then calculate the
# 		# loss
# 		pred = model(X) # <----------------------------------------- TUTAJ MOZNA DAC SLOWNIK, A POTEM OUTPUT[0] PRZEKAZAC JAKO INPUT DO DENSA
# 		loss = categorical_crossentropy(y, pred)
# 	# calculate the gradients using our tape and then update the
# 	# model weights
# 	grads = tape.gradient(loss, model.trainable_variables)
# 	opt.apply_gradients(zip(grads, model.trainable_variables))
#
#
# compute the number of batch updates per epoch
# for epoch in range(0, EPOCHS):
#     for step_no in range(num_steps):
#         ...


if __name__ == "__main__":
    main()
