# NER na PolBercie i innych modelach
from EncjoSzukacz import EncjoSzukacz
import nltk # min version 3.5
from transformers import *
POLBERT="dkleczek/bert-base-polish-cased-v1"
NKJP="/wymiana/Projekty/ML_Data/NKJP"
MAX_LEN=40

def demo():
    model = BertForMaskedLM.from_pretrained("dkleczek/bert-base-polish-cased-v1")
    tokenizer = BertTokenizer.from_pretrained("dkleczek/bert-base-polish-cased-v1")
    nlp = pipeline('fill-mask', model=model, tokenizer=tokenizer)
    for pred in nlp(f"Adam Mickiewicz wielkim polskim {nlp.tokenizer.mask_token} był."):
        print(pred)


class EncjoSzukaczPolbert(EncjoSzukacz):
    def __init__(self):
        # tokenizator mozna wziac z klasy DataProviderInterface
        self.tokenizer = AutoTokenizer.from_pretrained(POLBERT)
        self.tf_model = TFBertModel.from_pretrained(POLBERT, from_pt=True) # surowe outputy - uzywac wylacznie TFBertModel lub TFAutoModel, nie TF...ModelFor... bo te maja glowice
        self.trainset = None
        self.testset = None
        self.nkjp1m = None
        self.mode = 'train'
        
    # def slurp(self):
    #     if self.mode == 'train':
    #         self.nkjp1m = nltk.corpus.NKJPCorpusReader(root=NKJP) # e tam, pobożne życzenia... Poprawić trzeba w NLTK albo samemu napisać

    def train(self):
        zdania = ["Adam Mickiewicz wielkim poetą był", " Dopóki będę prezesem Narodowego Banku Polskiego, dopóty Polska nie wejdzie do mechanizmu ERM II i strefy euro – zapowiedział w środę Adam Glapiński"]        pass
        zakodowane = self.tokenizer.batch_encode_plus(zdania, padding=True, return_tensors='tf')
        input_ids = zakodowane['input_ids']
        token_types = zakodowane['token_type_ids']
        attn_masks = zakodowane['attention_mask']

    def predict(self):
        pass

    def save(self):
        pass

    def restore(self):
        pass

    def evaluate(self):
        pass


