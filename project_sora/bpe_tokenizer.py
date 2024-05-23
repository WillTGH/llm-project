from tokenizers import Tokenizer
from tokenizers.normalizers import (Sequence, Lowercase, NFD, StripAccents)
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import BPE
from tokenizers.decoders import BPEDecoder

from tokenizers.processors import TemplateProcessing

from tokenizers.trainers import BpeTrainer

import nltk
from nltk.data import find
from nltk.corpus import gutenberg

nltk.download('gutenberg', download_dir='V:/llm-project/datasets')
try:
    find('V:/llm-project/datasets/corpora/gutenberg')
    print('Corpora Gutenberg is There')
except LookupError:
    print('Corpora Gutenberg is Not There')
    
vocab_size = 1000000

class BPETokenizer():
    def __init__(self, vocab_size, text = None):
        self.plays = ['shakespeare-macbeth.txt','shakespeare-hamlet.txt','shakespeare-caesar.txt']
        self.shakespeare = [" ".join(s) for ply in self.plays for s in gutenberg.sents(ply)]

        self.special_tokens=["[UNK]","[CLS]","[SEP]","[PAD]","[MASK]"]
        self.temp_proc = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.special_tokens.index("[CLS]")),
                ("[SEP]", self.special_tokens.index("[SEP]")),
            ],
        )
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.normalizer = Sequence([NFD(),Lowercase(),StripAccents()])
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.decoder = BPEDecoder()
        self.tokenizer.post_processor=self.temp_proc

        print(len(self.shakespeare))
        print(self.shakespeare[100])

    def tokenizer_train(self):
        trainer = BpeTrainer(vocab_size=self.vocab_size,special_tokens=self.special_tokens)
        self.tokenizer.train_from_iterator(self.shakespeare, trainer=trainer)

        print(f"Trained vocab size: {self.tokenizer.get_vocab_size()}")

    def tokenize(self, text):
        #text = "in the village churches the medals won at Waterloo were hung up by those of Grossbehren and Leipzig."
        sen_enc=self.tokenizer.encode(text)
        return sen_enc.tokens
        # print(f"Output: {sen_enc.tokens}")