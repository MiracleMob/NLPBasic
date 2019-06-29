import spacy
from textClassification.utils.dataset import ClassificationDataSet
from torchtext.data import Field
from torchtext.vocab import Vectors
from functools import partial
import torch
spacy_en = spacy.load('en')
class dataProcessor():
    def __init__(self, config):
        self.config = config


    def preprocess(self):

        def tokenizer(text):
            return [tok.text for tok in spacy_en.tokenizer(text)]
        TEXT = Field(sequential=True, tokenize=tokenizer, lower=True,
                     include_lengths=True)
        LABEL = Field(sequential=False, use_vocab=False)

        train_fields = [('PhraseId', None), ('SentenceId', None),
                        ('Phrase', TEXT), ('Sentiment', LABEL)]
        test_fields = [('PhraseId', None), ('SentenceId', None),
                       ('Phrase', TEXT)]
        data_set = ClassificationDataSet(config=self.config)
        data_set.create_trainSet(train_fields)
        data_set.create_testSet(test_fields)

        train_set = data_set.train_set
        test_set = data_set.test_set

        if self.config.word_embedding_path:
            pretrained_embedding = Vectors(self.config.word_embedding_path,
                                           '.', unk_init=partial(
                                                torch.nn.init.uniform,
                                                a=-0.15, b=0.15))
            TEXT.build_vocab(train_set, test_set, vectors=pretrained_embedding)

        else:
            TEXT.build_vocab(train_set, test_set)

        field_list = (TEXT, LABEL)
        data_list = (train_set, test_set)

        return field_list, data_list











        