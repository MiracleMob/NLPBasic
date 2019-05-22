from torchtext.data import TabularDataset
from torchtext.data import Field

class ClassificationDataSet(object):

    def __init__(self, config):
        self.config = config
        self.train_set = None
        self.dev_set = None
        self.test_set = None


    def create_trainSet(self, fields):
        self.train_set = TabularDataset(path=self.config.data_path +
                                       'train.tsv',
                                  format='tsv',
                                  skip_header=True,
                                  fields=fields)


    def create_devSet(self, fields):
        self.dev_set = TabularDataset(path=self.config.data_path +
                                       'dev.tsv',
                                  format='tsv',
                                  skip_header=True,
                                  fields=fields)


    def create_testSet(self, fields):
        self.test_set = TabularDataset(path=self.config.data_path +
                                       'test.tsv',
                                  format='tsv',
                                  skip_header=True,
                                  fields=fields)



    
