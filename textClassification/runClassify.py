from textClassification.utils.dataprocessor import dataProcessor
from textClassification.utils.config import Config
from textClassification.model.textCNN import textCNN
from textClassification.model.fastText import fastText
from torchtext.data import BucketIterator
import torch.nn.functional as F
import torch.nn as nn
import torch
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score



class Runner(object):

    def __init__(self):
        parser = argparse.ArgumentParser(description="neural networks trainer")
        parser.add_argument("--test", help="validation set")
        parser.add_argument("--train", help="training set")
        parser.add_argument("--dev", help="development set")
        parser.add_argument("--webd", help="word embedding", required=False)

        parser.add_argument("--batch", help="batch size", default=512,
                            type=int)
        parser.add_argument("--epochs", help="n of epochs",
                            default=10, type=int)

        parser.add_argument("--seed", help="RNG seed", default=42, type=int)
        parser.add_argument("--optimizer", default="adam")
        parser.add_argument("--lr", default=0.001, type=float)

        parser.add_argument("--out", help="output model path",
                            default="output")
        parser.add_argument("--model", default="textcnn")
        parser.add_argument("--device", default="cpu")

        self.args = parser.parse_args()
        self.config = Config
        # self.device = torch.device(self.args.device)
        self.device = torch.device(self.args.device)


    def loss_fn(self, logits, label):

        loss = nn.CrossEntropyLoss()(logits, label)

        return loss

    def evaluate(self, all_label, all_predict):

        p = precision_score(all_label, all_predict)

        r = recall_score(all_label, all_predict)

        f1 = f1_score(all_label, all_predict, average='macro')

        return p, r, f1

    def train(self, data_iter, model, optimizer):
        for i in range(self.args.epochs):
            model.train()
            epoch_loss = 0.0
            cnt = 0
            predict_all = []
            label_all = []
            print('Epoch', i)
            for batch in data_iter:
                optimizer.zero_grad()
                text, text_len = batch.Phrase
                #[batch, seq_len]
                text = text.transpose(0, 1)
                label = batch.Sentiment
                cnt += 1
                logits, predict = model(text)
                predict = predict.tolist()

                assert len(predict) == len(label)
                predict_all.extend(predict)

                loss = self.loss_fn(logits, label)
                label = label.tolist()
                label_all.extend(label)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()


            epoch_loss = epoch_loss / cnt
            p, r, f1 = self.evaluate(label_all, predict_all)
            print("\nEpoch", i, " train loss: ", epoch_loss,
                  "\ntrain p: ", p,
                  " train r: ", r,
                  " train f1: ", f1)

        # model.save_model()

        return model


    def evaluate(self, all_label, all_predict):

        p = precision_score(all_label, all_predict, average='macro')

        r = recall_score(all_label, all_predict, average='macro')

        f1 = f1_score(all_label, all_predict, average='macro')


        return p ,r ,f1

    def predict(self, data_iter, model):

        all_predict = []

        for batch in data_iter:
            text, text_len = batch.Phrase
            text = text.transpose(0, 1)

            _, predict = model(text)
            predict = predict.tolist()
            all_predict.extend(predict)

        return all_predict



    def run(self):
        field_list, data_list = dataProcessor(Config).preprocess()
        TEXT, LABEL = field_list
        train_set, test_set = data_list
        embedding_matrix = TEXT.vocab.vectors



        train_iter = BucketIterator(train_set,
                                    batch_size=self.args.batch,
                                    train=True,
                                    shuffle=True,
                                    device=-1,
                                    sort_key=lambda x: len(x.Phrase)
                                    )
        # dev_iter = BucketIterator(dev_set,
        #                           batch_size=self.arg.batch,
        #                           train=False,
        #                           shuffle=True,
        #                           device=-1,
        #                           sort_key=lambda x: len(x.Phrase))
        test_iter = BucketIterator(test_set,
                                   batch_size=self.args.batch,
                                   train=False,
                                   shuffle=False,
                                   device=-1,
                                   sort=False)

        if self.config.embedding_size == -1:
            self.config.embedding_size = len(TEXT.vocab.itos)

        if self.args.model == "textcnn":
            model = textCNN(self.config, self.device, pretrain_embedding=embedding_matrix)
        elif self.args.model == "fastext":
            model = fastText(self.config, self.device, pretrain_embedding=embedding_matrix)
        else:
            print('None model!')

        if self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters_requires_grads(),
                                         lr=self.args.lr)
        elif self.args.optimizer == "adadelta":
            optimizer = torch.optim.Adadelta(model.parameters_requires_grads(),
                                             lr=self.args.lr)
        else:
            optimizer = torch.optim.SGD(model.parameters_requires_grads(),
                                        lr=self.args.lr,
                                        momentum=0.9)
        print(optimizer)
        model = self.train(train_iter, model, optimizer)
        all_predict = self.predict(test_iter, model)

        print(all_predict[0:10])







if __name__ == '__main__':
    Runner().run()
