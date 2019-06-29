class Config:
    batch_size = 32
    epoch = 10
    dropout = 0.3
    embedding_dim = 300
    embedding_size = -1
    lr = 0.001
    label_num = 5
    fast_text_hidden = 300
    word_embedding_path = '../data/glove.6B.300d.txt'
    data_path = '../data/textClassification/'
    textCnn_filter_num = 100
    textCnn_filter_size = [3, 4, 5]
