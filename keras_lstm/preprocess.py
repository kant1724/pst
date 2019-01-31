import data

def word_tokenizer(sentence):
    return sentence.replace("\n", "").split(" ")

def create_vocabulary(train_arr, max_vocabulary_size):
    vocab = {}
    for tokens in train_arr:
        for word in tokens:
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
        vocab_list = sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]

    with open('./keras_lstm/data/vocab.enc', 'w', encoding='utf8') as fw1:
        with open('./keras_lstm/data/vocab.dec', 'w', encoding='utf8') as fw2:
            for i in range(len(vocab_list)):
                fw1.write(vocab_list[i] + "\n")
                fw2.write(str(i + 1) + "\n")

def initialize_vocabulary():
    voca = {}
    with open('./keras_lstm/data/vocab.enc', 'r', encoding='utf8') as f1:
        with open('./keras_lstm/data/vocab.dec', 'r', encoding='utf8') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
            for i in range(len(lines1)):
                enc = lines1[i].replace("\n", "")
                dec = int(lines2[i].replace("\n", ""))
                voca[enc] = dec
    return voca

def sentence_to_token_ids(sentence, vocabulary):
    words = sentence

    return [vocabulary.get(w, 0) for w in words]

def data_to_token_ids(train_arr):
    vocab = initialize_vocabulary()
    token_arr = []
    for line in train_arr:
        token_ids = sentence_to_token_ids(line, vocab)
        token_arr.append(token_ids)
    return token_arr

def prepare_custom_data():
    train_1, _ = data.getKerasVpData()
    train_2, _ = data.getKerasNormalData()
    with open('./keras_lstm/data/label.txt', 'w', encoding='utf8') as f:
        for i in range(len(train_1)):
            f.write(str(1) + "\n")
        for i in range(len(train_2)):
            f.write(str(0) + "\n")

    train = train_1 + train_2
    create_vocabulary(train, 10000)
    train_ids = data_to_token_ids(train)

    with open('./keras_lstm/data/training_data.txt', 'w', encoding='utf8') as f:
        for ids in train_ids:
            f.write(str(ids) + "\n")

    return train_ids

def process_test_data(test_arr):
    test_ids = data_to_token_ids(test_arr)
    return test_ids
