import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding

cache = {}

def get_words(embedding_file):
    words = []
    with open(embedding_file, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            words.append(word)
            
    return words


def create_tokenizer(words, filters):
    word_index = {}
    i = 2
    for word in words:
        word_index[word] = i
        i += 1
    word_index['<UNK>'] = 1
        
    tokenizer = Tokenizer(lower=True, oov_token='<UNK>', filters=filters)
    tokenizer.word_index = word_index
    
    return tokenizer


def create_embedding_layer(word_index, max_sequence_length, dimension, trainable=False, mask_zero=False):
    key = '.'.join([str(id(word_index)), str(max_sequence_length), str(dimension), str(trainable), str(mask_zero)])
    if key in cache:
        embedding_matrix = cache[key]
    else:
        embeddings_index = {}
        with open(r'data\glove\glove.6B.' + str(dimension) + 'd.txt', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        
        embedding_matrix = np.zeros((len(word_index) + 2, dimension))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
            
    embedding_layer = Embedding(len(word_index) + 2,
                            dimension,
                            weights=[embedding_matrix],
                            input_length=max_sequence_length,
                            trainable=trainable,
                            mask_zero=mask_zero)
    
    cache[key] = embedding_matrix
    return embedding_layer
            