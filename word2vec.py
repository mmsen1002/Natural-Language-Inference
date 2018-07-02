import os
import pickle
import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import config


def load_id2word(filename):
    pkl_file = open(filename, 'rb')
    id2word = pickle.load(pkl_file)
    pkl_file.close()

    return id2word


def word2vec(corpus_filename, word2vec_filename,
             id2word_filename, embed_size, word_vocab_size):
    """
    train word2vec
    """
    sentences = LineSentence(corpus_filename)

    model = Word2Vec(sentences, sg=1, size=embed_size,
                     window=5, min_count=5, workers=4)
    word_vectors = model.wv
    del model

    id2word = load_id2word(id2word_filename)
    if word_vocab_size != len(id2word):
        print("VOCAB_SIZE == %d" % word_vocab_size)
        print("length of id2word %d" % len(id2word))
        exit()

    embedding_lst = []
    embedding_lst.append(-1+2*(np.random.rand(embed_size))) # for UNK
    embedding_lst.append(np.array([0]*embed_size)) # for PAD

    count = 0
    for i in range(word_vocab_size):
        if i > 1:
            if id2word[i] in word_vectors:
                embedding_lst.append(word_vectors[id2word[i]])
            else:
                count += 1
                embedding_lst.append(-1+2*(np.random.rand(embed_size)))
    print("%d words out of word2vec"%count)
    assert len(embedding_lst) == len(id2word)

    out = open(word2vec_filename, 'wb')
    pickle.dump(np.array(embedding_lst), out, 0)
    out.close()


def sample_word_embedding(word2vec_filename):
    if os.path.exists(word2vec_filename):
        pkl_file = open(word2vec_filename, 'rb')
        embed_matrix = pickle.load(pkl_file)
        pkl_file.close()
        print("shape of embed_matrix is ", np.shape(embed_matrix))
        print(embed_matrix[256])
    else:
        print("embed_matrix file not exist")


if __name__ == '__main__':
    """
    word2vec(
        corpus_filename='./data/ccks/save/sentence_for_train_embedding.txt',
        word2vec_filename='./data/ccks/embedding/gensim_word2vec/embed_matrix.pkl',
        id2word_filename='./data/ccks/save/id2word.pkl',
        embed_size=config.word_embed_size,
        word_vocab_size=config.word_vocab_size)

    sample_word_embedding(
        word2vec_filename='./data/ccks/embedding/gensim_word2vec/embed_matrix.pkl')
    """
    word2vec(
        corpus_filename='./data/atec/save/sentence_for_train_embedding.txt',
        word2vec_filename='./data/atec/embedding/gensim_word2vec/embed_matrix.pkl',
        id2word_filename='./data/atec/save/id2word.pkl',
        embed_size=config.word_embed_size,
        word_vocab_size=config.word_vocab_size)

    sample_word_embedding(
        word2vec_filename='./data/atec/embedding/gensim_word2vec/embed_matrix.pkl')
