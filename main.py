# coding=utf-8

import os
import sys
import pickle
import logging.config
from model import Model
from config import WORD_VOCAB_SIZE, WORD_EMBED_SIZE
from config import POS_VOCAB_SIZE, POS_EMBED_SIZE
from config import BATCH_SIZE, MAX_LEN, NUM_CLASS
from config import RNN_CELL_SIZE, RNN_LAYERS
from config import ATT_DA, ATT_R, MLP_HIDDEN_SIZE
from config import KEEP_PROB, LEARNING_RATE
from config import LAMBDA_L2, EPOCH

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

"""Log Configuration"""
LOG_FILE = './log/train.log'
handler = logging.FileHandler(LOG_FILE, mode='w')  # 实例化handler
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
formatter = logging.Formatter(fmt)  # 实例化formatter
handler.setFormatter(formatter)  # 为handler添加formatter
logger = logging.getLogger('trainlogger')  # 获取名为tst的logger
logger.addHandler(handler)  # 为logger添加handler
logger.setLevel(logging.DEBUG)


def load_embed_matrix():
    with open('./data/ccks/save/embed_matrix.pkl', 'rb') as pkl_file:
        embed_matrix = pickle.load(pkl_file)
    return embed_matrix


def train(embed_matrix):
    print("train mode")
    model_train = Model(
        batch_size=BATCH_SIZE,
        word_vocab_size=WORD_VOCAB_SIZE,
        word_embed_size=WORD_EMBED_SIZE,
        pos_vocab_size=POS_VOCAB_SIZE,
        pos_embed_size=POS_EMBED_SIZE,
        max_len=MAX_LEN,
        num_class=NUM_CLASS,
        embedding_matrix_init=embed_matrix,
        rnn_cell_size=RNN_CELL_SIZE,
        rnn_layers=RNN_LAYERS,
        att_da=ATT_DA,
        att_r=ATT_R,
        mlp_hidden_size=MLP_HIDDEN_SIZE,
        learning_rate=LEARNING_RATE,
        lambda_l2=LAMBDA_L2,
        grad_clip=6.0)

    model_train.build_graph()
    print("train_model has been built")
    model_train.train(epoch_total=EPOCH, keep_prob=KEEP_PROB)


def test(embed_matrix):
    print("test mode")
    model_test = Model(
        batch_size=BATCH_SIZE,
        word_vocab_size=WORD_VOCAB_SIZE,
        word_embed_size=WORD_EMBED_SIZE,
        pos_vocab_size=POS_VOCAB_SIZE,
        pos_embed_size=POS_EMBED_SIZE,
        max_len=MAX_LEN,
        num_class=NUM_CLASS,
        embedding_matrix_init=embed_matrix,
        rnn_cell_size=RNN_CELL_SIZE,
        rnn_layers=RNN_LAYERS,
        att_da=ATT_DA,
        att_r=ATT_R,
        mlp_hidden_size=MLP_HIDDEN_SIZE,
        learning_rate=LEARNING_RATE,
        lambda_l2=LAMBDA_L2,
        grad_clip=6.0)

    model_test.build_graph()
    print("test_model has been built")
    model_test.test(is_predict=True)
    print("model tested")


if __name__ == '__main__':
    print()
    try:
        option = sys.argv[1]
        if option == "-train":
            embed_matrix = load_embed_matrix()
            logger.debug("word2vec restored")
            print("word2vec restored")
            train(embed_matrix)

        elif option == "-test":
            embed_matrix = load_embed_matrix()
            logger.debug("word2vec restored")
            print("word2vec restored")
            test(embed_matrix)

        else:
            print("wrong option")
            print("use -train to train model")
            print("use -test to prediction")

    except IndexError:
        print("wrong option")
        print("use -train to train model")
        print("use -test to prediction")
