# coding=utf-8

import os
import sys
import pickle
import logging.config
from model import Model

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

"""Log Configuration"""
LOG_FILE = './log/train.log'
handler = logging.FileHandler(LOG_FILE, mode='w')  # 实例化handler
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
formatter = logging.Formatter(fmt)  # 实例化formatter
handler.setFormatter(formatter)  # 为handler添加formatter
logger = logging.getLogger('trainlogger')  # 获取名为tst的logger
logger.addHandler(handler)  # 为logger添加handler
logger.setLevel(logging.DEBUG)


def load_embed_matrix(fold_name, use_glove=False):
    if use_glove:
        with open(fold_name+'glove/glove_embed.pkl', 'rb') as pkl_file:
            embed_matrix = pickle.load(pkl_file)
    else:
        with open(fold_name+'gensim_word2vec/embed_matrix.pkl', 'rb') as pkl_file:
            embed_matrix = pickle.load(pkl_file)
    return embed_matrix


def train(embed_matrix):
    print("train mode")
    model_train = Model(
        embed_matrix_init=embed_matrix,
        grad_clip=6.0,
        is_train=True)

    model_train.build_graph()
    print("train_model has been built")
    model_train.train()


def test(embed_matrix):
    print("test mode")
    model_test = Model(
        embed_matrix_init=embed_matrix,
        grad_clip=6.0,
        is_train=False)

    model_test.build_graph()
    print("test_model has been built")
    model_test.test()
    print("model tested")


if __name__ == '__main__':
    print()
    try:
        option = sys.argv[1]
        if option == "-train":
            embed_matrix = load_embed_matrix(
                fold_name='./data/atec/embedding/')
            logger.debug("word2vec restored")
            print("word2vec restored")
            train(embed_matrix)

        elif option == "-test":
            embed_matrix = load_embed_matrix(
                fold_name='./data/ckks/embedding/')
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
