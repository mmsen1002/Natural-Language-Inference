# coding:utf-8

import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from utils import evaluation_rate
from bilstm_model import BiLSTM_Model
from config import VOCAB_SIZE, EMBED_SIZE, BATCH_SIZE
from config import MAX_LEN, NUM_CLASS
from config import RNN_CELL_SIZE, RNN_LAYERS
from config import ATT_DA, ATT_R, MLP_HIDDEN_SIZE
from config import LAMBDA_L2, EPOCH, LEARNING_RATE

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
MODEL_FILE = './save_model/'


def load_embed_matrix():
    with open('./data/ccks/save/embed_matrix.pkl', 'rb') as pkl_file:
        _embed_matrix = pickle.load(pkl_file)
    return _embed_matrix


def load_train_data(passage_filename, query_filename,
                    label_filename, passage_length_filename,
                    query_length_filename):

    # read data from pkl file
    passage_file = open(passage_filename, 'rb')
    len_passage_file = open(passage_length_filename, 'rb')
    query_file = open(query_filename, 'rb')
    len_query_file = open(query_length_filename, 'rb')
    label_file = open(label_filename, 'rb')

    # load data with type of np.ndarray
    passage = pickle.load(passage_file)
    passage_length = pickle.load(len_passage_file)
    query = pickle.load(query_file)
    query_length = pickle.load(len_query_file)
    label = pickle.load(label_file)

    passage_file.close()
    len_passage_file.close()
    query_file.close()
    len_query_file.close()
    label_file.close()


    assert len(passage) == len(query) == len(label) \
            == len(passage_length) == len(query_length)
    length = len(passage)

    return passage, query, passage_length, query_length, label


def batch_iter(passage, query, passage_length,
               query_length, label, batch_size):
    # shuffle data
    length = len(passage)
    indices = np.random.permutation(np.arange(length))
    passage_shuffle = passage[indices]
    query_shuffle = query[indices]
    passage_length_shuffle = passage_length[indices]
    query_length_shuffle = query_length[indices]
    label_shuffle = label[indices]

    # batch_iter
    left = 0 
    right = batch_size
    while right <= length:
        yield (passage_shuffle[left:right],
               query_shuffle[left:right],
               passage_length_shuffle[left:right],
               query_length_shuffle[left:right],
               label_shuffle[left:right])
  
        left = right
        right += batch_size


def load_test_data(passage_filename, query_filename,
                   passage_length_filename, query_length_filename):

    # read data from pkl file
    passage_file = open(passage_filename, 'rb')
    len_passage_file = open(passage_length_filename, 'rb')
    query_file = open(query_filename, 'rb')
    len_query_file = open(query_length_filename, 'rb')

    # load data with type of np.ndarray
    passage = pickle.load(passage_file)
    passage_length = pickle.load(len_passage_file)
    query = pickle.load(query_file)
    query_length = pickle.load(len_query_file)


    assert len(passage) == len(query) \
            == len(passage_length) == len(query_length)

    return passage, query, passage_length, query_length


def train(train_data, embed_matrix, epoch=EPOCH, is_continue=False):
    # limit the usage of gpu
    gpu_op = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    config = tf.ConfigProto(allow_soft_placement=False,
                            log_device_placement=False,
                            gpu_options=gpu_op)
    model = BiLSTM_Model(
        vocab_size=VOCAB_SIZE,
        embed_size=EMBED_SIZE,
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN,
        num_class=NUM_CLASS,
        embedding_matrix_init=embed_matrix,
        rnn_cell_size=RNN_CELL_SIZE,
        rnn_layers=RNN_LAYERS,
        mlp_hidden_size=MLP_HIDDEN_SIZE,
        learning_rate=LEARNING_RATE,
        lambda_l2=LAMBDA_L2)

    model.build_graph()
    print("train_model has been built")

    with tf.Session(config=config) as sess:
        if is_continue:
            ckpt = tf.train.get_checkpoint_state(MODEL_FILE)
            if ckpt and ckpt.model_checkpoint_path:
                print("found model, continue training")
                saver = tf.train.Saver()
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("model not found, start first-training model")
                saver = tf.train.Saver(max_to_keep=1)
                sess.run(tf.global_variables_initializer())
        else:
            print("start first-training model")
            saver = tf.train.Saver(max_to_keep=1)
            sess.run(tf.global_variables_initializer())

        for epoch_idx in range(epoch):
            valid_preds = []
            valid_labels = []
            train_preds = []
            train_labels = []
            passage, query, passage_length, query_length, label = train_data
            
            batch_idx = 0
            batch_train = batch_iter(passage, query, passage_length,
                                     query_length, label, model.batch_size)
            for passage_batch, query_batch, passage_length_batch, \
                query_length_batch, label_batch in batch_train:
                
                if batch_idx < 900:
                    # for train
                    feed_dict = {
                        model.passage: passage_batch,
                        model.query: query_batch,
                        model.passage_length: passage_length_batch,
                        model.query_length: query_length_batch,
                        model.label: label_batch,
                        model.keep_prob: 1.0
                    }

                    global_step = sess.run(model.add_global_step)

                    loss, prediction, label, _ = sess.run(
                        [model.loss, model.prediction, model.label, model.optimizer],
                        feed_dict=feed_dict)
                    train_preds.extend(list(prediction))
                    train_labels.extend(list(label))

                    if (batch_idx+1) % 100 == 0:
                        print('loss of epoch %d batch %d: %3.5f' %
                              (epoch_idx, batch_idx, loss))

                    batch_idx += 1

                else:
                    # for valid
                    feed_dict = {
                        model.passage: passage_batch,
                        model.query: query_batch,
                        model.passage_length: passage_length_batch,
                        model.query_length: query_length_batch,
                        model.label: label_batch,
                        model.keep_prob: 1.0
                    }

                    batch_preds, batch_labels = sess.run(
                        [model.prediction, model.label],
                        feed_dict=feed_dict)

                    valid_preds.extend(batch_preds)
                    valid_labels.extend(batch_labels)
                    batch_idx += 1

            train_precision, train_recall, train_F1, train_accuracy = evaluation_rate(
                train_preds, train_labels)

            print('train at epoch %d : [precision: %3.4f], '\
                  '[recall: %3.4f], [F1: %3.4f], [accuracy: %3.4f]'
                  %(epoch_idx, train_precision, train_recall, train_F1, train_accuracy))

            valid_precision, valid_recall, valid_F1, valid_accuracy = evaluation_rate(
                valid_preds, valid_labels)

            print('valid at epoch %d : [precision: %3.4f], '\
                  '[recall: %3.4f], [F1: %3.4f], [accuracy: %3.4f]'
                  %(epoch_idx, valid_precision, valid_recall, valid_F1, valid_accuracy))
            print()

            # save model per epoch
            saver.save(sess=sess,
                       save_path=MODEL_FILE + 'model.ckpt',
                       global_step=global_step,
                       write_meta_graph=True)


def evaluate(data, embed_matrix, batch_size=BATCH_SIZE):
    # limit the usage of gpu
    gpu_op = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    config = tf.ConfigProto(
        allow_soft_placement=False,
        log_device_placement=False,
        gpu_options=gpu_op)

    model = BiLSTM_Model(
        vocab_size=VOCAB_SIZE,
        embed_size=EMBED_SIZE,
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN,
        num_class=NUM_CLASS,
        embedding_matrix_init=embed_matrix,
        rnn_cell_size=RNN_CELL_SIZE,
        rnn_layers=RNN_LAYERS,
        mlp_hidden_size=MLP_HIDDEN_SIZE,
        learning_rate=LEARNING_RATE,
        lambda_l2=LAMBDA_L2)

    model.build_graph()
    print("test_model has been built")

    with tf.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_FILE)
        if ckpt and ckpt.model_checkpoint_path:
            print("found model, and restored")
            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)

            preds = []
            labels = []

            passage, query, label, passage_length, query_length = data
            length = len(passage)
            batch_idx = 1
            batch_step = batch_size
            while batch_step <= length:
                feed_dict = {
                    model.passage: passage[batch_step-batch_size:batch_step],
                    model.query: query[batch_step-batch_size:batch_step],
                    model.label: label[batch_step-batch_size:batch_step],
                    model.passage_length: passage_length[batch_step-batch_size:batch_step],
                    model.query_length: query_length[batch_step-batch_size:batch_step],
                    model.keep_prob: 1.0
                }

                batch_preds, batch_labels = sess.run(
                    [model.prediction, model.label], feed_dict=feed_dict)
                preds.extend(batch_preds)
                labels.extend(batch_labels)

            with open("./data/ccks/dev/result.txt", "w") as result_file:
                for item in preds:
                    print(item, file=result_file)

            precision, recall, F1, accuracy = evaluation_rate(
                preds, labels)

            print('[precision: %3.4f], [recall: %3.4f], [F1: %3.4f], [accuracy: %3.4f]'
                  %(precision, recall, F1, accuracy))
        else:
            print("model restored failed")


if __name__ == '__main__':
    option = sys.argv[1]
    embed_matrix = load_embed_matrix()
    if option == "-train":
        train_data = load_train_data(
            passage_filename='./data/ccks/train/passage.pkl',
            query_filename='./data/ccks/train/query.pkl',
            label_filename='./data/ccks/train/label.pkl',
            passage_length_filename='./data/ccks/train/passage_length.pkl',
            query_length_filename='./data/ccks/train/query_length.pkl')

        train(train_data, embed_matrix)

    elif option == "-test":
        test_data = load_test_data(
            passage_filename='./data/ccks/dev/passage.pkl',
            query_filename='./data/ccks/dev/query.pkl',
            passage_length_filename='./data/ccks/dev/passage_length.pkl',
            query_length_filename='./data/ccks/dev/query_length.pkl')

        evaluate(test_data, embed_matrix)

