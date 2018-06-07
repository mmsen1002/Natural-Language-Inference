#coding=utf-8
import os
import sys
import re
import jieba
import pickle
import numpy as np
import tensorflow as tf
from collections import Counter
from collections import OrderedDict
from config import VOCAB_SIZE, MAX_LEN


def mkfile(filename):
    file_dir = os.path.split(filename)[0]
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    if not os.path.exists(filename):
        os.system(r'touch %s' % filename)


def corpus_filter(string):
    ret = re.sub('\d+(\.\d+)?', '#', string)
    punct = '！？。：；，、·~`……【】《》{}+“”‘’（）——,.!?:;()<>/@$%&*\"\''
    ret = re.sub(r'[{}]+'.format(punct), ' ', ret)
    return re.sub(' ', '', ret)


def segment(sentence):
    return list(jieba.cut(corpus_filter(sentence)))

def write_pickle(filename, _data):
    with open(filename, 'wb') as wf:
        pickle.dump(_data, wf, 0)


def devide_data(read_filename, is_train=True):
    file_dir = os.path.split(read_filename)[0]
    if is_train:
        fold_name = './data/ccks/train/'
    else:
        fold_name = './data/ccks/test/'

    mkfile(fold_name + 'passage.txt')
    passage_file = open(fold_name + 'passage.txt', 'w')
    mkfile(fold_name + 'query.txt')
    query_file = open(fold_name + 'query.txt', 'w')
    mkfile(fold_name + 'label.txt')
    label_file = open(fold_name + 'label.txt', 'w')
    mkfile(fold_name + 'label.pkl')
    label_pkl_file = open(fold_name + 'label.pkl', 'wb')
    mkfile('./data/ccks/save/sentence_for_train_embedding.txt')
    sent_file = open('./data/ccks/save/sentence_for_train_embedding.txt', 'a+')

    min_len = 10000
    max_len = 0
    with open(read_filename, 'r', encoding='utf-8') as rf:
        labels = []
        for line in rf.readlines():
            if is_train:
                query_1, query_2, label = line.strip().split('\t')
                labels.append(int(label))
            else:
                query_1, query_2, = line.strip().split('\t')
                labels.append(0)

            passage = segment(query_1)
            query = segment(query_2)

            min_len = min((min_len, len(passage), len(query)))
            max_len = max((max_len, len(passage), len(query)))

            p_str = ' '.join(passage)
            q_str = ' '.join(query)

            print(p_str.strip(), file=passage_file)
            print(q_str.strip(), file=query_file)
            print(p_str.strip(), file=sent_file)
            print(q_str.strip(), file=sent_file)

            if is_train:
                print(label, file=label_file)
            else:
                pass
        pickle.dump(np.array(labels), label_pkl_file, 0)

    passage_file.close()
    query_file.close()
    label_file.close()
    label_pkl_file.close()
    sent_file.close()

    print("min length: %d" % min_len)
    print("max length: %d" % max_len)


def build_vocab_and_save(file_name):
    # get most n frequent word in file_name.txt

    with open(file_name) as f:
        words_box = []
        for line in f:
            words_box.extend(line.strip().split())
    word_num = Counter(words_box).most_common(VOCAB_SIZE-2)
    print("length of word_num %d" % (len(word_num)+2))

    # add UNK and PAD to word dictionary
    word2id = OrderedDict()
    word2id['UNK'] = 0
    word2id['PAD'] = 1

    idx = 2
    for word, _ in word_num:
        word2id[word] = idx
        idx += 1
    id2word = OrderedDict(zip(word2id.values(), word2id.keys()))
    mkfile('./data/ccks/save/word2id.pkl')
    output_word2id = open('./data/ccks/save/word2id.pkl', 'wb')
    pickle.dump(word2id, output_word2id, 0)
    output_word2id.close()

    mkfile('./data/ccks/save/id2word.pkl')
    output_id2word = open('./data/ccks/save/id2word.pkl', 'wb')
    pickle.dump(id2word, output_id2word, 0)
    output_id2word.close()

    return word2id

def transfer_to_id(read_file_name, write_file_name,
                   length_file_name, word2id):
    # add <UNK>,<PAD> to the file_name.txt

    with open(read_file_name, 'r', encoding='utf-8') as rf:
        sentences_lst = []
        length_lst = []
        for line in rf.readlines():
            sentence = []
            for word in line.split():
                if word not in word2id:
                    sentence.append(word2id['UNK'])
                else:
                    sentence.append(word2id[word])
            real_length = len(line.split())
            if real_length > MAX_LEN:
                sentence = sentence[:MAX_LEN]
                real_length = MAX_LEN
            elif real_length < MAX_LEN:
                for _ in range(MAX_LEN - real_length):
                    sentence.append(word2id['PAD'])
            sentences_lst.append(sentence)
            length_lst.append(real_length)

        write_file = open(write_file_name, 'wb')
        pickle.dump(np.array(sentences_lst), write_file, 0)
        length_file = open(length_file_name, 'wb')
        pickle.dump(np.array(length_lst), length_file, 0)
        write_file.close()
        length_file.close()


def read_dataset(passage_filename, len_passage_filename,
                 query_filename, len_query_filename, label_filename,
                 batch_size, is_train=True, is_shuffle=True):

    # read data from pkl file
    passage_file = open(passage_filename, 'rb')
    len_passage_file = open(len_passage_filename, 'rb')
    query_file = open(query_filename, 'rb')
    len_query_file = open(len_query_filename, 'rb')
    label_file = open(label_filename, 'rb')

    # load data with type of np.ndarray
    passage = pickle.load(passage_file)
    len_passage = pickle.load(len_passage_file)
    query = pickle.load(query_file)
    len_query = pickle.load(len_query_file)
    label = pickle.load(label_file)
    assert len(passage) == len(len_passage) \
            == (len(query)) == len(len_query) == len(label)

    passage_file.close()
    len_passage_file.close()
    query_file.close()
    len_query_file.close()
    label_file.close()
    
    dataset = tf.data.Dataset.from_tensor_slices(
        (passage, len_passage, query, len_query, label))
    if is_train:
        dataset = dataset.shuffle(100000)
    dataset = dataset.batch(batch_size)
    return dataset

    if is_train and is_shuffle:
        # shuffle data
        length = len(passage)
        indices = np.random.permutation(np.arange(length))
        passage = passage[indices]
        query = query[indices]
        len_passage = len_passage[indices]
        len_query = len_query[indices]
        label = label[indices]

    if is_train:
        pt1 = int(len(passage)*0.9)
        pt2 = int(len(passage)*1.0)
        # create dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (passage[:pt1], len_passage[:pt1],
             query[:pt1], len_query[:pt1], label[:pt1]))

        valid_dataset = tf.data.Dataset.from_tensor_slices(
            (passage[pt1:pt2], len_passage[pt1:pt2],
             query[pt1:pt2], len_query[pt1:pt2], label[pt1:pt2]))

        train_dataset = train_dataset.shuffle(10000).batch(batch_size)
        valid_dataset = valid_dataset.shuffle(10000).batch(batch_size)

        """
        # create extra evaluation data
        write_pickle('./data/ccks/extra_eval/passage.pkl', passage[pt2:])
        write_pickle('./data/ccks/extra_eval/query.pkl', query[pt2:])
        write_pickle('./data/ccks/extra_eval/passage_length.pkl', len_passage[pt2:])
        write_pickle('./data/ccks/extra_eval/query_length.pkl', len_query[pt2:])
        write_pickle('./data/ccks/extra_eval/label.pkl', label[pt2:])
        """

        return train_dataset, valid_dataset
    else:
        dataset = tf.data.Dataset.from_tensor_slices(
            (passage, len_passage, query, len_query, label))
        dataset = dataset.batch(batch_size)
        return dataset


def evaluation_rate(preds, labels):
    assert len(preds) == len(labels)
    num = len(preds)

    labels_true = sum(labels)
    preds_true = sum(preds)
    preds_correct = 0
    positive_true = 0
    for _pd, _lb in zip(preds, labels):
        if _pd == _lb:
            preds_correct += 1
        if _pd == _lb == 1:
            positive_true += 1

    precision = positive_true / preds_true
    recall = positive_true / labels_true
    F1 = (2.0 * precision * recall) / (precision + recall)
    accuracy = preds_correct / num

    return precision, recall, F1, accuracy


def check_data(filename):
    pkl_file = open(filename, 'rb')
    queries = pickle.load(pkl_file)
    pkl_file.close()
    print(filename, " shape of queries is ", np.shape(queries))
    print(queries[256])


if __name__ == '__main__':
    try:
        option = sys.argv[1]
        if option == "-devide":
            devide_data('./data/ccks/origin_data/train.txt', is_train=True)
            devide_data('./data/ccks/origin_data/dev.txt', is_train=False)
        elif option == "-vocab":
            word2id = build_vocab_and_save(
                './data/ccks/save/sentence_for_train_embedding.txt')
            # train dataset
            transfer_to_id(
                read_file_name='./data/ccks/train/passage.txt',
                write_file_name='./data/ccks/train/passage.pkl',
                length_file_name='./data/ccks/train/passage_length.pkl',
                word2id=word2id)
            transfer_to_id(
                read_file_name='./data/ccks/train/query.txt',
                write_file_name='./data/ccks/train/query.pkl',
                length_file_name='./data/ccks/train/query_length.pkl',
                word2id=word2id)

            # test dataset
            transfer_to_id(
                read_file_name='./data/ccks/test/passage.txt',
                write_file_name='./data/ccks/test/passage.pkl',
                length_file_name='./data/ccks/test/passage_length.pkl',
                word2id=word2id)
            transfer_to_id(
                read_file_name='./data/ccks/test/query.txt',
                write_file_name='./data/ccks/test/query.pkl',
                length_file_name='./data/ccks/test/query_length.pkl',
                word2id=word2id)

        elif option == "-check":
            check_data('./data/ccks/train/passage.pkl')
            check_data('./data/ccks/train/query.pkl')
            check_data('./data/ccks/test/passage.pkl')
            check_data('./data/ccks/test/query.pkl')
        
        else:
            print("wrong option")
            print("use -devide to devide train and test dataset")
            print("use -vocab to build vocabulary and translate words to ids")
            print("use -check to check train and test data in ids")
 
    except IndexError:
        print("wrong option")
        print("use -devide to devide train and test dataset")
        print("use -vocab to build vocabulary and translate words to ids")
        print("use -check to check train and test data in ids")
