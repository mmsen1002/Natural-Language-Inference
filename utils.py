#coding=utf-8
import os
import sys
import re
import jieba
import pickle
import jieba.posseg as psg
import numpy as np
import tensorflow as tf
from collections import Counter
from collections import OrderedDict
from config import WORD_VOCAB_SIZE, MAX_LEN


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


def pos_seg(sentence):
    tokens = []
    words = []
    pos = []
    for item in psg.cut(corpus_filter(sentence)):
        tokens.append(item.word.strip() + '/' + item.flag.strip())
        words.append(item.word)
        pos.append(item.flag)
    return tokens, words, pos


def write_pickle(filename, _data):
    with open(filename, 'wb') as wf:
        pickle.dump(_data, wf, 0)


def devide_data(read_filename, is_train=True):
    if is_train:
        fold_name = './data/ccks/train/'
    else:
        fold_name = './data/ccks/dev/'

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
    labels = []
    word_box = []
    pos_box = []

    with open(read_filename, 'r', encoding='utf-8') as rf:
        for line in rf.readlines():
            if is_train:
                query_1, query_2, label = line.strip().split('\t')
                labels.append(int(label))

                p_with_pos, p, p_pos = pos_seg(query_1)
                q_with_pos, q, q_pos = pos_seg(query_2)
                word_box.extend(p)
                word_box.extend(q)
                pos_box.extend(p_pos)
                pos_box.extend(q_pos)

                p_str = ' '.join(p)
                q_str = ' '.join(q)
                print(p_str.strip(), file=sent_file)
                print(q_str.strip(), file=sent_file)
                print(label.strip(), file=label_file)
            else:
                query_1, query_2, = line.strip().split('\t')
                labels.append(0)
                p_with_pos, _, _ = pos_seg(query_1)
                q_with_pos, _, _ = pos_seg(query_2)

            min_len = min((min_len, len(p_with_pos), len(q_with_pos)))
            max_len = max((max_len, len(p_with_pos), len(q_with_pos)))

            p_with_pos_str = ' '.join(p_with_pos)
            q_with_pos_str = ' '.join(q_with_pos)

            print(p_with_pos_str.strip(), file=passage_file)
            print(q_with_pos_str.strip(), file=query_file)

    pickle.dump(np.array(labels), label_pkl_file, 0)

    passage_file.close()
    query_file.close()
    label_file.close()
    label_pkl_file.close()
    sent_file.close()

    print("min length: %d" % min_len)
    print("max length: %d" % max_len)

    if is_train:
        build_vocab_and_save(
            word_box,
            './data/ccks/save/word2id.pkl',
            './data/ccks/save/id2word.pkl')
        build_vocab_and_save(
            pos_box,
            './data/ccks/save/pos2id.pkl',
            './data/ccks/save/id2pos.pkl',
            is_pos=True)


def build_vocab_and_save(token_box, token2id_file_name,
                         id2token_file_name, is_pos=False):
    token2id = OrderedDict()

    if is_pos:
        token_num = Counter(token_box).most_common()
        print("length of pos_dict: %d" % (len(token_num)+2))
        token2id['UNK_POS'] = 0
        token2id['PAD_POS'] = 1
    else:
        token_num = Counter(token_box).most_common(WORD_VOCAB_SIZE-2)
        print("length of total_word_dict: %d" % (len(token_num)+2))
        token2id['UNK'] = 0
        token2id['PAD'] = 1

    idx = 2
    for token, _ in token_num:
        token2id[token] = idx
        idx += 1
    id2token = OrderedDict(zip(token2id.values(), token2id.keys()))

    # write token2id
    output_token2id = open(token2id_file_name, 'wb')
    pickle.dump(token2id, output_token2id, 0)
    output_token2id.close()

    # write id2token
    output_id2token = open(id2token_file_name, 'wb')
    pickle.dump(id2token, output_id2token, 0)
    output_id2token.close()


def transfer_to_id(read_file_name, write_file_name,
                   length_file_name, pos_file_name,
                   word2id, pos2id):

    with open(read_file_name, 'r', encoding='utf-8') as rf:
        sentence_lst = []
        length_lst = []
        pos_lst = []
        for line in rf.readlines():
            sentence = []
            pos = []
            for token in line.split():
                word, tag = token.split('/')
                if word not in word2id:
                    sentence.append(word2id['UNK'])
                    pos.append(pos2id['UNK_POS'])
                else:
                    sentence.append(word2id[word])
                    if tag in pos2id:
                        pos.append(pos2id[tag])
                    else:
                        pos.append(pos2id['UNK_POS'])

            real_length = len(line.split())
            if real_length > MAX_LEN:
                sentence = sentence[:MAX_LEN]
                pos = pos[:MAX_LEN]
                real_length = MAX_LEN
            elif real_length < MAX_LEN:
                for _ in range(MAX_LEN - real_length):
                    sentence.append(word2id['PAD'])
                    pos.append(pos2id['PAD_POS'])

            assert len(sentence) == len(pos)
            sentence_lst.append(sentence)
            pos_lst.append(pos)
            length_lst.append(real_length)

    assert len(sentence_lst) == len(pos_lst) == len(length_lst)

    with open(write_file_name, 'wb') as write_file:
        pickle.dump(np.array(sentence_lst), write_file, 0)

    with open(pos_file_name, 'wb') as pos_file:
        pickle.dump(np.array(pos_lst), pos_file, 0)

    with open(length_file_name, 'wb') as length_file:
        pickle.dump(np.array(length_lst), length_file, 0)


def read_dataset(
        passage_filename, passage_pos_filename, passage_len_filename,
        query_filename, query_pos_filename, query_len_filename,
        label_filename, batch_size, is_train=True, is_shuffle=True):

    # read data from pkl file
    passage_file = open(passage_filename, 'rb')
    passage_pos_file = open(passage_pos_filename, 'rb')
    passage_len_file = open(passage_len_filename, 'rb')

    query_file = open(query_filename, 'rb')
    query_pos_file = open(query_pos_filename, 'rb')
    query_len_file = open(query_len_filename, 'rb')

    label_file = open(label_filename, 'rb')

    # load data with type of np.ndarray
    passage = pickle.load(passage_file)
    passage_pos = pickle.load(passage_pos_file)
    passage_len = pickle.load(passage_len_file)

    query = pickle.load(query_file)
    query_pos = pickle.load(query_pos_file)
    query_len = pickle.load(query_len_file)
    label = pickle.load(label_file)
    assert len(passage) == len(passage_pos) == len(passage_len) \
            == (len(query)) == len(query_pos) == len(query_len) \
            == len(label)

    passage_file.close()
    passage_pos_file.close()
    passage_len_file.close()
    query_file.close()
    query_pos_file.close()
    query_len_file.close()
    label_file.close()

    if is_train and is_shuffle:
        # shuffle data
        length = len(passage)
        indices = np.random.permutation(np.arange(length))
        passage = passage[indices]
        passage_pos = passage_pos[indices]
        passage_len = passage_len[indices]

        query = query[indices]
        query_pos = query_pos[indices]
        query_len = query_len[indices]
        label = label[indices]

    if is_train:
        pt = int(len(passage)*0.9)
        # create dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (passage[:pt], passage_pos[:pt], passage_len[:pt],
             query[:pt], query_pos[:pt], query_len[:pt], label[:pt]))

        valid_dataset = tf.data.Dataset.from_tensor_slices(
            (passage[pt:], passage_pos[pt:], passage_len[pt:],
             query[pt:], query_pos[pt:], query_len[pt:], label[pt:]))

        train_dataset = train_dataset.shuffle(90000).batch(batch_size)
        valid_dataset = valid_dataset.shuffle(10000).batch(batch_size)

        dataset = (train_dataset, valid_dataset)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(
            (passage, passage_pos, passage_len,
             query, query_pos, query_len, label))
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
    print(queries[4])


if __name__ == '__main__':
    try:
        option = sys.argv[1]
        if option == "-d":
            devide_data('./data/ccks/origin_data/train.txt', is_train=True)
            devide_data('./data/ccks/origin_data/dev.txt', is_train=False)
        elif option == "-t":
            with open('./data/ccks/save/word2id.pkl', 'rb') as word2id_file:
                _word2id = pickle.load(word2id_file)
            with open('./data/ccks/save/pos2id.pkl', 'rb') as pos2id_file:
                _pos2id = pickle.load(pos2id_file)
            # train dataset
            transfer_to_id(
                read_file_name='./data/ccks/train/passage.txt',
                write_file_name='./data/ccks/train/passage.pkl',
                length_file_name='./data/ccks/train/passage_length.pkl',
                pos_file_name='./data/ccks/train/passage_pos.pkl',
                word2id=_word2id,
                pos2id=_pos2id)
            transfer_to_id(
                read_file_name='./data/ccks/train/query.txt',
                write_file_name='./data/ccks/train/query.pkl',
                length_file_name='./data/ccks/train/query_length.pkl',
                pos_file_name='./data/ccks/train/query_pos.pkl',
                word2id=_word2id,
                pos2id=_pos2id)

            # test dataset
            transfer_to_id(
                read_file_name='./data/ccks/dev/passage.txt',
                write_file_name='./data/ccks/dev/passage.pkl',
                length_file_name='./data/ccks/dev/passage_length.pkl',
                pos_file_name='./data/ccks/dev/passage_pos.pkl',
                word2id=_word2id,
                pos2id=_pos2id)
            transfer_to_id(
                read_file_name='./data/ccks/dev/query.txt',
                write_file_name='./data/ccks/dev/query.pkl',
                length_file_name='./data/ccks/dev/query_length.pkl',
                pos_file_name='./data/ccks/dev/query_pos.pkl',
                word2id=_word2id,
                pos2id=_pos2id)

        elif option == "-c":
            check_data('./data/ccks/train/passage.pkl')
            check_data('./data/ccks/train/passage_pos.pkl')
            check_data('./data/ccks/train/query.pkl')
            check_data('./data/ccks/train/query_pos.pkl')
            print()
            check_data('./data/ccks/dev/passage.pkl')
            check_data('./data/ccks/dev/passage_pos.pkl')
            check_data('./data/ccks/dev/query.pkl')
            check_data('./data/ccks/dev/query_pos.pkl')

        else:
            print("wrong option")
            print("use -devide to devide train and test dataset")
            print("use -vocab to build vocabulary and translate words to ids")
            print("use -check to check train and test data in ids")

    except IndexError:
        print("wrong option")
        print("use -d to devide train and test dataset")
        print("use -t to build vocabulary and translate words to ids")
        print("use -c to check train and test data in ids")
