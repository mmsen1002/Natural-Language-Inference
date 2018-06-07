# encoding:utf-8
import tensorflow as tf
import sys

def analyse_sentence_len():
    filename = './data/ccks/sentence_for_train_embedding.txt'
    with open(filename, 'r', encoding='utf-8') as rf:
        len_lst = []
        for line in rf.readlines():
           len_lst.append(len(line.strip().split()))

    print(max(len_lst))
    print(min(len_lst))
    print(sum(len_lst)/len(len_lst))
    len_lst.sort()
    print(len_lst[int(len(len_lst)*0.95)])
    for idx, l in enumerate(len_lst):
        if l > 25:
            print((idx+1) / len(len_lst))
            break


def test_reuse():

    with tf.variable_scope(name_or_scope='test_reuse', reuse=tf.AUTO_REUSE):
        initializer = tf.random_normal_initializer(
            mean=0.0, stddev=1.0, seed=113, dtype=tf.float32)

        weights = tf.get_variable(
            name="weights",
            shape=[2, 2],
            initializer=initializer)
    return weights


def main(is_test=False, model_file='./for_test/'):
    weights = test_reuse()
    if is_test:
        ckpt = tf.train.get_checkpoint_state(model_file)
        if ckpt and ckpt.model_checkpoint_path:
            print("found model,continue training")
        else:
            print("model not found, build model later")

    with tf.Session() as sess:
        if is_test and ckpt:
            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('model restored')
        else:
            saver = tf.train.Saver(max_to_keep=1)
 
        sess.run(tf.global_variables_initializer())
        print(sess.run(weights))

        if not is_test:
            saver.save(sess=sess,
                       save_path='./for_test/'+'model.ckpt',
                       write_meta_graph=True)
            print('model saved')


def data_slice():
    import numpy as np

    passage = np.array([[1, 1, 1, 0, 0],
                                    [2, 2, 2, 2, 0],
                                    [3, 3, 0, 0, 0],
                                    [4, 4, 4, 4, 4]])
    passage_len = np.array([3, 4, 2, 5])

    query = np.array([[5, 5, 5, 0, 0],
                                  [6, 6, 6, 6, 0],
                                  [7, 7, 7, 7, 7],
                                  [8, 0, 0, 0, 0]])
    query_len = np.array([3, 4, 5, 1])

    labels = np.array([0, 1, 0, 1])

    dataset = tf.data.Dataset.from_tensor_slices(
        (passage,  passage_len, query, query_len, labels))
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(2)

    return dataset


def get_data(dataset):
    iterator = tf.data.Iterator.from_structure(
        dataset.output_types, dataset.output_shapes)
    passage, passage_length, query, query_length, labels = iterator.get_next()

    init = iterator.make_initializer(dataset)

    print("shape of passage == ", passage.shape.as_list())
    with tf.Session() as sess:
        sess.run(init)
        while True:
            try:
                psg, psg_len = sess.run([passage, passage_length])
                
                print(psg)
                print(psg_len)
                print()
            except tf.errors.OutOfRangeError:
                print("complete")
                break

def test_eval():
    from utils import evaluation_rate
    preds = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    precision, recall, F1, accuracy = evaluation_rate(preds, labels)
    print('[precision: %3.4f], [recall: %3.4f], [F1: %3.4f], [accuracy: %3.4f]'%
          precision, recall, F1, accuracy)

if __name__ == '__main__':
    # analyse_sentence_len()
    # main()
    # main(is_test=True)
    #dataset = data_slice()
    #get_data(dataset)
    """
    p_states = tf.convert_to_tensor(
                             [[[1, 1, 1, 1],
                               [2, 2, 2, 2],
                               [0, 0, 0, 0]],
                              [[3, 3, 3, 3],
                               [4, 4, 4, 4],
                               [5, 5, 5, 5]]])
    print(p_states.shape.as_list())

    ps_last = p_states[:, -1, :]
    print(ps_last.shape.as_list())

    pc = tf.concat((ps_last, ps_last), axis=-1)
    print(pc.shape.as_list())
    """

    #test_eval()
    import numpy as np
    arr = np.array([[1, 2, 3],
                    [3, 4, 5],
                    [6, 7, 8],
                    [1, 1, 1],
                    [2, 2, 2]])
    label = np.array([11, 12, 13, 14, 15])
    idxs = np.arange(len(arr))
    np.random.shuffle(idxs)
    arr_shuffled = []
    for idx in idxs:
        arr_shuffled.append(arr[idx])

    arr_shuffled = np.array(arr_shuffled)
    print(np.shape(arr_shuffled))

