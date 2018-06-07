import pickle


if __name__ == '__main__':
    with open('../data/ccks/train/label.pkl', 'rb') as label_file:
        label = list(pickle.load(label_file))

    print("length of label: ", len(label))
    print("number of 1: ", sum(label))
