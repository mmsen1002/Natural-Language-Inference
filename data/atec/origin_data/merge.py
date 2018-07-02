# coding=utf-8

def merge_data(file_name1, file_name2, write_file_name):
    with open(file_name1, 'r') as file_1:
        lines_1 = file_1.readlines()
    with open(file_name2, 'r') as file_2:
        lines_2 = file_2.readlines()
    with open(write_file_name, 'w') as write_file:
        for idx_1, line1 in enumerate(lines_1):
            print(line1.strip(), file=write_file)
        for idx_2, line2 in enumerate(lines_2):
            line = line2.strip().split('\t')
            print(line[1].strip()+'\t'+line[2].strip()+'\t'+line[3].strip(),
                  file=write_file)


if __name__ == '__main__':
    merge_data(file_name1='./atec_nlp_sim_train.csv',
               file_name2='./atec_nlp_sim_train_add.csv',
               write_file_name='./atec_train.csv')

