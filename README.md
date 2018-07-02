# Natural-Language-Inference
NLI_by_shenjikun

### Useage ###
(1) data process
python utils.py -div
python utils.py -vocab
python utils.py -trans

(2) train word embedding
python word2vec.py

(3) train
python main.py -train

(4)test
python main.py -test

## data ##
./data/atec/train
#
./data/atec/save/word_box.txt
./data/atec/save/sentence_for_train_embedding.txt
./data/atec/save/word2id.pkl
./data/atec/save/id2word.pkl


