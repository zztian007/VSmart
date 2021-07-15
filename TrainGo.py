from gensim.models import Word2Vec
from keras.initializers import glorot_uniform
from keras.preprocessing import sequence
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import Experiment.Evaluation as My_evaluation
import Experiment.Models as My_model
import Experiment.AccLoss as My_acc_loss
from keras.models import load_model
from datetime import datetime
from keras import backend as K
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import joblib
from tcn import TCN

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
KTF.set_session(sess)

EMBEDDING_LEN = 80

MAX_LENGTH = 1000
CLASS_NUMBER = 6
TEST_SIZE = 0.3
VALID_SIZE = 0.5
SEED = 7
SEED = np.random.seed(SEED)

SRC_WORD2VEC = "The word2vec model path"
SRC_DATA = "The data path(.txt)"
MODEL_PATH = "The model save path/"
FONT = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}


def word_vec():
    word2vec_model = Word2Vec.load(SRC_WORD2VEC)
    vocab_list = [word for word, Vocab in word2vec_model.wv.vocab.items()]
    word_index = {" ": 0}
    word_vector = {}
    embeddings_matrix = np.zeros((len(vocab_list) + 1, word2vec_model.vector_size))
    for i in range(len(vocab_list)):
        word = vocab_list[i]
        word_index[word] = i + 1
        word_vector[word] = word2vec_model.wv[word]
        embeddings_matrix[i + 1] = word2vec_model.wv[word]
    return word_index, word_vector, embeddings_matrix


def get_data(word_index, src_data, max_length):
    file = open(src_data)
    lines = file.readlines()
    file.close()
    texts = []
    labels = []
    for line in lines:
        line_1 = line.split('#')
        texts.append(line_1[0].rstrip())
        b = line_1[1].replace("\n", "").replace("\n", "")
        label = {'v0.8': 5, 'v0.7': 4, 'v0.6': 3, 'v0.5': 2, 'v0.4': 1, 'v0.x': 0,
                 'v0.8 ': 5, 'v0.7 ': 4, 'v0.6 ': 3, 'v0.5 ': 2, 'v0.4 ': 1, 'v0.x ': 0}
        labels.append(label[b])
    labels = to_categorical(labels, CLASS_NUMBER)
    data = []
    for sentence in texts:
        new_txt = []
        for word in sentence.split(' '):
            try:
                new_txt.append(word_index[word])
            except:
                new_txt.append(0)
        data.append(new_txt)
    texts = sequence.pad_sequences(data, maxlen=max_length)
    x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=TEST_SIZE,
                                                        random_state=SEED, shuffle=True)
    x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=VALID_SIZE,
                                                        random_state=SEED, shuffle=True)
    return x_train, x_test, y_train, y_test, x_valid, y_valid


def use_text_cnn_svm():
    start = datetime.now()
    model, mm = My_model.text_cnn_svm(x_train, x_test, y_train, y_test, embeddings_matrix,
                                      MAX_LENGTH, CLASS_NUMBER, MODEL_PATH, EMBEDDING_LEN)
    interval = (datetime.now() - start).seconds
    print("The time interval:", end="")
    print(interval)
    My_acc_loss.acc_loss(model, FONT)
    x_v = mm.predict(x_valid)
    My_evaluation.model_evaluation(x_v, y_valid, joblib.load(MODEL_PATH + "TEXT_CNN_SVM.h5"),
                                   CLASS_NUMBER, "machine_learning", FONT)


def use_text_cnn_random_forest():
    start = datetime.now()
    model, mm = My_model.text_cnn_random_forest(x_train, x_test, y_train, y_test, embeddings_matrix,
                                                MAX_LENGTH, CLASS_NUMBER, MODEL_PATH, EMBEDDING_LEN)
    interval = (datetime.now() - start).seconds
    print("The time interval:", end="")
    print(interval)
    My_acc_loss.acc_loss(model, FONT)
    x_v = mm.predict(x_valid)
    My_evaluation.model_evaluation(x_v, y_valid, joblib.load(MODEL_PATH + "TEXT_CNN_RANDOM_FOREST.h5"),
                                   CLASS_NUMBER, "machine_learning", FONT)


def use_dp_cnn():
    start = datetime.now()
    model = My_model.dp_cnn(x_train, x_test, y_train, y_test, embeddings_matrix,
                            MAX_LENGTH, CLASS_NUMBER, MODEL_PATH, EMBEDDING_LEN)
    interval = (datetime.now() - start).seconds
    print("The time interval:", end="")
    print(interval)
    My_acc_loss.acc_loss(model, FONT)
    My_evaluation.model_evaluation(x_valid, y_valid, load_model(MODEL_PATH + "DP_CNN.h5"), CLASS_NUMBER, "", FONT)


def use_text_cnn():
    start = datetime.now()
    model = My_model.text_cnn(x_train, x_test, y_train, y_test, embeddings_matrix,
                              MAX_LENGTH, CLASS_NUMBER, MODEL_PATH, EMBEDDING_LEN)
    interval = (datetime.now() - start).seconds
    print("The time interval:", end="")
    print(interval)
    My_acc_loss.acc_loss(model, FONT)
    My_evaluation.model_evaluation(x_valid, y_valid, load_model(MODEL_PATH + "TEXT_CNN.h5"), CLASS_NUMBER, "", FONT)


def use_bi_lstm():
    start = datetime.now()
    model = My_model.bi_lstm(x_train, x_test, y_train, y_test, embeddings_matrix,
                             MAX_LENGTH, CLASS_NUMBER, MODEL_PATH, EMBEDDING_LEN)
    interval = (datetime.now() - start).seconds
    print("The time interval:", end="")
    print(interval)
    My_acc_loss.acc_loss(model, FONT)
    My_evaluation.model_evaluation(x_valid, y_valid, load_model(MODEL_PATH + "BI_LSTM.h5"), CLASS_NUMBER, "", FONT)


def use_cnn_lstm():
    start = datetime.now()
    model = My_model.cnn_lstm(x_train, x_test, y_train, y_test, embeddings_matrix,
                              MAX_LENGTH, CLASS_NUMBER, MODEL_PATH, EMBEDDING_LEN)
    interval = (datetime.now() - start).seconds
    print("The time interval:", end="")
    print(interval)
    My_acc_loss.acc_loss(model, FONT)
    My_evaluation.model_evaluation(x_valid, y_valid, load_model(MODEL_PATH + "CNN_LSTM.h5"), CLASS_NUMBER, "", FONT)


def use_tcn():
    start = datetime.now()
    model = My_model.tcn_(x_train, x_test, y_train, y_test, embeddings_matrix,
                          MAX_LENGTH, CLASS_NUMBER, MODEL_PATH, EMBEDDING_LEN)
    interval = (datetime.now() - start).seconds
    print("The time interval:", end="")
    print(interval)
    My_acc_loss.acc_loss(model, FONT)
    t_model = load_model(MODEL_PATH + "TCN.h5", custom_objects={'TCN': TCN, 'GlorotUniform': glorot_uniform()})
    My_evaluation.model_evaluation(x_valid, y_valid, t_model, CLASS_NUMBER, "", FONT)


if __name__ == '__main__':
    word_index, word_vector, embeddings_matrix = word_vec()
    x_train, x_test, y_train, y_test, x_valid, y_valid = get_data(word_index, SRC_DATA, MAX_LENGTH)
    use_text_cnn_svm()
    plt.clf()
    K.clear_session()

