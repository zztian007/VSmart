from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
from keras.layers import Activation, MaxPooling1D, Add, BatchNormalization, PReLU, GlobalAveragePooling1D
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout, CuDNNLSTM, \
    Bidirectional, Lambda
from keras import Model, Input
from tcn import TCN
from keras import backend as K

filter_nr = 128
filter_size = 3
max_pool_size = 3
max_pool_strides = 2
dense_nr = 256
spatial_dropout = 0.3
dense_dropout = 0.3
conv_kern_reg = l2(0.0001)
conv_bias_reg = l2(0.0001)
BATCH_SIZE = 64
EPOCHES = 100
RP = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto',
                       min_delta=0.001, cooldown=0, min_lr=0)
ES = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', min_delta=0.00001)


# ----------------------------------------------------------------------------------------------------
def svm(x_train, y_train, path, name):
    print("SVM GO")
    clf = SVC(C=2, kernel='rbf', probability=True)
    clf.fit(x_train, y_train)
    print("SVM OVER")
    joblib.dump(clf, path + name)


def text_cnn_svm(x_train, x_test, y_train, y_test, embeddings_matrix, max_length, class_number, path, emb_length):
    embedding_layer = Embedding(input_dim=len(embeddings_matrix),
                                output_dim=emb_length,
                                weights=[embeddings_matrix],
                                input_length=max_length)
    sequence_input = Input(shape=(max_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    convs = []
    for kernel_size in [2, 3, 4]:
        c = Conv1D(128, kernel_size, activation='relu')(embedded_sequences)
        c = GlobalMaxPooling1D()(c)
        convs.append(c)
    x = Concatenate()(convs)
    x = Dropout(0.3)(x)
    x = Dense(class_number)(x)
    output = Activation('softmax')(x)
    model = Model(sequence_input, outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    checkpoint = ModelCheckpoint(path + "TEXT_CNN.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint, ES, RP]
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHES,
              callbacks=callbacks_list,
              validation_data=(x_test, y_test))
    mm = Model(inputs=model.input, outputs=model.get_layer('dense_1').output)
    x_train_output = mm.predict(x_train)
    y_train_output = y_train.argmax(axis=-1)
    svm(x_train_output, y_train_output, path, "TEXT_CNN_SVM.h5")
    return model, mm


# ----------------------------------------------------------------------------------------------------
def random_forest(x_train, y_train, path, name):
    print("Random_Forest GO")
    model = RandomForestClassifier(n_estimators=3000, max_features='auto', verbose=True, n_jobs=6)
    model.fit(x_train, y_train)
    print("Random_Forest OVER")
    joblib.dump(model, path + name)


def text_cnn_random_forest(x_train, x_test, y_train, y_test, embeddings_matrix, max_length, class_number, path, emb_length):
    embedding_layer = Embedding(input_dim=len(embeddings_matrix),
                                output_dim=emb_length,
                                weights=[embeddings_matrix],
                                input_length=max_length)
    sequence_input = Input(shape=(max_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    convs = []
    for kernel_size in [2, 3, 4]:
        c = Conv1D(128, kernel_size, activation='relu')(embedded_sequences)
        c = GlobalMaxPooling1D()(c)
        convs.append(c)
    x = Concatenate()(convs)
    x = Dropout(0.3)(x)
    x = Dense(class_number)(x)
    output = Activation('softmax')(x)
    model = Model(sequence_input, outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    checkpoint = ModelCheckpoint(path + "TEXT_CNN.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint, ES, RP]
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHES,
              callbacks=callbacks_list,
              validation_data=(x_test, y_test))
    mm = Model(inputs=model.input, outputs=model.get_layer('dense_1').output)
    x_train_output = mm.predict(x_train)
    y_train_output = y_train.argmax(axis=-1)
    random_forest(x_train_output, y_train_output, path, "TEXT_CNN_RANDOM_FOREST.h5")
    return model, mm


# ----------------------------------------------------------------------------------------------------
def cnn(x):
    block = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                   kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(x)
    block = BatchNormalization()(block)
    block = PReLU()(block)
    block = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                   kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block)
    block = BatchNormalization()(block)
    block = PReLU()(block)
    return block


def dp_cnn(x_train, x_test, y_train, y_test, embeddings_matrix, max_length, class_number, path, emb_length):
    embedding_layer = Embedding(input_dim=len(embeddings_matrix),
                                output_dim=emb_length,
                                weights=[embeddings_matrix],
                                input_length=max_length)
    sequence_input = Input(shape=(max_length,), dtype='int32')
    embedding_layer(sequence_input)
    emb_comment = embedding_layer(sequence_input)
    resize_emb = Conv1D(filter_nr, kernel_size=1, padding='same', activation='relu',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(emb_comment)
    resize_emb = PReLU()(resize_emb)
    # one
    block1 = cnn(emb_comment)
    block1_output = Add()([block1, resize_emb])
    block1_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output)
    # two
    block2 = cnn(block1_output)
    block2_output = Add()([block2, block1_output])
    block2_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block2_output)
    # three
    block3 = cnn(block2_output)
    block3_output = Add()([block3, block2_output])
    block3_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block3_output)
    # four
    block4 = cnn(block3_output)
    block4_output = Add()([block4, block3_output])
    block4_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block4_output)
    # five
    block5 = cnn(block4_output)
    block5_output = Add()([block5, block4_output])
    block5_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block5_output)
    # six
    block6 = cnn(block5_output)
    block6_output = Add()([block6, block5_output])
    block6_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block6_output)
    # seven
    block7 = cnn(block6_output)
    block7_output = Add()([block7, block6_output])
    # dense
    output = GlobalMaxPooling1D()(block7_output)
    output = Dense(dense_nr, activation='linear')(output)
    output = Dropout(dense_dropout)(output)
    output = Dense(class_number)(output)
    output = Activation('softmax')(output)
    model = Model(inputs=sequence_input, outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    checkpoint = ModelCheckpoint(path + "DP_CNN.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint, ES, RP]
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHES,
              callbacks=callbacks_list,
              validation_data=(x_test, y_test))
    return model


# ----------------------------------------------------------------------------------------------------
def text_cnn(x_train, x_test, y_train, y_test, embeddings_matrix, max_length, class_number, path, emb_length):
    embedding_layer = Embedding(input_dim=len(embeddings_matrix),
                                output_dim=emb_length,
                                weights=[embeddings_matrix],
                                input_length=max_length)
    sequence_input = Input(shape=(max_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    convs = []
    for kernel_size in [2, 3, 4]:
        c = Conv1D(128, kernel_size, activation='relu')(embedded_sequences)
        c = GlobalMaxPooling1D()(c)
        convs.append(c)
    x = Concatenate()(convs)
    x = Dropout(0.3)(x)
    output = Dense(class_number, activation='softmax')(x)
    model = Model(sequence_input, outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    checkpoint = ModelCheckpoint(path + "TEXT_CNN.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint, ES, RP]
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHES,
              callbacks=callbacks_list,
              validation_data=(x_test, y_test))
    return model


# ----------------------------------------------------------------------------------------------------
def bi_lstm(x_train, x_test, y_train, y_test, embeddings_matrix, max_length, class_number, path, emb_length):
    model = Sequential()
    model.add(Embedding(input_dim=len(embeddings_matrix),
                        output_dim=emb_length,
                        weights=[embeddings_matrix],
                        input_length=max_length,
                        ))
    model.add(Bidirectional(CuDNNLSTM(128, return_sequences=False), merge_mode='concat', name='out_layer'))
    model.add(Dropout(0.3))
    model.add(Dense(class_number, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    checkpoint = ModelCheckpoint(path + "BI_LSTM.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint, ES, RP]
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHES,
              callbacks=callbacks_list,
              validation_data=(x_test, y_test))
    return model


# ----------------------------------------------------------------------------------------------------
def cnn_lstm(x_train, x_test, y_train, y_test, embeddings_matrix, max_length, class_number, path, emb_length):
    embedding_layer = Embedding(input_dim=len(embeddings_matrix),
                                output_dim=emb_length,
                                weights=[embeddings_matrix],
                                input_length=max_length)
    sequence_input = Input(shape=(max_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    convs = []
    for kernel_size in [2, 3, 4]:
        c = Conv1D(128, kernel_size, activation='relu')(embedded_sequences)
        c = GlobalMaxPooling1D()(c)
        convs.append(c)
    x = Concatenate()(convs)
    x = Lambda(lambda m: K.expand_dims(m, axis=-1))(x)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=False), merge_mode='concat', name='out_layer')(x)
    x = Dropout(0.3)(x)
    output = Dense(class_number, activation='softmax')(x)
    model = Model(sequence_input, outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    checkpoint = ModelCheckpoint(path + "CNN_LSTM.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint, ES, RP]
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHES,
              callbacks=callbacks_list,
              validation_data=(x_test, y_test))
    return model


# ----------------------------------------------------------------------------------------------------
def tcn_(x_train, x_test, y_train, y_test, embeddings_matrix, max_length, class_number, path, emb_length):
    embedding_layer = Embedding(input_dim=len(embeddings_matrix),
                                output_dim=emb_length,
                                weights=[embeddings_matrix],
                                input_length=max_length)
    sequence_input = Input(shape=(max_length,), dtype='int32')
    emb_sequence = embedding_layer(sequence_input)
    x = TCN(nb_filters=128, kernel_size=3, return_sequences=False, activation='relu', name="TCN")(emb_sequence)
    x = Dropout(0.3)(x)
    output = Dense(class_number, activation='softmax')(x)
    model = Model(sequence_input, outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    checkpoint = ModelCheckpoint(path + "TCN.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint, ES, RP]
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHES,
              callbacks=callbacks_list,
              validation_data=(x_test, y_test))
    return model

