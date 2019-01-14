# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, GRU, Masking
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.layers import LSTM, Bidirectional, TimeDistributed
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.models import Sequential

from models.attention_layer import AttLayer


def fasttext_model(max_len=300,
                   vocabulary_size=20000,
                   embedding_dim=128,
                   num_classes=4):
    model = Sequential()

    # embed layer by maps vocab index into emb dimensions
    model.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=max_len))
    # pooling the embedding
    model.add(GlobalAveragePooling1D())
    # output multi classification of num_classes
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def cnn_model(max_len=400,
              vocabulary_size=20000,
              embedding_dim=128,
              hidden_dim=128,
              num_filters=512,
              filter_sizes="3,4,5",
              num_classses=4,
              dropout=0.5):
    print("Creating text CNN Model...")
    # a tensor
    inputs = Input(shape=(max_len,), dtype='int32')
    # emb
    embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim,
                          input_length=max_len, name="embedding")(inputs)
    # convolution block
    if "," in filter_sizes:
        filter_sizes = filter_sizes.split(",")
    else:
        filter_sizes = [3, 4, 5]
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=int(sz),
                             strides=1,
                             padding='valid',
                             activation='relu')(embedding)
        conv = MaxPooling1D()(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    conv_concate = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    dropout_layer = Dropout(dropout)(conv_concate)
    output = Dense(hidden_dim, activation='relu')(dropout_layer)
    output = Dense(num_classses, activation='softmax')(output)
    # model
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def rnn_model(max_len=400,
              vocabulary_size=20000,
              embedding_dim=128,
              hidden_dim=128,
              num_classes=4):
    print("Bidirectional LSTM...")
    inputs = Input(shape=(max_len,), dtype='int32')
    embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim,
                          input_length=max_len, name="embedding")(inputs)
    lstm_layer = Bidirectional(LSTM(hidden_dim))(embedding)
    output = Dense(num_classes, activation='softmax')(lstm_layer)
    model = Model(inputs, output)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()
    return model


def han_model(max_len=400,
              vocabulary_size=20000,
              embedding_dim=128,
              hidden_dim=128,
              max_sentences=16,
              num_classes=4):
    print("Hierarchical Attention Network...")
    inputs = Input(shape=(max_len,), dtype='int32')
    embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim,
                          input_length=max_len, name="embedding")(inputs)
    lstm_layer = Bidirectional(LSTM(hidden_dim, return_sequences=True))(embedding)
    lstm_layer_att = AttLayer(hidden_dim)(lstm_layer)
    sentEncoder = Model(inputs, lstm_layer_att)

    doc_inputs = Input(shape=(max_sentences, max_len), dtype='int32', name='doc_input')
    doc_encoder = TimeDistributed(sentEncoder)(doc_inputs)
    doc_layer = Bidirectional(LSTM(hidden_dim, return_sequences=True))(doc_encoder)
    doc_layer_att = AttLayer(hidden_dim)(doc_layer)
    output = Dense(num_classes, activation='softmax')(doc_layer_att)
    model = Model(doc_inputs, output)

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.summary()
    return model
