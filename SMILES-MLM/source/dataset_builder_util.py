# -*- coding: utf-8 -*-
"""
@author: NelsonRCM
"""

import pandas as pd
import numpy as np
import random
from layers_utils import *
import json
import re
from operator import itemgetter
import periodictable as pt

class dataset_builder():
    """
    Set of SMILES processing functions
    
    Args:
    - data_path [Dict]: SMILES Dataset and Dictionary Path

    """

    def __init__(self, data_path, **kwargs):
        super(dataset_builder, self).__init__(**kwargs)
        self.data_path = data_path


    # Read SMILES Dataset (CSV Format) & SMILES Dictionary (Dict Format)
    def get_data(self):
        dataset = pd.read_csv(self.data_path['data'], sep='\t', memory_map=True)
        smiles_dictionary = json.load(open(self.data_path['smiles_dict']))

        return (dataset, smiles_dictionary,)


    # Convert SMILES Tokens to Integer Values using the SMILES Dictionary and padded up to a maximum value of max_len
    def data_conversion(self, data, dictionary, max_len):
        keys = list(i for i in dictionary.keys() if len(i) > 1)

        if len(keys) == 0:
            data = pd.DataFrame([list(i) for i in data])

        else:
            char_list = []
            for i in data:
                positions = []
                for j in keys:
                    positions.extend([(k.start(), k.end() - k.start()) for k in re.finditer(j, i)])

                positions = sorted(positions, key=itemgetter(0))

                if len(positions) == 0:
                    char_list.append(list(i))

                else:
                    new_list = []
                    j = 0
                    positions_start = [k[0] for k in positions]
                    positions_len = [k[1] for k in positions]

                    while j < len(i):
                        if j in positions_start:
                            new_list.append(str(i[j] + i[j + positions_len[positions_start.index(j)] - 1]))
                            j = j + positions_len[positions_start.index(j)]
                        else:
                            new_list.append(i[j])
                            j = j + 1
                    char_list.append(new_list)

            data = pd.DataFrame(char_list)

        data.replace(dictionary, inplace=True)

        data = data.fillna(0)
        if len(data.iloc[0,:]) == max_len:
            return data
        elif len(data.iloc[0,:]) < max_len:
            zeros_array = np.zeros(shape=(len(data.iloc[:,0]),max_len-len(data.iloc[0,:])))
            data = pd.concat((data,pd.DataFrame(zeros_array)),axis=1)
            return data
        elif len(data.iloc[0,:]) > max_len:
            data = data.iloc[:,:max_len]
            return data


    # Convert SMILES DataFrame Format to Tensor Format
    def transform_dataset(self, smiles_column, smiles_max_len):

        smiles_data = self.data_conversion(self.get_data()[0][smiles_column],
                                           self.get_data()[1], smiles_max_len).astype('int64')

        return tf.convert_to_tensor(smiles_data, dtype=tf.int32)


 
# Traditional MLM masking setup used in BERT: masks 15% of the tokens of the input sequence
# 15% : 80% replaced with MASK Token, 10% with a random SMILES token, 10% remains unaltered
def create_mask(FLAGS, tokens, rng, max_preds):
    tokens_copy = np.copy(tokens)
    labels = -1 * np.ones(tokens.shape, dtype=int)
    indexes = [i for i in range(len(tokens))][1:]

    rng.shuffle(indexes)

    count = 0

    for idx in indexes:
        if count >= max_preds:
            break

        else:
            if rng.random() < 0.8:
                count += 1
                tokens_copy[idx] = FLAGS.smiles_dict_len + 2
                labels[idx] = 1

            else:
                if rng.random() < 0.5:
                    count += 1
                    labels[idx] = 1

                else:
                    count += 1
                    tokens_copy[idx] = rng.randint(1, FLAGS.smiles_dict_len)
                    labels[idx] = 1

    return tokens_copy, labels


def proc_data_mask(FLAGS, rng=random.Random(12345), mask_prob=0.15):
    data_proc = [tf.convert_to_tensor(i.numpy()) for i in load_proc_data()][0]

    labels = -1 * np.ones(data_proc.shape, dtype=int)
    y_labels = np.copy(data_proc)
    data_proc_masked = np.copy(data_proc)
    sample_weights = np.ones(labels.shape)

    for i in range(len(data_proc)):
        if i % 10000 == 0:
            print(i)
        num_to_predict = max(1, int(round(len(data_proc[i][data_proc[i] != 0]) * mask_prob)))
        t_copy, l_copy = create_mask(FLAGS, data_proc[i][data_proc[i] != 0], rng, num_to_predict)

        labels[i, :len(data_proc[i][data_proc[i] != 0])] = l_copy
        data_proc_masked[i, :len(data_proc[i][data_proc[i] != 0])] = t_copy

        # sample_weights[i,:][labels[i,:]==-1] = 0
        #
        # data_example = process_data_mask(data_proc_masked[i][None,:], y_labels[i][None,:], sample_weights[i][None,:])
        #
        # with tf.io.TFRecordWriter('./data/data_mask.tfrecords') as writer:
        #     writer.write(data_example)

    sample_weights[labels == -1] = 0

    del data_proc
    return data_proc_masked, y_labels, sample_weights



#-----------TFRecords Save and Load Functions-----------


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.numpy()]))


def mask_serialize(data, labels, weights):
    feature_data = {
        'data': _bytes_feature(data)
    }
    feature_label = {
        'labels': _bytes_feature(labels)
    }
    feature_weights = {
        'weights': _bytes_feature(weights)
    }

    data_example = tf.train.Example(features=tf.train.Features(feature=feature_data))
    labels_example = tf.train.Example(features=tf.train.Features(feature=feature_label))
    weights_example = tf.train.Example(features=tf.train.Features(feature=feature_weights))

    return data_example.SerializeToString(), labels_example.SerializeToString(), weights_example.SerializeToString()


def process_data_mask(data, labels, weights):
    serialized_data = tf.io.serialize_tensor(tf.convert_to_tensor(data, dtype=tf.int32))
    serialized_labels = tf.io.serialize_tensor(tf.convert_to_tensor(labels, dtype=tf.int32))
    serialized_weights = tf.io.serialize_tensor(tf.convert_to_tensor(weights, dtype=tf.float32))

    # data_example, labels_example, weights_example = mask_serialize(serialized_data, serialized_labels, serialized_weights)
    return mask_serialize(serialized_data, serialized_labels, serialized_weights)


def parse_labels_mask(data_example):
    feature_description = {
        'labels': tf.io.FixedLenFeature([], tf.string)
    }

    parsed_data = tf.io.parse_single_example(
        data_example, feature_description)

    data = tf.io.parse_tensor(parsed_data['labels'], tf.int32)

    return data


def parse_weights_mask(data_example):
    feature_description = {
        'weights': tf.io.FixedLenFeature([], tf.string)
    }

    parsed_data = tf.io.parse_single_example(
        data_example, feature_description)

    data = tf.io.parse_tensor(parsed_data['weights'], tf.float32)

    return data


def parse_data_mask(data_example):
    feature_description = {
        'data': tf.io.FixedLenFeature([], tf.string)
    }

    parsed_data = tf.io.parse_single_example(
        data_example, feature_description)
    data = tf.io.parse_tensor(parsed_data['data'], tf.int32)

    return data


def save_data_mask(FLAGS, path_data='../data/data_mask.tfrecords', path_labels='../data/labels_mask.tfrecords',
                   path_weights='../data/weights_mask.tfrecords'):
    data_proc_mask, y_labels, sample_weights = proc_data_mask(FLAGS)
    data_example, labels_example, weights_example = process_data_mask(data_proc_mask, y_labels, sample_weights)

    with tf.io.TFRecordWriter(path_data) as writer:
        writer.write(data_example)

    with tf.io.TFRecordWriter(path_labels) as writer:
        writer.write(labels_example)

    with tf.io.TFRecordWriter(path_weights) as writer:
        writer.write(weights_example)




def load_data_mask(path_data='../data/data_mask.tfrecords', path_labels='../data/labels_mask.tfrecords',
                   path_weights='../data/weights_mask.tfrecords'):
    data = tf.data.TFRecordDataset(path_data)
    data = data.map(parse_data_mask)
    data = [i.numpy() for i in data][0]

    labels = tf.data.TFRecordDataset(path_labels)
    labels = labels.map(parse_labels_mask)
    labels = [i.numpy() for i in labels][0]

    weights = tf.data.TFRecordDataset(path_weights)
    weights = weights.map(parse_weights_mask)
    weights = [i.numpy() for i in weights][0]

    dataset = tf.data.Dataset.from_tensor_slices((data, labels, weights))

    return dataset



def save_proc_data(FLAGS, path='../data/chembl.tfrecords'):
    data_proc = dataset_builder(FLAGS.data_path).transform_dataset('Smiles', FLAGS.smiles_len)
    data_proc = add_reg_token(data_proc, FLAGS.smiles_dict_len)

    data_proc = tf.io.serialize_tensor(data_proc)
    feature = {'data': _bytes_feature(data_proc)}
    data_example = tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

    with tf.io.TFRecordWriter(path) as writer:
        writer.write(data_example)


def parse_proc_data(data_example, feature_name='data'):
    parsed_data = tf.io.parse_single_example(data_example, {feature_name: tf.io.FixedLenFeature([], tf.string)})

    parsed_data = tf.io.parse_tensor(parsed_data[feature_name], tf.int32)

    return parsed_data


def load_proc_data(path='../data/chembl.tfrecords'):
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(parse_proc_data)

    return dataset
