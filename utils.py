"""Utility functions for loading datasets and computing performance"""
from __future__ import print_function

from random import random

import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from os.path import join

class WordEmb(object):
    """Loads the word2vec model"""
    def __init__(self, args):
        print('processing corpus ' + str(args.emb_loc))
        if args.embvocab > 0:
            self.wvec = KeyedVectors.load_word2vec_format(args.emb_loc, binary=True,
                                                          limit=args.embvocab)
        else:
            self.wvec = KeyedVectors.load_word2vec_format(args.emb_loc, binary=True)
        self.unk = np.array([random() for _ in range(self.wvec.vector_size)])
        self.is_case_sensitive = True if (self.wvec['the'] != self.wvec['The']).all() else False
        if not self.is_case_sensitive:
            print("Warning: dictionary is NOT case-sensitive")

    def __getitem__(self, word):
        if not self.is_case_sensitive:
            word = word.lower()
        try:
            return self.wvec[word]
        except KeyError:
            return self.unk

def is_capital(word):
    '''returns an array 1 if capital else 0'''
    if ord('A') <= ord(word[0]) <= ord('Z'):
        return np.array([1, 0])
    else:
        return np.array([0, 1])

def get_feature(args, word):
    '''returns features for the given word'''
    features = []
    if args.feat_cap:
        pass
    if args.feat_dict:
        pass
    return features

def get_namedentities(args, tokens, prediction):
    '''Get list of named entitiess'''
    assert len(tokens) == len(prediction)
    entities = []
    indices = []
    incon = 0
    if args.n_classes == 2:
        found = False
        entity = ''
        indstr = ''
        for i, label in enumerate(prediction):
            if label == 0:
                if found:
                    if entity != '':
                        entity += "_{}".format(i)
                        indstr += "_{}".format(tokens[i])
                else:
                    if entity == '':
                        entity = "{}".format(i)
                        indstr = "{}".format(tokens[i])
                        found = True
            else:
                found = False
                if entity != '':
                    entities.append(entity)
                    indices.append(indstr)
                    entity = ''
                    indstr = ''
    elif args.n_classes == 3:
        entity = ''
        indstr = ''
        for i, label in enumerate(prediction):
            if label == 0:
                if entity != '':
                    entities.append(entity)
                    indices.append(indstr)
                entity = "{}".format(i)
                indstr = "{}".format(tokens[i])
            elif label == 1:
                if entity != '':
                    entity += "_{}".format(i)
                    indstr += "_{}".format(tokens[i])
                else:
                    # print("shouldn't be {} {}".format(i, tokens[i]))
                    incon += 1
                    entity = "{}".format(i)
                    indstr = "{}".format(tokens[i])
            else:
                if entity != '':
                    entities.append(entity)
                    indices.append(indstr)
                    entity = ''
                    indstr = ''
    print("Inconsistenceis\t{}".format(incon))
    assert len(indices) == len(entities)
    return indices, entities

def get_ne_indexes(args, tags):
    '''Get named entities by indices'''
    entities = []
    if args.n_classes == 2:
        found = False
        entity = ''
        for i, label in enumerate(tags):
            if label == 0:
                if found:
                    if entity != '':
                        entity += "_{}".format(i)
                else:
                    if entity == '':
                        entity = "{}".format(i)
                        found = True
            else:
                found = False
                if entity != '':
                    entities.append(entity)
                    entity = ''
    elif args.n_classes == 3:
        entity = ''
        for i, label in enumerate(tags):
            if label == 0:
                if entity != '':
                    entities.append(entity)
                entity = "{}".format(i)
            elif label == 1:
                if entity != '':
                    entity += "_{}".format(i)
                else:
                    # print("shouldn't be {}".format(i))
                    entity = "{}".format(i)
            else:
                if entity != '':
                    entities.append(entity)
                    entity = ''
    return entities

def write_errors(tokens, true_pos, false_pos, false_neg, fname='results.txt'):
    '''Write the named entities into a file for error analysis'''
    print("TP {} FP {} FN {}".format(len(true_pos), len(false_pos), len(false_neg)))
    rfile = open(fname, 'w')
    print("TP {} FP {} FN {}".format(len(true_pos), len(false_pos), len(false_neg)), file=rfile)
    print("--TP--", file=rfile)
    for i, item in enumerate(true_pos):
        for index in item.split('_'):
            print("{}\t{}\t{}".format(i, item, tokens[int(index)]), file=rfile)
    print("--FP--", file=rfile)
    for i, item in enumerate(false_pos):
        for index in item.split('_'):
            print("{}\t{}\t{}".format(i, item, tokens[int(index)]), file=rfile)
    print("--FN--", file=rfile)
    for i, item in enumerate(false_neg):
        for index in item.split('_'):
            print("{}\t{}\t{}".format(i, item, tokens[int(index)]), file=rfile)
    rfile.close()

def phrasalf1score(args, tokens, prediction, target, write_err=False):
    '''Compute phrasal F1 score for the results'''
    gold_entities = get_ne_indexes(args, np.argmax(target, 1))
    pred_entities = get_ne_indexes(args, np.argmax(prediction, 1))
    # inefficient but easy to understand
    true_pos = [x for x in pred_entities if x in gold_entities]
    false_pos = [x for x in pred_entities if x not in gold_entities]
    false_neg = [x for x in gold_entities if x not in pred_entities]
    precision = 1.0 * len(true_pos)/(len(true_pos) + len(false_pos))
    recall = 1.0 * len(true_pos)/(len(true_pos) + len(false_neg))
    f1sc = 2.0 * precision * recall / (precision + recall)
    if write_err:
        write_errors(tokens, true_pos, false_pos, false_neg, "runs/ne_{:.5f}".format(f1sc)+".txt")
    return precision, recall, f1sc

def write_results(words, prediction, target, fname='results.txt'):
    '''Write results to file'''
    target = np.argmax(target, 1)
    prediction = np.argmax(prediction, 1)
    rfile = open(fname, 'w')
    for i, label in enumerate(target):
        print("{}\t{}\t{}".format(words[i], prediction[i], label), file=rfile)
    rfile.close()

def write_pred_and_entities(args, tokens, prediction, pmid):
    '''Write results to file'''
    prediction = np.argmax(prediction, 1)
    fname = join(args.outdir, pmid + '_pred.txt')
    rfile = open(fname, 'w')
    for i, label in enumerate(prediction):
        if args.n_classes == 2:
            label = 'I' if label == 0 else 'O'
        elif args.n_classes == 3:
            if label == 0:
                label = 'B'
            elif label == 1:
                label = 'I'
            else:
                label = 'O'
        print("{}\t{}".format(tokens[i], label), file=rfile)
    rfile.close()
    fname = join(args.outdir, pmid + '_nes.txt')
    rfile = open(fname, 'w')
    indices, entities = get_namedentities(args, tokens, prediction)
    for i, index in enumerate(indices):
        print("{}\t{}".format(index, entities[i]), file=rfile)
    rfile.close()
    print("{}\t{} entities found".format(pmid, len(entities)))

def f1score(class_size, prediction, target):
    '''Compute F1 score for the results'''
    true_pos = np.array([0] * (class_size + 1))
    false_pos = np.array([0] * (class_size + 1))
    false_neg = np.array([0] * (class_size + 1))
    target = np.argmax(target, 1)
    prediction = np.argmax(prediction, 1)
    for i, label in enumerate(target):
        if label == prediction[i]:
            true_pos[label] += 1
        else:
            false_pos[label] += 1
            false_neg[prediction[i]] += 1
    unnamed_entity = class_size - 1
    for i in range(class_size):
        if i != unnamed_entity:
            true_pos[class_size] += true_pos[i]
            false_pos[class_size] += false_pos[i]
            false_neg[class_size] += false_neg[i]
    precision = []
    recall = []
    fscore = []
    for i in range(class_size + 1):
        precision.append(true_pos[i] * 1.0 / (true_pos[i] + false_pos[i]))
        recall.append(true_pos[i] * 1.0 / (true_pos[i] + false_neg[i]))
        fscore.append(2.0 * precision[i] * recall[i] / (precision[i] + recall[i]))
    return precision[class_size], recall[class_size], fscore[class_size]

def get_input_distsup(word_emb_model, input_file, window=5):
    '''loads distance supervision dataset'''
    center = int(window/2)
    # print("processing file: {} and neighbors = {}".format(input_file, center))
    words = []
    tags = []
    instances = []
    for line in open(input_file):
        assert len(line.split('\t')) == window
        tag = line.split('\t')[center].split()[1]
        tags.append(np.array([1, 0]) if tag == 'I' else np.array([0, 1]))
        word = ' '.join([x.split()[0] for x in line.split('\t')])
        words.append(word)
        context = []
        for token in word.split():
            context = np.append(context, word_emb_model[token])
        instances.append(context)
        # instances.append([word_emb_model[x] for x in word.split()])
    assert len(instances) == len(tags) == len(words)
    return words, instances, tags

def get_input(args, word_emb_model, input_file):
    '''loads input dataset'''
    n_neighbors = int(args.window_size/2)
    print("processing file: {} and neighbors = {}".format(input_file, n_neighbors))
    padding = "<s>"
    words = []
    labels = []
    for _ in range(n_neighbors):
        words.append(padding)
    for line in open(input_file):
        assert len(line.split()) == 2
        word = line.split()[0]
        label = line.split()[1]
        words.append(word)
        if args.n_classes == 3:
            if label == 'B':
                labels.append(np.array([1, 0, 0]))
            elif label == 'I':
                labels.append(np.array([0, 1, 0]))
            elif label == 'O':
                labels.append(np.array([0, 0, 1]))
            else:
                print("Invalid tag {} found for word {}".format(label, word))
        elif args.n_classes == 2:
            if label == 'I':
                labels.append(np.array([1, 0]))
            elif label == 'O':
                labels.append(np.array([0, 1]))
            else:
                print("Invalid tag {} found for word {}".format(label, word))
        else:
            print("Invalid number of classes {}. Only 2,3 allowed.".format(args.n_classes))
    for _ in range(n_neighbors):
        words.append(padding)
    instances = []
    for i in range(n_neighbors, len(words)-n_neighbors):
        context = []
        for j in range(-n_neighbors, n_neighbors+1):
            context = np.append(context, word_emb_model[words[i+j]])
            context = np.append(context, get_feature(args, words[i+j]))
        instances.append(context)
    words = words[n_neighbors:len(words)-n_neighbors]
    assert len(words) == len(instances) == len(labels)
    return words, instances, labels

def get_input_pub(args, word_emb_model, input_file):
    '''loads input pub files for annotation'''
    n_neighbors = int(args.window_size/2)
    # print("processing file: {} and neighbors = {}".format(input_file, n_neighbors))
    padding = "<s>"
    words = []
    for _ in range(n_neighbors):
        words.append(padding)
    for line in open(input_file):
        word = line.strip()
        words.append(word)
    for _ in range(n_neighbors):
        words.append(padding)
    instances = []
    for i in range(n_neighbors, len(words)-n_neighbors):
        context = []
        for j in range(-n_neighbors, n_neighbors+1):
            context = np.append(context, word_emb_model[words[i+j]])
            context = np.append(context, get_feature(args, words[i+j]))
        instances.append(context)
    words = words[n_neighbors:len(words)-n_neighbors]
    assert len(words) == len(instances)
    return words, instances
