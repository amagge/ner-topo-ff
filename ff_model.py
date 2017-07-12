'''
A Deep Neural network with two layers for independent classification
'''

from __future__ import print_function

import argparse
from os import listdir
from os.path import isfile, join

import numpy as np
import tensorflow as tf

from utils import WordEmb, get_input, get_input_distsup, get_input_pub
from utils import f1score, write_results, write_pred_and_entities, phrasalf1score

class FFModel(object):
    '''Feed-Forward model'''
    def __init__(self, n_input, n_classes, hid_dim, lrn_rate):
        # placeholders
        self.input_x = tf.placeholder(tf.float32, [None, n_input])
        self.output_y = tf.placeholder(tf.float32, [None, n_classes])
        self.dropout = tf.placeholder(tf.float32)
        # weights and biases
        weights_w1 = tf.Variable(tf.random_normal([n_input, hid_dim]))
        weights_w2 = tf.Variable(tf.random_normal([hid_dim, hid_dim]))
        weights_out = tf.Variable(tf.random_normal([hid_dim, n_classes]))
        biases_b1 = tf.Variable(tf.random_normal([hid_dim])),
        biases_b2 = tf.Variable(tf.random_normal([hid_dim])),
        biases_b3 = tf.Variable(tf.random_normal([n_classes]))
        # operations for predictions
        # self.input_x = tf.nn.dropout(self.input_x, self.dropout)
        layer_1 = tf.add(tf.matmul(self.input_x, weights_w1), biases_b1)
        layer_1 = tf.nn.relu(layer_1)
        # layer_1 = tf.nn.dropout(layer_1, self.dropout)
        layer_2 = tf.add(tf.matmul(layer_1, weights_w2), biases_b2)
        layer_2 = tf.nn.relu(layer_2)
        layer_2 = tf.nn.dropout(layer_2, self.dropout)
        self.pred = tf.matmul(layer_2, weights_out) + biases_b3
        # determine cost and optimize all variables
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                                   (logits=self.pred, labels=self.output_y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lrn_rate).minimize(self.cost)

def train(args):
    '''Training method'''
    # Load Word Embeddings
    word_emb = WordEmb(args)
    # Load training, test and validation tokens, vector instances (input vector) and labels
    test_t, test_v, test_l = get_input(args, word_emb, args.test)
    train_t, train_v, train_l = get_input(args, word_emb, args.train)
    valid_t, valid_v, valid_l = get_input(args, word_emb, args.val)
    n_input = len(test_v[0])
    print("Input size detected ", n_input)
    # Create model
    model = FFModel(n_input, args.n_classes, args.hid_dim, args.lrn_rate)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        def evaluate(tokens, instances, labels, write_result=False):
            '''Evaluate and print results'''
            prediction, target = sess.run([model.pred, model.output_y],
                                          feed_dict={model.input_x: np.asarray(instances),
                                                     model.output_y: np.asarray(labels),
                                                     model.dropout: 1.0})
            # prec, recall, f1sc = f1score(args.n_classes, prediction, target)
            prec, recall, f1sc = phrasalf1score(args, tokens, prediction, target)
            if write_result:
                prec, recall, f1sc = f1score(args.n_classes, prediction, target)
                print("Found MAX\n--Tokenwise P:{:.5f}".format(prec), "R:{:.5f}".format(recall),
                      "F1:{:.5f}".format(f1sc))
                prec, recall, f1sc = phrasalf1score(args, tokens, prediction, target)
                print("--Phrasal P:{:.5f}".format(prec), "R:{:.5f}".format(recall),
                      "F1:{:.5f}".format(f1sc))
                write_results(tokens, prediction, target, "runs/res_{:.5f}".format(f1sc)+".txt")
            return f1sc
        # Distance Supervision cycle
        for epoch in range(args.dist_epochs):
            dist_dir = args.dist + str(args.window_size)
            dist_files = [f for f in listdir(dist_dir) if isfile(join(dist_dir, f))]
            for _, distfile in enumerate(dist_files):
                dist_t, dist_v, dist_l = get_input_distsup(word_emb, join(dist_dir, distfile))
                dist_cost = 0.
                total_batch = int(len(dist_v)/args.batch_size)
                for ptr in range(0, len(dist_v), args.batch_size):
                    # Run backprop and cost during training
                    _, part_cost = sess.run([model.optimizer, model.cost], feed_dict={
                        model.input_x: np.asarray(dist_v[ptr:ptr + args.batch_size]),
                        model.output_y: np.asarray(dist_l[ptr:ptr + args.batch_size]),
                        model.dropout: 0.5})
                    # Compute average loss across batches
                    dist_cost += part_cost / total_batch
                print("Epoch:", '%02d' % (epoch+1), "File:", distfile,
                      "cost=", "{:.5f}".format(dist_cost))
            dist_f1 = evaluate(dist_t, dist_v, dist_l)
            train_f1 = evaluate(train_t, train_v, train_l)
            print("-Distance : {:.5f}".format(dist_f1), "Training : {:.5f}".format(train_f1))
        # Training cycle
        if args.train_epochs > 0:
            maxf1 = 0.0
            for epoch in range(args.train_epochs):
                avg_cost = 0.
                total_batch = int(len(train_v)/args.batch_size)
                for ptr in range(0, len(train_v), args.batch_size):
                    # Run backprop and cost during training
                    _, epoch_cost = sess.run([model.optimizer, model.cost], feed_dict={
                        model.input_x: np.asarray(train_v[ptr:ptr + args.batch_size]),
                        model.output_y: np.asarray(train_l[ptr:ptr + args.batch_size]),
                        model.dropout: args.dropout})
                    # Compute average loss across batches
                    avg_cost += epoch_cost / total_batch
                print("Epoch:", '%02d' % (epoch+1), "cost=", "{:.5f}".format(avg_cost))
                if epoch % args.eval_interval == 0:
                    train_f1 = evaluate(train_t, train_v, train_l)
                    val_f1 = evaluate(valid_t, valid_v, valid_l)
                    print("-Training : {:.5f}".format(train_f1), "Val : {:.5f}".format(val_f1))
                    if val_f1 > maxf1:
                        maxf1 = val_f1
                        evaluate(test_t, test_v, test_l, True)
                        if args.save is not None:
                            # Write model checkpoint to disk
                            # saver.restore(sess, args.save)
                            print("Saving model to {}".format(args.save))
                            saver.save(sess, args.save)
            print("Optimization Finished!")
        # Load best model and evaluate model on the test set before applying to production
        if args.restore is not None:
            saver = tf.train.import_meta_graph(args.restore + '.meta')
            saver.restore(sess, args.restore)
            print("Model from {} restored.".format(args.restore))
        evaluate(test_t, test_v, test_l, True)
        # load the pubmed files for annotation pubdir
        pub_files = [f for f in listdir(args.pubdir) if isfile(join(args.pubdir, f))]
        for _, pubfile in enumerate(pub_files):
            pub_t, pub_v = get_input_pub(args, word_emb, join(args.pubdir, pubfile))
            prediction = sess.run(model.pred, feed_dict={model.input_x: np.asarray(pub_v),
                                                         model.dropout: 1.0})
            write_pred_and_entities(args, pub_t, prediction, pubfile.replace(".txt", ""))


def main():
    '''Main method : parse input arguments and train'''
    parser = argparse.ArgumentParser()
    # Input files
    # parser.add_argument('--train', type=str, default='data/io/train-io.txt',
    #                     help='train file location')
    # parser.add_argument('--test', type=str, default='data/io/test-io.txt',
    #                     help='test file location')
    # parser.add_argument('--val', type=str, default='data/io/val-io.txt',
    #                     help='val file location')
    parser.add_argument('--train', type=str, default='data/bio/train-bio.txt',
                        help='train file location')
    parser.add_argument('--test', type=str, default='data/bio/test-bio.txt',
                        help='test file location')
    parser.add_argument('--val', type=str, default='data/bio/val-bio.txt',
                        help='val file location')
    parser.add_argument('--dist', type=str, default='data/dist/',
                        help='distance supervision files dir.')
    parser.add_argument('--pubdir', type=str, default='data/pubmed/',
                        help='pubmed files dir. To be production set. ')
    parser.add_argument('--outdir', type=str, default='out/pubmed/',
                        help='Output dir for ffmodel annotated pubmed files.')
    # Word Embeddings
    parser.add_argument('--emb_loc', type=str, default="data/PMC-w2v.bin",
                        help='word2vec embedding location')
    parser.add_argument('--embvocab', type=int, default=-1, help='load top n words in word emb')
    # Hyperparameters
    parser.add_argument('--hid_dim', type=int, default=200, help='dimension of hidden layers')
    parser.add_argument('--lrn_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--feat_cap', type=str, default=None, help='Capitalization feature')
    parser.add_argument('--feat_dict', type=str, default=None, help='Dictionary feature')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability')
    # Settings
    parser.add_argument('--window_size', type=int, default=5, help='context window size - 3/5/7')
    parser.add_argument('--dist_epochs', type=int, default=2, help='number of distsup epochs')
    parser.add_argument('--train_epochs', type=int, default=50, help='number of train epochs')
    parser.add_argument('--eval_interval', type=int, default=1, help='evaluate once in _ epochs')
    parser.add_argument('--n_classes', type=int, default=2, choices=range(2, 4),
                        help='number of classes')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of training')
    # Model save and restore paths
    parser.add_argument('--restore', type=str, default="model/ffm", help="path of saved model")
    parser.add_argument('--save', type=str, default="model/ffm", help="path to save model")
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
