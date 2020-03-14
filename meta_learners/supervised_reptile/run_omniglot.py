"""
Train a model on Omniglot.
"""

import random

import tensorflow as tf

from meta_learners.args import argument_parser, model_kwargs, train_kwargs, evaluate_kwargs
from meta_learners.supervised_reptile.supervised_reptile.eval import evaluate
from meta_learners.supervised_reptile.supervised_reptile.models import OmniglotModel
from meta_learners.supervised_reptile.supervised_reptile.omniglot import read_dataset, split_dataset, augment_dataset
from meta_learners.supervised_reptile.supervised_reptile.train import train

DATA_DIR = 'data/omniglot'

def main():
    """
    Load data and train a model on it.
    """
    args = argument_parser().parse_args()
    random.seed(args.seed)

    # Get the meta-learning dataset. Each item in train_set and test_set is a task:
    train_set, test_set = split_dataset(read_dataset(DATA_DIR))
    train_set = list(augment_dataset(train_set))
    test_set = list(test_set)

    model = OmniglotModel(args.classes, **model_kwargs(args))

    with tf.Session() as sess:
        if not args.pretrained:
            print('Training...')
            train(sess, model, train_set, test_set, args.checkpoint, **train_kwargs(args))
        else:
            print('Restoring from checkpoint...')
            tf.train.Saver().restore(sess, tf.train.latest_checkpoint(args.checkpoint))

        print('Evaluating...')
        eval_kwargs = evaluate_kwargs(args)
        print('Train accuracy: ' + str(evaluate(sess, model, train_set, **eval_kwargs)))
        print('Test accuracy: ' + str(evaluate(sess, model, test_set, **eval_kwargs)))

if __name__ == '__main__':
    main()
