# Custom Named Entity (Disease) Recognition in clinical text with spaCy in Python https://www.youtube.com/watch?v=DxLcMI-EMYI
# Named Entity Recognition using Spacy and Tensorflow https://aihub.cloud.google.com/p/products%2F2290fc65-0041-4c87-a898-0289f59aa8ba
# SpaCy Training using GPU https://jerilkuriakose.medium.com/spacy-training-using-gpu-e6ea71916007

import random
import time
from itertools import chain
from os import path, mkdir

import matplotlib.pyplot as plt
import numpy as np
import spacy
import thinc_gpu_ops
from matplotlib.ticker import MaxNLocator
from spacy import displacy
from spacy.util import minibatch, compounding

if not path.isdir('data/'):
    mkdir('data/')
if not path.isdir('models/'):
    mkdir('models/')


def load_data_spacy(file_path):
    """ Converts data from:
    label \t word \n label \t word \n \n label \t word
    to: sentence, {entities : [(start, end, label), (start, end, label)]}
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        file = f.readlines()

    training_data, entities, sentence, unique_labels = [], [], [], []
    start, end = 0, 0  # initialize counter to keep track of start and end characters

    for line in file:
        line = line.strip('\n').split('\t')
        # lines with len > 1 are words
        if len(line) > 1:
            label = line[1]
            label_type = label[0]  # beginning of annotations - "B-xxx", intermediate - "I-xxx"

            word = line[0]
            sentence.append(word)
            start = end
            end += (len(word) + 1)  # length of the word + trailing space

            if label_type == 'I':  # if at the end of an annotation
                entities.append((start, end - 1, label))  # append the annotation
            if label_type == 'B':  # if beginning new annotation
                entities.append((start, end - 1, label))  # start annotation at beginning of word

            if label != 'O' and label not in unique_labels:
                unique_labels.append(label)

        # lines with len == 1 are breaks between sentences
        if len(line) == 1:
            if len(entities) > 0:
                sentence = ' '.join(sentence)
                training_data.append([sentence, {'entities': entities}])
            # reset the counters and temporary lists
            start, end = 0, 0
            entities, sentence = [], []

    return training_data, unique_labels


def calc_precision(pred, true):
    precision = len([x for x in pred if x in true]) / (len(pred) + 1e-20)  # true positives / total pred
    return precision


def calc_recall(pred, true):
    recall = len([x for x in true if x in pred]) / (len(true) + 1e-20)  # true positives / total test
    return recall


def calc_f1(precision, recall):
    f1 = 2 * ((precision * recall) / (precision + recall + 1e-20))
    return f1


# run the predictions on each sentence in the test dataset, and return the spacy object
def evaluate(ner, data):
    preds = [ner(x[0]) for x in data]

    precisions, recalls, f1s = [], [], []

    # iterate over predictions and test data and calculate precision, recall, and F1-score
    for pred, true in zip(preds, data):
        true = [x[2] for x in list(chain.from_iterable(true[1].values()))]  # x[2] = annotation, true[1] = (start, end, annot)
        pred = [i.label_ for i in pred.ents]  # i.label_ = annotation label, pred.ents = list of annotations

        precision = calc_precision(true, pred)
        precisions.append(precision)
        recall = calc_recall(true, pred)
        recalls.append(recall)
        f1s.append(calc_f1(precision, recall))

    return {
        'textcat_p': np.mean(precisions),
        'textcat_r': np.mean(recalls),
        'textcat_f': np.mean(f1s)
    }


# A simple decorator to log function processing time
def timer(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print("Completed in {} seconds".format(int(te - ts)))
        return result

    return timed


# Data must be of the form (sentence, {entities: [start, end, label]})
@timer
def train_spacy(train_data, labels, iterations, dropout=0.2, display_freq=1):
    """ Train a spacy NER model, which can be queried against with test data

    train_data : training data in the format of (sentence, {entities: [(start, end, label)]})
    labels : a list of unique annotations
    iterations : number of training iterations
    dropout : dropout proportion for training
    display_freq : number of epochs between logging losses to console
    """
    valid_f1scores, test_f1scores = [], []

    if thinc_gpu_ops.AVAILABLE:
        spacy.require_gpu()
        print('Using GPU for model training')
    else:
        spacy.prefer_gpu()
        print('Using CPU for model training')

    nlp = spacy.load('en_core_web_sm')
    # nlp = spacy.load('en')
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    else:
        ner = nlp.get_pipe('ner')

    # Add entity labels to the NER pipeline
    for i in labels:
        ner.add_label(i)

    # Disable other pipelines in SpaCy to only train NER
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        # nlp.vocab.vectors.name = 'spacy_model'  # without this, spaCy throws an "unnamed" error
        optimizer = nlp.begin_training()
        for itr in range(iterations):
            random.shuffle(train_data)  # shuffle the training data before each iteration
            losses = {}
            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,
                    annotations,
                    drop=dropout,
                    sgd=optimizer,
                    losses=losses)
            # if itr % display_freq == 0:
            #     print("Iteration {} Loss: {}".format(itr + 1, losses))

            print('\n========================================')
            print(f'Interaction = {str(itr + 1)}')
            print(f'Losses = {str(losses)}')

            scores = evaluate(nlp, VALID_DATA)
            valid_f1scores.append(scores["textcat_f"])
            print('========= VALID DATA ====================')
            print(f'Precision = {str(scores["textcat_p"])}')
            print(f'Recall = {str(scores["textcat_r"])}')
            print(f'F1-score = {str(scores["textcat_f"])}')

            scores = evaluate(nlp, TEST_DATA)
            test_f1scores.append(scores["textcat_f"])
            print('========= TEST DATA =====================')
            print(f'Precision = {str(scores["textcat_p"])}')
            print(f'Recall = {str(scores["textcat_r"])}')
            print(f'F1-score = {str(scores["textcat_f"])}')

    return nlp, valid_f1scores, test_f1scores


def load_model(model_path):
    """ Loads a pre-trained model for prediction on new test sentences

    model_path : directory of model saved by spacy.to_disk
    """
    nlp = spacy.blank('en')
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)

    ner = nlp.from_disk(model_path)
    return ner


if __name__ == '__main__':
    TRAIN_DATA, LABELS = load_data_spacy('data/train.tsv')  # 60% of ../description/all-description.txt
    VALID_DATA, _ = load_data_spacy('data/devel.tsv')  # 20% of ../description/all-description.txt
    TEST_DATA, _ = load_data_spacy('data/test.tsv')  # 20% of ../description/all-description.txt

    # Train (and save) the NER model
    ner, valid_f1scores, test_f1scores = train_spacy(TRAIN_DATA, LABELS, 3)
    ner.to_disk('models/spacy_example')

    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(valid_f1scores, label='Validation F1-score')
    ax.plot(test_f1scores, label='Test F1-score')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('F1-score')
    ax.legend()
    ax.set_title('F1-score vs iterations for validation and test data')
    plt.show()

    # Let's test our model on test data
    ner = load_model('models/spacy_example')

    test_sentences = [x[0] for x in TEST_DATA[:300]]  # extract the sentences from [sentence, entity]
    for test_sentence in test_sentences:
        doc = ner(test_sentence)
        # for ent in doc.ents:
        #     print(ent.text, ent.start_char, ent.end_char, ent.label_)
        displacy.render(doc, jupyter=True, style='ent')
