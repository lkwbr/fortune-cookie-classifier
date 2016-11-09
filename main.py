# Luke Weber, 11398889
# CptS 570, HW #4
# Created 11/04/2016

# Problem 8

"""
This classifier will be used to classify fortune cookie messages into two
classes:
    a. Messages that predict what will happen in the future (class 1)
    b. Messages that just contain a wise saying (class 0)

(1) Preprocessing step:
    Convert messages into features, using "bag of words" representation
        - Form the vocabulary (i.e. all words in training data with stop
            words removed), keeping alphabetical order.
        - Construct features: for vocabulary size M, each message will have
            feature vector of size M; each value takes 0 or 1, representing
            if corresponding word (alphabetical order) for ith slot is
            present or not.
    
(2) Classification step:
    Implement Naive Bayes classifier with laplace smoothing and run it on training
    data and compute training and testing accuracy.
"""

import math
import operator
import time

class NaiveBayesModel:
    """
    Naive Bayes probability model, predicting class of given feature
    vector based on what we've seen in the past
    """

    # Model data as lists with indices corresponding to same feature
    # e.g. [[x_1,...,x_m], ...] [y, ...]
    seen_features = []
    seen_labels = []
    seen_messages = []
    vocab = []

    def get_label_prob(self, feature, check_label):
        """
        Doing P(Class=label|x_1,...,x_m) =
        P(Class=label) * Product{i: 1->m}P(x=x_i|Class=label)
        """
        
        # Compute P(Class=label) = len(Class=label) / len(Classes)
        num_of_label = 0
        num_total_labels = len(self.seen_labels)
        for past_label in self.seen_labels:
            if past_label == check_label: num_of_label += 1

        # Laplace smoothing on p_class as well
        p_label = (num_of_label + 1) / (num_total_labels + 1)

        print("\t\tP(Class=" + str(check_label) + ")=" +
              str(p_label) + " (" + str(num_of_label) + "/" +
              str(num_total_labels) + ")")
        
        count_w_and_c = [0 for x in range(len(self.vocab))]
        # Go through past feature vectors of same class and
        # count up occurences of x_i = feature_i
        for i in range(len(self.seen_features)):
            f = self.seen_features[i]
            l = self.seen_labels[i]
            if l == check_label:
                # Go through words in each vector
                for j in range(len(f)):
                    if f[j] == feature[j]: count_w_and_c[j] += 1

        # Multiply all prob_w_c together:
        # Compute Product{i: 1->m}P(x=x_i|Class=label) where
        # P(x=x_i|Class=label) = count(w,c) / count(c)
        p_product = 1
        i = 0
        for c in count_w_and_c:
            # NOTES:
            #   - Was originally /100
            #   - Going to /200 helped
            #   - It seems that our accuracy depends on adjusting the
            #       various parameters, because the implementation
            #       seems correct
            #   - Going to /600 helped again
            # Laplace smoothing for binomial case
            prob_w_c = (c + 1) / (num_of_label + 1)
                                  #(len(self.vocab)/1000) + 1)            
            p_product *= prob_w_c
    
        return p_label * p_product

    def predict(self, feature, label, is_testing=True):
        """
        Predict class as 0 or 1, and build up model as we go if flag
        is_testing is true
        """
        
        # Determine P(Class=0 | x_1,...,x_m) and P(Class=1 | x_1,...,x_m)
        label_0_prob = self.get_label_prob(feature, 0)
        label_1_prob = self.get_label_prob(feature, 1)

        print("\t\tlabel_0_prob = " + str(label_0_prob) +
              "\n\t\tlabel_1_prob = " + str(label_1_prob))

        # Pick label with max probability
        pred_label = 0
        if (label_1_prob > label_0_prob): pred_label = 1

        print("\t\tActual label = " + str(label) + ", predicted label = " +
              str(pred_label))

        if is_testing is True:
            # Add this example to model's seen data, used for future
            # predictions
            self.seen_features.append(feature)
            self.seen_labels.append(label)

        print()
        
        return pred_label

    def __init__(self, vocab):
        """
        Default constructor: sets size of vocabulary, which we use for
        Laplace smoothing and such
        """
        self.vocab = vocab

def preprocess(data_loc, label_loc, extern_vocab=None):
    """
    Converts messages into features and returns list of these messages,
    vocabulary, feature vectors, and training labels
    """

    print("Preprocesing data...")

    # Location of stop words
    stop_list_loc = "data/stoplist.txt"

    # Get data into memory as list of lines
    with open(data_loc, 'r') as file:
        data_raw = file.read().splitlines()
    with open(label_loc, 'r') as file:
        labels_raw = file.read().splitlines()
    with open(stop_list_loc, 'r') as file:
        stop_list_raw = file.read().splitlines()

    # Parse stop words into unique set
    stop_words = set()
    for line in stop_list_raw:
        word = line
        stop_words.add(word)

    # If vocabulary provided, use that; otherwise,
    # get train (message) words into unique set,
    # ensuring not to add stop words, and convert
    # then to sorted list
    vocab = set()
    if extern_vocab is None:
        for line in data_raw:
            words = line.split()
            for word in words:
                if word not in stop_words:
                    vocab.add(word)
        sorted_vocab = sorted(vocab)
    else: vocab = extern_vocab

    # Turn each class label to int (e.g. '0' to 0, '1' to 1)
    labels = list(map(int, labels_raw))

    # Mark which vocab words appeared in each message
    feature_sets = []
    for message in data_raw:
        message_words = dict.fromkeys(vocab, 0)
        words = message.split()
        for word in words:
            if word in message_words:
                message_words[word] = 1
        feature_sets.append(message_words)

    # Turn feature sets into feature vectors (sorted alphebetically)
    features = []
    for feature in feature_sets:
        sorted_set = sorted(feature.items(), key=operator.itemgetter(0))
        sorted_set_occ = [x[1] for x in sorted_set]
        features.append(sorted_set_occ)

    # Return different things, depending on if we've been given
    # an external vocabulary
    if extern_vocab is None:
        return data_raw, sorted_vocab, features, labels
    else: return data_raw, features, labels
        
def train(messages, vocab, features, labels):
    """
    Train Naive Bayes classifier (with Laplace smoothing) with feature-label
    pairs, computing training accuracy; source of info:
    https://en.wikipedia.org/wiki/Naive_Bayes_classifier
    """
    
    print("Training Naive Bayes classifier on data " +
          "(may take a minute)...")

    # Initialize our probability model with vocabulary
    model = NaiveBayesModel(vocab)

    # Collect accuracy
    num_mistakes = 0
    num_features = len(features)

    # Train on each feature, sequentially
    for i in range(len(messages)):
        message = messages[i]
        feature = features[i]
        label = labels[i]
        print("\t(" + str(i + 1) + ") '" + str(message) + "'")
        pred_label = model.predict(feature, label)
        if (pred_label != label): num_mistakes += 1

    # Print stats
    train_accuracy = (1 - (num_mistakes / num_features)) * 100
    print("\n" + "Completed with accuracy = %0.2f%%" % train_accuracy)

    return model

def test(model, features, labels):
    """
    Determine testing accuracy of Naive Bayes classifier (model)
    """

    print("Testing model accuracy...")

    # Collect stats
    num_mistakes = 0
    num_features = len(features)

    # Train on each feature, sequentially
    for i in range(num_features):
        feature = features[i]
        label = labels[i]
        pred_label = model.predict(feature, label, True)
        if (pred_label != label):
            num_mistakes += 1

    test_accuracy = (1 - (num_mistakes / num_features)) * 100
    print("\n" + "Our classifier works with accuracy = %0.2f%%" % test_accuracy)

def main():
    """
    Driver function to do the following:
        1. Parse train data
        2. Train and build model on data
        3. Parse test data (using model/training vocabulary)
        4. Test model with data
        
    All while watching the time it takes to complete
    """
    # Train
    time_start = time.clock()
    train_messages, vocab, train_features, train_labels = preprocess(
        "data/traindata.txt", "data/trainlabels.txt")
    model = train(train_messages, vocab, train_features, train_labels)
    time_end = time.clock()
    print("Took %.2f minutes to train" % ((time_end - time_start)/60))
    # Test
    time_start = time.clock()
    test_messages, test_features, test_labels = preprocess(
        "data/testdata.txt", "data/testlabels.txt", vocab)
    test(model, test_features, test_labels)
    time_end = time.clock()
    print("Took %.2f minutes to test" % ((time_end - time_start)/60))

# Start
main()

