# implements learning as maximum likelihood estimation
# of parameters via counting on corpus
from collections import defaultdict
import shared_data as sd
import numpy as np


# defaultdicts are used to implement a sparse data representation
# we actually store on elements (e.g. single tag, tag pairs or word-tag pairs)
# that only occur in our corpus.
# When retrieving the counts for a tag that has never been seen in training
# the default value for the associated datatype (0 or another default dict)
# is returned
# In general, dictionaty is really the natural data structure to store
# the key-value pairs that we need, and also provides good performance
tag_count = defaultdict(int)

# membership can be checked in O(1) and no duplicates allowed
vocabulary = set()

# store co-occurrence counts for pairs of tags
tag_pair_count = defaultdict(lambda: defaultdict(int))

# first level uses words as keys
# second level uses tags as keys
word_tag_count = defaultdict(lambda: defaultdict(int))

# stores our approximation of the transition
# probability distribution
# float uses 64 bits, providing 15 digits precision
tag_tag_distribution = defaultdict(lambda: defaultdict(float))

# stores our approximation of the emission probability distribution
word_tag_distribution = defaultdict(lambda: defaultdict(float))


def train(training_set):
    file = open("processed_files/" + training_set + ".txt", "r")
    
    # counting
    for line in file:
        line = line.strip("\n")
        if line != "END":
            if line == "START":
                prev_tag = sd.tag_set['START']
                curr_tag = prev_tag
                tag_count[prev_tag] = tag_count[prev_tag] + 1
            else:
                splitted_line = line.split("\t")

                curr_word = splitted_line[0]
                vocabulary.add(curr_word)

                prev_tag = curr_tag
                curr_tag = sd.tag_set[splitted_line[1]]

                tag_pair_count[prev_tag][curr_tag] = tag_pair_count[prev_tag][curr_tag] + 1
                word_tag_count[curr_word][curr_tag] = word_tag_count[curr_word][curr_tag] + 1
        else:
            prev_tag = curr_tag
            curr_tag = sd.tag_set['END']
            tag_pair_count[prev_tag][curr_tag] = tag_pair_count[prev_tag][curr_tag] + 1

        tag_count[curr_tag] = tag_count[curr_tag] + 1

    file.close()

    pair_count = 0.0
    single_count = 0.0
    # computing transitions probabilities
    for tag_prev in sd.tag_set:
        tag_prev_val = sd.tag_set[tag_prev]
        # END will be the final state of the chain
        # so we cannot exit it
        if tag_prev_val != sd.tag_set['END']:
            total = sum(tag_tag_distribution[tag_prev_val].values()) + len(sd.tag_set)
            for tag_curr in sd.tag_set:
                tag_curr_val = sd.tag_set[tag_curr]
                # START will be the first state of the chain
                # so we cannot enter it
                if tag_curr_val != sd.tag_set['START']:
                    pair_count = np.log(tag_pair_count[tag_prev_val][tag_curr_val] + 1)
                    single_count = np.log(tag_count[tag_prev_val] + total)
                    tag_tag_distribution[tag_prev_val][tag_curr_val] =  pair_count - single_count

    # computing emission probabilities
    pair_count = 0.0
    single_count = 0.0
    for word in vocabulary:
        for tag in sd.tag_set:
            tag_val = sd.tag_set[tag]
            total = tag_count[tag_val] + len(vocabulary)
            if tag_val != sd.tag_set['START'] and tag_val != sd.tag_set['END']:
                pair_count = np.log(word_tag_count[word][tag_val] + 1)
                single_count = np.log(tag_count[tag_val] + total)
                word_tag_distribution[word][tag_val] = pair_count - single_count