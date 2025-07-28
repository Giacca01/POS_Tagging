# implements several smoothing strategies
# to account for words unseen during training
import shared_data as sd
import numpy as np
import training as tr
from collections import defaultdict

K = 5

# probability distribution of tags
# for the unknown words, represented
# as the UNK tag
unknown_dist = np.zeros(len(sd.tag_set), dtype=float)

# all these functions estimate P(unk|tag)
def nouns_smoothing():
    for tag in sd.tag_set:
        tag_val = sd.tag_set[tag]
        if tag_val == sd.tag_set["NOUN"]:
            tr.word_tag_distribution["UNK"][tag_val] = np.log(0.8)
        else:
            tr.word_tag_distribution["UNK"][tag_val] = np.log(0.2/(len(sd.tag_set) - 3))

def nouns_verb_smoothing():
    for tag in sd.tag_set:
        tag_val = sd.tag_set[tag]
        if tag_val == sd.tag_set["NOUN"] or tag_val == sd.tag_set["VERB"]:
            tr.word_tag_distribution["UNK"][tag_val] = np.log(0.4)
        else:
            tr.word_tag_distribution["UNK"][tag_val] = np.log(0.2/(len(sd.tag_set) - 4))

def uniform_smoothing():
    for tag in sd.tag_set:
        tag_val = sd.tag_set[tag]
        tr.word_tag_distribution["UNK"][tag_val] = 1 / (tr.tag_count[tag_val] + K)

def single_word_smoothing():
    word_tag_count = dict()
    word_count = defaultdict(int)
    single_occ_word_count = 0

    file = open("processed_files/" + sd.DEV_SET + ".txt", "r")
    for line in file:
        line = line.strip("\n")
        if line != "START" and line != "END":
            splitted_line = line.split("\t")
            curr_tag = sd.tag_set[splitted_line[1]]
            curr_word = splitted_line[0]
            word_count[curr_word] = word_count[curr_word] + 1
            # discard words with more than one occurrence
            if word_count[curr_word] == 2:
                word_tag_count.pop(curr_word)
                single_occ_word_count = single_occ_word_count - 1
            elif word_count[curr_word] == 1:
                word_tag_count[curr_word] = curr_tag
                single_occ_word_count = single_occ_word_count + 1

    file.close()

    # compute total occurrences for tags of
    # words that occur only once
    for word in word_tag_count:
        tag = word_tag_count[word]
        tr.word_tag_distribution["UNK"][tag] = tr.word_tag_distribution["UNK"][tag] + 1

    pair_count = 0
    single_count = 0
    # estimate emission probabilities
    for tag in sd.tag_set:
        tag_val = sd.tag_set[tag]
        pair_count = np.log(tr.word_tag_distribution["UNK"][tag_val])
        single_count = np.log(single_occ_word_count)
        tr.word_tag_distribution["UNK"][tag_val] = pair_count / single_count