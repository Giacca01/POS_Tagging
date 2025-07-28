import numpy as np
import training as tr
import shared_data as sd
import sys

# viterbi algorithm implementation
def viterbi(observations):
    # START and END are in the tag set
    # so no need to increase this by 2
    N = len(sd.tag_set)
    T = len(observations)
    # rows store states, columns store observation
    # column 0 is for start
    viterbi_mat = np.zeros((N, T+1))
    backpointer = np.zeros((N, T+1))
    path = np.zeros(T)
    START = sd.tag_set['START']
    END = sd.tag_set['END']


    # salto START ed END
    word = observations[0]
    if (word not in tr.vocabulary):
        word = "UNK"
    # computing probabilities of being in state I
    # after the first (index 0) observation
    for i in range(1, len(sd.tag_set) - 1):
        viterbi_mat[i][0] = tr.tag_tag_distribution[START][i] + tr.word_tag_distribution[word][i]
        backpointer[i][0] = START
    
    # from seecond observation onwards
    for t in range(1, len(observations)):
        # excludes START because we can't get back to it
        # excludes END because it will be treated separately
        for s in range(1, len(sd.tag_set) - 1):
            word = observations[t]
            max_index = find_max_index(s, t-1, viterbi_mat)
            if (word not in tr.vocabulary):
                word = "UNK"
            
            viterbi_mat[s][t] = viterbi_mat[max_index][t-1] + tr.tag_tag_distribution[max_index][s]+ tr.word_tag_distribution[word][s]
            backpointer[s][t] = max_index
    
    # computes probability of being in state END
    # after all the obserations
    max_index = find_max_index(END, T-1, viterbi_mat)
    viterbi_mat[END][T] = viterbi_mat[max_index][T - 1] + tr.tag_tag_distribution[max_index][END]
    backpointer[END][T] = max_index


    # loops path backwards
    # to compute tags sequence
    curr_obs = T
    curr_state = backpointer[END][curr_obs]
    while curr_state != START:
        path[curr_obs - 1] = curr_state
        curr_obs = curr_obs - 1
        curr_state = backpointer[(int)(curr_state)][curr_obs]
    
    return path

def find_max_index(s, prev_t, viterbi_mat):
    max_prob = -sys.float_info.max
    max_index = -1

    for tag_prev in sd.tag_set:
        tag_prev_val = sd.tag_set[tag_prev]    
        if (tag_prev_val != sd.tag_set["START"]): 
            if (tag_prev_val != sd.tag_set["END"]):
                prob = viterbi_mat[tag_prev_val][prev_t] + tr.tag_tag_distribution[tag_prev_val][s]
                if prob > max_prob:
                    max_prob = prob
                    max_index = tag_prev_val

    return max_index

# implements majority tagging
def majority_tagging(observations):
    T = len(observations)
    result = np.zeros(T)

    for i in range(T):
        if (observations[i] in tr.word_tag_distribution):
            word = observations[i]
        else:
            word = "UNK"
    
        max_prob = -sys.float_info.max
        for tag in tr.word_tag_distribution[word]:
            if (tr.word_tag_distribution[word][tag] > max_prob):
                max_prob = tr.word_tag_distribution[word][tag]
                max_tag = tag
        
        result[i] = max_tag
        
    return result
