# numerical tag representation
# as specified in https://universaldependencies.org/u/pos/
tag_set = {
    # bogus token to represent start of sentece
    # we acutally don't use it to tag any words
    'START': 0,
    'ADJ': 1,
    'ADV': 2,
    'NOUN': 3,
    'VERB': 4,
    'PROPN': 5,
    'INTJ': 6,
    'ADP': 7,
    'AUX': 8,
    'CCONJ': 9,
    'DET': 10,
    'NUM': 11,
    'PART': 12,
    'PRON': 13,
    'SCONJ': 14,
    'PUNCT': 15,
    'SYM': 16,
    'X': 17,
    # bogus token to represent end of sentece
    'END': 18
}

# Filename only, no extension needed
TRAINING_SET = "mixed-train"
TEST_SET = "mixed-test"
DEV_SET = "mixed-dev"

# selects smooting strategy
#{
#    "NOUNS": 0,
#    "NOUNS_VERBS": 1,
#    "UNIFORM": 2,
#    "SINGLE_WORDS": 3
#}
SMOOTHING = 3