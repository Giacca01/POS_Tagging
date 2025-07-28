import decoding as dc
import shared_data as sd
import training as tr
import smoothing as sm
import preprocessing as pr
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


observations = []
correct_tagging = []

def main():
    print("Preprocessing Training Set...\n")
    pr.preprocess(sd.TRAINING_SET)

    print("Training...\n")
    tr.train(sd.TRAINING_SET)

    if (sd.SMOOTHING == 3):
        print("Preprocessing Dev Set...\n")
        pr.preprocess(sd.DEV_SET)
        print("Smoothing...\n")
        sm.single_word_smoothing()
    elif (sd.SMOOTHING == 0):
        print("Smoothing...\n")
        sm.nouns_smoothing()
    elif(sd.SMOOTHING == 1):
        print("Smoothing...\n")
        sm.nouns_verb_smoothing()
    else:
        print("Smoothing...\n")
        sm.uniform_smoothing()

    print("Preprocessing Test Set...\n")
    pr.preprocess(sd.TEST_SET)
    print("Reading Test Set...\n")
    read_test_set()
    print("Evaluating majority tagging: \n")
    evaluate(dc.majority_tagging)

    print("Evaluating viterbi: \n")
    evaluate(dc.viterbi)

# reading test set for evaluation
def read_test_set():
    sentence = []
    sentece_tagging = []

    file = open("processed_files/" + sd.TEST_SET + ".txt")
    for line in file:
        line = line.strip("\n")
        if line != "START":
            if line != "END":
                splitted_line = line.split("\t")
                curr_word = splitted_line[0]
                sentence.append(curr_word)

                curr_token = sd.tag_set[splitted_line[1]]
                sentece_tagging.append(curr_token)
            else:
                # END non viene inserito nella frase
                observations.append(sentence)
                correct_tagging.append(sentece_tagging)
                sentence = []
                sentece_tagging = []


# 2 - applicazione dell'algoritmo di tagging
def evaluate(pos_algorithm):
    gold_standard = []
    results = []

    for i in range(len(observations)):
        result = pos_algorithm(observations[i])
        results.extend(result)

        sentece_tagging = correct_tagging[i]
        gold_standard.extend(sentece_tagging)
    
    print(classification_report(gold_standard, results))
    cm = confusion_matrix(gold_standard, results)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.show()


if __name__ == "__main__":
    main()
