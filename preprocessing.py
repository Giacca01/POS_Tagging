# Preprocess a .conllu file keeping only
# words and their tags
def preprocess(raw_file):
    file = open("raw_files/" + raw_file + ".conllu", "r")
    processed_file = open("processed_files/" + raw_file + ".txt", "w")

    start_counter = 0

    for line in file:
        if (line != "\n"):
            if (line[0] != '#'):
                splitted_line = line.split("\t")
                # discards multiword tokens
                # whose components will be treated one by one
                # while processing the next lines
                if len(splitted_line[0]) < 3:
                    processed_file.write(splitted_line[1] + "\t" + splitted_line[3] + "\n")
            else:
                start_counter = start_counter + 1
                if start_counter == 2:
                    # to represent the beginning of the sentence
                    # and be able to tag the first real word.
                    processed_file.write("START\n")
                    start_counter = 0
        else:
            processed_file.write("END\n")

    file.close()
    processed_file.close()