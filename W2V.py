
import gensim.downloader as api
from gensim.models import Word2Vec
from gensim.utils import tokenize
import json
import os
from scipy import spatial
import pandas as pd

def read_corpus(corpus):
    for doc in corpus:
        yield list(tokenize(doc))


def TrainW2VModel(book_name, corpus_list, vector_size,window_size,output_path):

    output_text_file = output_path + book_name + "_w2v.txt"
    dir_path = os.path.dirname(output_text_file)
    # create the directory if it does not exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # load the corpus
    corpus = list(read_corpus(corpus_list))
    new_list = [[]]
    for i in corpus_list:
        new_list[0].append(i)

    # create a new Word2Vec model with the same parameters as the pre-trained model
    #fine_tuned_model = Word2Vec(sentences=new_list,vector_size=vector_size, min_count=5, window=window_size, sg=1)

    # build the vocabulary from the corpus
    #fine_tuned_model.build_vocab(corpus)

    # fine-tune the model on the new corpus
    #fine_tuned_model.train(corpus, total_examples=fine_tuned_model.corpus_count, epochs=10)
    fine_tuned_model = Word2Vec(new_list, vector_size=vector_size, window=window_size, min_count=5, workers=4, epochs=10, negative=10)
    fine_tuned_model.wv.save_word2vec_format(output_text_file, binary=False)

    # Open up that text file and convert to JSON
    f = open(output_text_file)
    v = {"vectors": {}}
    for line in f:
        w, n = line.split(" ", 1)
        v["vectors"][w] = list(map(float, n.split()))

    # Save to a JSON file
    # Could make this an optional argument to specify output file
    with open(output_text_file[:-4]+".json", "w") as out:
        #json.dump(v, out)
        print(output_text_file[:-4]+"json")
        json.dump(v, out)


    print("Done - W2V Training")
    return v





def GenW2V(entities,vectors):
    # All possible pairs in List
    all_pairs = [(a, b) for idx, a in enumerate(entities) for b in entities[idx + 1:]]
    for i in range(0,len(all_pairs)):
      all_pairs[i]=list(all_pairs[i])
    i = 0
    # compute cosine similarity
    while i < len(all_pairs):
        if len(all_pairs[i][0][0])>1:
            first_in_pair = all_pairs[i][0][0]
            sec_in_pair = all_pairs[i][1][0]
        else:
            first_in_pair = all_pairs[i][0]
            sec_in_pair = all_pairs[i][1]
        if ((not (first_in_pair in vectors['vectors'])) or (not (sec_in_pair in vectors['vectors']))):
            del all_pairs[i]
            continue
        sim1 = 1 - spatial.distance.cosine(vectors['vectors'][first_in_pair], vectors['vectors'][sec_in_pair])
        all_pairs[i].append(sim1)
        i += 1

    all_pairs.sort(key=lambda x: x[2])
    all_pairs.reverse()
    for i in range(len(all_pairs)):
        if len(all_pairs[i][0][0])>1:
            all_pairs[i][0] = all_pairs[i][0][0]
            all_pairs[i][1] = all_pairs[i][1][0]
        else:
            all_pairs[i][0] = all_pairs[i][0]
            all_pairs[i][1] = all_pairs[i][1]
    return all_pairs





