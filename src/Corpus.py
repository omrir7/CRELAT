import numpy as np

import Book
import os
import json
import chardet
import pandas as pd
import W2V
import Graphs
import statistics
import BERT_Inference_Without_Finetune

def Corpus_Main(books_dir, output_dir, entities_per_book, w2v_vector_size, w2v_window, cooc_window, corpus_name):
    # Initialize the lists to store the data
    txt_list = []
    names_list = []
    txt_path_list = []
    names_path_list = []
    # Iterate over the folders in the specified directory
    for folder_name in os.listdir(books_dir):
        folder_path = os.path.join(books_dir, folder_name)

        # Check if the item in the directory is a folder
        if os.path.isdir(folder_path):
            # Read the .txt file
            txt_file_path = os.path.join(folder_path, folder_name + ".txt")
            txt_path_list.append(txt_file_path)

            # Read the text file inside the "outputdir" folder
            entities_path = os.path.join(folder_path, "")
            file_path = os.path.join(entities_path,"Entities")
            names_path_list.append(file_path)

    book_object_list = []
    for cur_names_path, cur_txt_path in zip(names_path_list, txt_path_list):
        book_name = os.path.splitext(os.path.basename(cur_txt_path))[0]
        cur_book = Book.Book(cur_txt_path, cur_names_path,book_name)
        cur_book.RemoveStopWords()
        cur_book.ToOneRef()
        cur_book.TrainW2VModel(w2v_vector_size, w2v_window, output_dir)
        #cur_book.GenW2V()
        cur_book.GenCoOc(cooc_window)
        cur_book.extract_entity_contexts()
        cur_book.inference_bert()
        cur_book.average_bert_embeddings()
        book_object_list.append(cur_book)
    corpus_entities = []
    entities_vectors_dict_w2v = dict()
    entities_vectors_dict_w2v['vectors'] = dict()
    entities_vectors_dict_bert = dict()
    entities_vectors_dict_bert['vectors'] = dict()

    books_entities_lists_sep = []
    for book in book_object_list:
        cur_entities = []
        for character in book.entities[:entities_per_book]:
            corpus_entities.append(character[0])
            cur_entities.append(character[0])
        books_entities_lists_sep.append(cur_entities)
        original_w2v_dict = book.w2v_vectors['vectors']
        keys_to_keep = [key for key in cur_entities if key in original_w2v_dict.keys()]
        short_dict = dict()
        for key in keys_to_keep:
            short_dict[key] = original_w2v_dict[key]
        entities_vectors_dict_w2v['vectors'].update(short_dict)
        entities_vectors_dict_bert['vectors'].update(book.entities_embeddings_averaged_bert)
    #remove entities that appear only in one of the dicts (bert or w2v)
    common_keys = set(entities_vectors_dict_w2v['vectors'].keys()).intersection(entities_vectors_dict_bert['vectors'].keys())
    entities_vectors_dict_w2v['vectors'] = {key: entities_vectors_dict_w2v['vectors'][key] for key in common_keys}
    entities_vectors_dict_bert['vectors'] = {key: entities_vectors_dict_bert['vectors'][key] for key in common_keys}

    print(1)

    #Generate a Cosine Similarity file for the COrpus
    all_pairs = W2V.GenW2V(corpus_entities,entities_vectors_dict_w2v)
    all_pairs_bert = W2V.GenW2V(corpus_entities,entities_vectors_dict_bert)

    all_pairs, all_pairs_bert = order_lists(all_pairs,all_pairs_bert)

    #Same Book or Not indication
    for idx, pair in enumerate(all_pairs):
        char1 = pair[0]
        char2 = pair[1]
        # Check if both characters are in the same extra list
        same_book = any(char1 in book_entities_list and char2 in book_entities_list for book_entities_list in books_entities_lists_sep)

        # Append the corresponding number to the sublist
        pair.append(1 if same_book else 0)
        for book in book_object_list:
            cur_book_entities = [ent[0] for ent in book.entities][:entities_per_book]
            if char1 in cur_book_entities:
                char1_book = book.book_name
            if char2 in cur_book_entities:
                char2_book = book.book_name
        pair.append(char1_book)
        pair.append(char2_book)
        pair.append(all_pairs_bert[idx][2])



    df = pd.DataFrame(all_pairs, columns=['Char1', 'Char2', 'Cosine Similarity W2V','Same Book','Char Book', 'Char2 Book', 'Cosine Similarity BERT'])
    df.to_csv(output_dir+corpus_name+".csv", index=False, encoding='utf-8')


    for i in range(0,len(all_pairs)):
        char1_index = corpus_entities.index(all_pairs[i][0])
        char2_index = corpus_entities.index(all_pairs[i][1])

        char1_book_idx = char1_index//entities_per_book
        char2_book_idx = char2_index//entities_per_book
        all_pairs[i].append(char1_book_idx)
        all_pairs[i].append(char2_book_idx)


    #all pairs [0] - char1
    #all pairs [1] - char2
    #all pairs [2] - cosine similarity
    #all pairs [3] - char1 book
    #all pairs [4] - char2 book

    # COV Matrix Building:
    # Option 1 - M[i,j] - AVG Cosine of book_i, book_j.
    # Option 2 - M[i,j] - Actual covariance of cosine similarity.

    #option 1:
    book_names = [name.split("/")[-1][:-4] for name in txt_path_list]
    cov_option_1 = np.zeros((len(book_names),len(book_names)))
    for i in book_names:
        for j in book_names:
            cur_pairs = [pair for pair in all_pairs if (pair[4] == i and pair[5] == j) or (pair[4] == j and pair[5] == i)]
            if (len(cur_pairs)>0):
                cosine_similarities = [pair[2] for pair in cur_pairs]
                average = statistics.mean(cosine_similarities)
                cov_option_1[book_names.index(i)][book_names.index(j)] = average


    Graphs.PlotCov(cov_option_1,book_names)



    #Graphs.PlotGraph(all_pairs,1,output_dir+corpus_name+"_W2V_Graph",corpus_name+"_W2V_Graph",0,5)
    #Graphs.PlotW2vEmbeddings2D(entities_vectors_dict_w2v['vectors'],corpus_entities,entities_per_book)
    #Graphs.PlotW2vEmbeddings3D(entities_vectors_dict_w2v['vectors'],corpus_entities,entities_per_book)

    print("done")
    return book_object_list


def order_lists(list1, list2):
    # Convert each list to dictionaries for easy lookup
    dict1 = {tuple(sorted(item[:2])): item[2] for item in list1}
    dict2 = {tuple(sorted(item[:2])): item[2] for item in list2}

    # Get all unique character pairs from both lists
    all_chars = set(dict1.keys()).union(set(dict2.keys()))

    # Create ordered lists
    ordered_list1 = [[char[0], char[1], dict1[char]] for char in all_chars if char in dict1]
    ordered_list2 = [[char[0], char[1], dict2[char]] for char in all_chars if char in dict2]

    return ordered_list1, ordered_list2

books_dir =  '../../Data/Short_Stories'
output_dir = "../test/Short Stories Corpus"
entities_per_book = 5
w2v_vector_size = 200
w2v_window = 7
cooc_window = 15
corpus_name = "Short Stories"
book_object_list = Corpus_Main(books_dir, output_dir, entities_per_book, w2v_vector_size, w2v_window, cooc_window, corpus_name)