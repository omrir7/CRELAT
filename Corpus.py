import numpy as np

import Book
import os
import json
import chardet
import pandas as pd
import W2V
import Graphs
import statistics
print(1)
books_dir =  '/home/omrirafa/thesis/NLP/BookNLP/Amos Oz - United'
output_dir = "output_corpus/"
entities_per_book = 5
w2v_vector_size = 200
w2v_window = 7
cooc_window = 15
corpus_name = "Amos_Oz_Corpus"


# Initialize the lists to store the data
txt_list = []
names_list = []
txt_path_list = []
names_path_list = []
num_of_books = 0
# Iterate over the folders in the specified directory
for folder_name in os.listdir(books_dir):
    folder_path = os.path.join(books_dir, folder_name)

    # Check if the item in the directory is a folder
    if os.path.isdir(folder_path):
        num_of_books+=1
        # Read the .txt file
        txt_file_path = os.path.join(folder_path, folder_name + ".txt")
        txt_path_list.append(txt_file_path)

        # Read the text file inside the "outputdir" folder
        outputdir_path = os.path.join(folder_path, "outputdir")
        for file_name in os.listdir(outputdir_path):
            file_path = os.path.join(outputdir_path, file_name)
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
    book_object_list.append(cur_book)

corpus_entities = []
entities_vectors_dict = dict()
entities_vectors_dict['vectors'] = dict()
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
    entities_vectors_dict['vectors'].update(short_dict)
print(1)

#Generate a Cosine Similarity file for the COrpus
all_pairs = W2V.GenW2V(corpus_entities,entities_vectors_dict)




#Same Book or Not indication
for pair in all_pairs:
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



df = pd.DataFrame(all_pairs, columns=['Char1', 'Char2', 'Cosine Similarity','Same Book','Char Book', 'Char2 Book'])
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
        cosine_similarities = [pair[2] for pair in cur_pairs]
        average = statistics.mean(cosine_similarities)
        cov_option_1[book_names.index(i)][book_names.index(j)] = average


Graphs.PlotCov(cov_option_1,book_names)



#Graphs.PlotGraph(all_pairs,1,output_dir+corpus_name+"_W2V_Graph",corpus_name+"_W2V_Graph",0,5)
#Graphs.PlotW2vEmbeddings2D(entities_vectors_dict['vectors'],corpus_entities,entities_per_book)
#Graphs.PlotW2vEmbeddings3D(entities_vectors_dict['vectors'],corpus_entities,entities_per_book)
print("done")

