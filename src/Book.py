
from nltk.corpus import stopwords
import string
import nltk
from gensim.parsing.preprocessing import remove_stopwords
from nltk.corpus import stopwords
import CoOccurances
import W2V
nltk.download('stopwords')
import os
import Graphs
import pandas as pd
import Clustering

class Book:
    def __init__(self,book_txt_path, Entities_file_path,book_name):
        file = open(book_txt_path, encoding="utf-8")
        line = file.read()
        tokens = line.split()
        self.raw_txt_list = [i.lower() for i in tokens]
        file.close()

        file = open(Entities_file_path, encoding="utf-8")
        entities = []
        for line in file:
            stripped_line = line.strip()
            line_list = stripped_line.split()
            entities.append(line_list)
        file.close()
        self.entities = entities

        #Initializations
        self.book_name = book_name
        self.no_stop_words_list = None
        self.one_ref_list = None
        self.co_occurances_raw = None

    def RemoveStopWords(self):
        stop_words = set(stopwords.words('english'))

        for word in stop_words:
            word = word.replace('\"', '')
            word = word.replace('“', '')
            word = word.replace('”', '')
            word = word.replace("’", '')
            word = word.replace('\'', '')

        words = []
        for r in self.raw_txt_list:
            r = r.translate(str.maketrans('', '', string.punctuation))
            r = r.replace('\"', '')
            r = r.replace('“', '')
            r = r.replace('”', '')
            r = r.replace("’", '')
            r = r.replace('\'', '')
            if not r in stop_words:
                words.append(r)
        words_lower = [word.lower() for word in words]
        self.no_stop_words_list = words_lower
        print("(RemoveStopWords) Done")

    def index_2d(self, names2d, appearance):
        # if (appearance=="yonatan"):
        #  print(1)
        for i, x in enumerate(names2d):
            if appearance in x:
                return i
        return False

    def ToOneRef(self):
        self.one_ref_list = self.no_stop_words_list
        for i in range(0, len(self.one_ref_list)):
            entity_apear = self.index_2d(self.entities, self.one_ref_list[i])
            if entity_apear is not False:
                self.one_ref_list[i] = self.entities[entity_apear][0]
        print("(ToOneRef) Done")
    def BookName(self):
        print(self.book_name)
    def NumOfEntities(self):
        print(len(self.entities))
    def GenCoOc(self,window_size):
        co_occurances = CoOccurances.CoOcCount(self.entities,self.one_ref_list,window_size)
        self.co_occurances_raw = co_occurances
    def TrainW2VModel(self, vector_size,window_size,output_path): #otput path without file, only directory
        if not os.path.exists(output_path):
            raise Exception(f"Output path '{output_path}' does not exist.")
            return False
        self.w2v_vectors = W2V.TrainW2VModel(self.book_name,self.one_ref_list,vector_size,window_size,output_path)
    def GenW2V(self):
        self.w2v_pairs = W2V.GenW2V(self.entities,self.w2v_vectors)
    def PloHM(self,data_sel,save_plot,save_path):
        if data_sel==0:
            data=self.co_occurances_raw
        elif data_sel==1:
            data=self.w2v_pairs
        Graphs.PlotHM(data,save_plot,save_path)
    def PlotGraph(self,data_sel,save_plot,save_path):
        if data_sel==0:
            data=self.co_occurances_raw
            title = self.book_name + " Co-Occurances"
        elif data_sel==1:
            data=self.w2v_pairs
            title = self.book_name + " W2V Cosine Similarity"
        Graphs.PlotGraph(data,save_plot,save_path,title)
    def PrintChars(self):
        i=1
        print("(Entities)")
        for char in self.entities:
            print(f"{i}. {char}")
            i+=1
    def SubGraph(self,data_sel0,data_sel1,save_plot,save_path):
        #Normalizing Co_Ocuurances to be 0-1
        co_oc_max = max([i[2] for i in self.co_occurances_raw])
        norm_co_occurances = [[i[0], i[1], i[2]/co_oc_max] for i in self.co_occurances_raw]

        if data_sel0==0:
            data0=norm_co_occurances
            data0_type = "Co-Occurances"
        elif data_sel0==1:
            data0=self.w2v_pairs
            data0_type = " W2V Cosine Similarity"

        if data_sel1==0:
            data1=norm_co_occurances
            data1_type = "Co-Occurances"
        elif data_sel1==1:
            data1=self.w2v_pairs
            data1_type = " W2V Cosine Similarity"

        diff_list = []
        for a in data0:
            for b in data1:
                if set(a[:2]) == set(b[:2]):
                    diff_list.append([a[0], a[1], a[2] - b[2]]) #TBD
        self.diff_list = diff_list
        self.diff_list.sort(key=lambda x: x[2])
        title = self.book_name+" - Difference Between "+ data0_type+" and "+data1_type
        Graphs.PlotGraph(diff_list,save_plot,save_path,title)
    def PrintDiffList(self):
        print(self.diff_list)
    def SaveDiffList(self,save_path):
        df = pd.DataFrame(self.diff_list, columns=['Char1', 'Char2','Difference'])
        df.to_csv(save_path, index=False, sep='\t')
    def ClusterList(self,data_sel,save_path,n_clusters):
        if data_sel==0:
            data = self.co_occurances_raw
            data_type = "Co-Occurances"
        elif data_sel==1:
            data=self.w2v_pairs
            data_type = "W2V Cosine Similarity"
        clusters_labels,clusters_centers = Clustering.Cluster(data,n_clusters)

        new_list = []
        for label, couple in zip(clusters_labels, data):
            temp_couple = couple
            temp_couple.append(label)
            new_list.append(temp_couple)
        new_list.sort(key=lambda x: x[2])
        df = pd.DataFrame(new_list, columns=['Char1', 'Char2',data_type,'Cluster'])
        df.to_csv(save_path, index=False, sep='\t')
        print("!")
    def ClusterAll(self,save_path,n_clusters):
        data_cooc = self.co_occurances_raw
        data_w2v = self.w2v_pairs
        clusters_labels_cooc,clusters_centers_cooc = Clustering.Cluster(data_cooc,n_clusters)
        clusters_labels_w2v,clusters_centers_w2v = Clustering.Cluster(data_w2v,n_clusters)

        clusters_centers_cooc1 = sorted(list(clusters_centers_cooc))
        clusters_centers_w2v1 = sorted(list(clusters_centers_w2v))
        cooc_map = dict()
        w2v_map = dict()
        for i in range(0,len(clusters_centers_cooc)):
            cooc_map[i] = clusters_centers_cooc1.index(clusters_centers_cooc[i])
        for i in range(0,len(clusters_centers_w2v)):
            w2v_map[i] = clusters_centers_w2v1.index(clusters_centers_w2v[i])

        for i in range(0,len(clusters_labels_cooc)):
            clusters_labels_cooc[i] = cooc_map[clusters_labels_cooc[i]]
        for i in range(0,len(clusters_labels_w2v)):
            clusters_labels_w2v[i] = w2v_map[clusters_labels_w2v[i]]

        new_list_cooc = []
        for label, couple in zip(clusters_labels_cooc, data_cooc):
            temp_couple = couple
            temp_couple.append(label)
            new_list_cooc.append(temp_couple)
        new_list_cooc.sort(key=lambda x: x[2])

        new_list_w2v = []
        for label, couple in zip(clusters_labels_w2v, data_w2v):
            temp_couple = couple
            temp_couple.append(label)
            new_list_w2v.append(temp_couple)
        new_list_w2v.sort(key=lambda x: x[2])

        # Concatenate the two lists
        concatenated_list = []
        concatenated_list.extend(new_list_cooc)
        concatenated_list.extend(new_list_w2v)

        # Create a DataFrame
        df = pd.DataFrame(concatenated_list, columns=['char1', 'char2','Co-Occurances', 'Co-Occurances Cluster'])

        # Create a dictionary to store the values for each pair of characters
        values_dict = {}

        # Iterate over the concatenated list
        for sublist in concatenated_list:
            charA, charB, val1, val2 = sublist
            key = frozenset([charA, charB])
            if key not in values_dict:
                values_dict[key] = [val1, val2]
            else:
                values_dict[key].extend([val1, val2])

        # Create a new list to store the modified rows
        modified_rows = []

        # Update the DataFrame with the additional values
        for index, row in df.iterrows():
            charA, charB = row['char1'], row['char2']
            key = frozenset([charA, charB])
            if key in values_dict:
                values = values_dict[key]
                if len(values)==2:
                    #giving couples with no CoOc value 0 with cluster that have the lowest center
                    lowest_value = min(clusters_centers_cooc)
                    lowest_index = list(clusters_centers_cooc).index(lowest_value)
                    values = [0,lowest_index] + values
                modified_rows.append([charA, charB, values[0], values[1], values[2], values[3]])
                values_dict.pop(key)  # Remove the used values from the dictionary

        # Create the final DataFrame
        df = pd.DataFrame(modified_rows, columns=['char1', 'char2', 'Co-Occurances', 'Co-Occurances Cluster', 'Cosine Similarity', 'Cosine Similarity Cluster'])

        df.to_csv(save_path, index=False, sep='\t')
        print("!")

