import sys
import Book

# Arguments:
# 1. `book_path`    - represents the path to the corresponding `.txt` file of the book under analysis.
# 2. `entities_pth` - specifies a list of entities present in the book.
#                     These entities are identified using various ways of reference, as exemplified in the test folder.
# 3. `book_name`    - Refers to the name of the book being processed.
# 4. `output_dir`   - Designates the desired output directory for the generated results.



# For running the test use the following command and run it from root dir:
#python3 src/RunCRELAT_Book.py "test/Harry Potter 1/Harry Potter 1.txt" "test/Harry Potter 1/Entities" "Harry Potter 1" "test/output"


def main():
    if len(sys.argv) != 5:
        print(f"Usage: RunCRELAT_Book.py haven't recieved enough arguments. Expected 4 got {len(sys.argv)-1}")
        return

    book_path = sys.argv[1]
    entities_path = sys.argv[2]
    book_name = sys.argv[3]
    output_dir = sys.argv[4]



    Book1 = Book.Book(book_path, entities_path, book_name)
    Book1.RemoveStopWords()
    Book1.ToOneRef()
    Book1.PrintChars()
    Book1.GenCoOc(15)
    Book1.TrainW2VModel(7, 15, output_dir)
    Book1.GenW2V()
    #Book1.PloHM(0,False,"C:/Users/Dani2.OFER3090PC/Desktop/omrirafa/NLP/ReLAN/output/Great_Expectations_CoOc_HM")
    #Book1.PloHM(1,False,"C:/Users/Dani2.OFER3090PC/Desktop/omrirafa/NLP/ReLAN/output/Great_Expectations_W2V_HM")
    #Book1.PlotGraph(0,False,"C:/Users/Dani2.OFER3090PC/Desktop/omrirafa/NLP/ReLAN/output/Great_Expectations_CoOc_Graph")
    #Book1.PlotGraph(1, False, "C:/Users/Dani2.OFER3090PC/Desktop/omrirafa/NLP/ReLAN/output/A_Catcher_In_The_Rye_Graph")
    #Book1.SubGraph(0,1,True,"C:/Users/Dani2.OFER3090PC/Desktop/omrirafa/NLP/ReLAN/output/Great_Expectations_W2V_vs_CoOc_Graph")
    #Book1.SaveDiffList("C:/Users/Dani2.OFER3090PC/Desktop/omrirafa/NLP/ReLAN/output/Great_Expectations_Diff_List.xls")
    #Book1.PrintDiffList()
    #Book1.ClusterList(1,"C:/Users/Dani2.OFER3090PC/Desktop/omrirafa/NLP/ReLAN/output/Great_Expectations_Clusters_W2V.xls",3)
    #Book1.ClusterAll("C:/Users/Dani2.OFER3090PC/Desktop/omrirafa/NLP/ReLAN/output/Great_Expectations_Clusters_All.xls",5)
    print("End Of script")

#if __name__ == "__main__":
#    main()


#------------------------------------
book_path = '/Users/omrirafa/Desktop/University/Thesis/CRELAT/test/Harry Potter 1/Harry Potter 1.txt'
entities_path = '/Users/omrirafa/Desktop/University/Thesis/CRELAT/test/Harry Potter 1/Entities'
book_name = 'Harry Potter 1'
output_dir = '/Users/omrirafa/Desktop/University/Thesis/CRELAT/test/output'



Book1 = Book.Book(book_path, entities_path, book_name)
Book1.RemoveStopWords()
Book1.ToOneRef()
Book1.PrintChars()
Book1.GenCoOc(window_size=15)
#Co-Occurances Graph
Book1.PlotGraph(data_sel=0,save_plot=True,save_path=f"{output_dir}/{book_name}_CoOc_Graph")
#Co-Occurances Heat Map
Book1.PloHM(data_sel=0,save_plot=True,save_path=f"{output_dir}/{book_name}_CoOc_HeatMap")
Book1.TrainW2VModel(7, 15, output_dir)
Book1.GenW2V()
#W2V Cosine Graph
Book1.PlotGraph(data_sel=1,save_plot=True,save_path=f"{output_dir}/{book_name}_Cosine_Graph")
#W2V Cosine HM
Book1.PloHM(data_sel=1,save_plot=True,save_path=f"{output_dir}/{book_name}_Cosine_HeatMap")
#Cosine-Similarity minus Normalized Co-Occurances Graph (to highlight Gap)
Book1.SubGraph(data_sel0=0,data_sel1=1,save_plot=True,save_path=f"{output_dir}/{book_name}_CoOc_Minus_Cosine_Graph")
#Excel file with the difference between the 2 metrics for each pair of characters
Book1.SaveDiffList(save_path=f"{output_dir}/{book_name}_Diff_List.xls")
#cluster pairs connection according to each metric (pairs with similar values will be assign to the same cluster)
Book1.ClusterAll(save_path=f"{output_dir}/{book_name}_Clustered.xls",n_clusters=5)
print("End Of script")