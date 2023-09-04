import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
from networkx.drawing.nx_agraph import graphviz_layout
from sklearn.decomposition import PCA

# raw_data example:
# def PlotHM1(raw_data, save_plot, save_path):
#     # Convert the data into a pandas dataframe
#     df = pd.DataFrame(raw_data, columns=['char1', 'char2', 'value'])
#
#     # Pivot the dataframe to create a matrix
#     matrix = df.pivot(index='char1', columns='char2', values='value')
#
#     # Create the heatmap using seaborn
#     sns.heatmap(matrix, cmap='coolwarm', annot=True, fmt='.1f')
#     if save_plot==True:
#         plt.savefig(save_path)
#     plt.show()


def PlotHM(raw_data, save_plot, save_path):
    # Convert the data into a pandas dataframe
    G = nx.Graph()
    df = pd.DataFrame(raw_data, columns=['char1', 'char2', 'value'])

    # Get a list of unique characters to define the order of axes
    characters = sorted(list(set(df['char1'].tolist() + df['char2'].tolist())))

    # Pivot the dataframe to create a matrix with characters in a specific order
    matrix = df.pivot(index='char1', columns='char2', values='value')
    matrix = matrix.reindex(index=characters[::-1], columns=characters[::-1])  # Reversed order
    matrix = matrix.fillna(0)
    # Create the heatmap using seaborn
    ax = sns.heatmap(matrix, cmap='coolwarm', annot=True, fmt='.1f')

    # Ensure same labels on both x and y axes
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    if save_plot:
        plt.savefig(save_path)
    plt.close()
    print("(PlotHM) Done")




def PlotCov(matrix,book_names):
    # Create a seaborn heatmap
    sns.heatmap(matrix, annot=True, cmap="YlGnBu")

    # Set labels for rows and columns
    plt.xticks([x + 0.5 for x in range(len(matrix))], book_names)
    plt.yticks([y + 0.5 for y in range(len(matrix))], book_names)

    # Display the plot
    plt.show()

def generate_random_colors(length):
    colors = []
    for _ in range(length):
        # Generate a random color in hexadecimal format
        color = '#' + ''.join(random.choices('0123456789ABCDEF', k=6))
        colors.append(color)
    return colors

def PlotGraph(raw_data, save_plot, save_path,title,draw_edges = 1, num_of_books=1):
    # Create an empty graph
    G = nx.Graph()

    #generate random colors list
    colors = generate_random_colors(num_of_books)
    # Add the nodes to the graph
    if num_of_books>1:
        for pair in raw_data:
            G.add_node(pair[0],color=colors[pair[6]])
            G.add_node(pair[1],color=colors[pair[7]])
    else:
        for pair in raw_data:
            G.add_node(pair[0])
            G.add_node(pair[1])
    # Add the edges to the graph with their weights
    for edge in raw_data:
        G.add_edge(edge[0], edge[1], weight=round(edge[2],2))
    # Set the position of the nodes for visualization
    pos1 = nx.spring_layout(G,weight='weight', scale=10)
    # Draw the nodes and edges
    plt.figure(figsize=(16, 16))
    plt.title(title)
    if num_of_books>1:
        node_colors = [G.nodes[node]['color'] for node in G.nodes]
        nx.draw_networkx_nodes(G, pos1, node_size=400,node_color=node_colors)
    else:
        nx.draw_networkx_nodes(G, pos1, node_size=400)
    if draw_edges:
        nx.draw_networkx_edges(G, pos1, width=2, edge_color='gray',
                               edge_cmap=plt.cm.Blues, edge_vmin=0, edge_vmax=1)
    # Add the labels to the nodes
    labels = {}
    for node in G.nodes():
        labels[node] = node
    nx.draw_networkx_labels(G, pos1, labels, font_size=12, font_weight='bold')

    # Add the labels to the edges
    edge_labels = {}
    if draw_edges:
        for edge in G.edges(data=True):
            edge_labels[(edge[0], edge[1])] = edge[2]['weight']
        nx.draw_networkx_edge_labels(G, pos1, edge_labels, font_size=5)
    if save_plot==True:
        plt.savefig(save_path)
    # Show the plot
    plt.axis('off')
    plt.close()
    print("(PlotGraph) Done")
def PlotW2vEmbeddings2D(embeddings_dict,corpus_entities,entities_per_book):
    names = list(embeddings_dict.keys())
    book_labels = []
    for name in names:
        name_index = corpus_entities.index(name)
        book_idx =  name_index//entities_per_book
        book_labels.append(book_idx)
    embeddings = np.array(list(embeddings_dict.values()))
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(20, 20))

    # Create a list of unique group labels
    groups = list(set(book_labels))

    # Assign a color to each group
    colors = plt.cm.get_cmap('tab10', len(groups))

    # Plot the embeddings for each group with the corresponding color
    for i, group in enumerate(groups):
        indices = [idx for idx, label in enumerate(book_labels) if label == group]
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], color=colors(i), s=100)

    # Add labels to the points
    for i, name in enumerate(embeddings_dict.keys()):
        plt.annotate(name, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

    # Add explained variance of each component
    for i, explained_variance in enumerate(pca.explained_variance_ratio_):
        plt.text(0.05, 0.95 - i * 0.05, f"Component {i + 1} Variance: {explained_variance:.2f}", transform=plt.gca().transAxes)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Visualization of Word Embeddings')
    plt.show()

def PlotW2vEmbeddings3D(embeddings_dict, corpus_entities, entities_per_book):
    names = list(embeddings_dict.keys())
    book_labels = []
    for name in names:
        name_index = corpus_entities.index(name)
        book_idx = name_index // entities_per_book
        book_labels.append(book_idx)
    embeddings = np.array(list(embeddings_dict.values()))
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create a list of unique group labels
    groups = list(set(book_labels))

    # Assign a color to each group
    colors = plt.cm.get_cmap('tab10', len(groups))

    # Plot the embeddings for each group with the corresponding color
    for i, group in enumerate(groups):
        indices = [idx for idx, label in enumerate(book_labels) if label == group]
        ax.scatter(embeddings_3d[indices, 0], embeddings_3d[indices, 1], embeddings_3d[indices, 2], color=colors(i), s=100)

    # Add labels to the points
    for i, name in enumerate(embeddings_dict.keys()):
        ax.text(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], name)

    # Add explained variance of each component
    for i, explained_variance in enumerate(pca.explained_variance_ratio_):
        ax.text2D(0.05, 0.95 - i * 0.05, f"Component {i + 1} Variance: {explained_variance:.2f}", transform=ax.transAxes)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('Visualization of Word Embeddings')

    plt.show()