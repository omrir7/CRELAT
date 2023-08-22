
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans



def Cluster(data, n_cluters):

    kmeans = KMeans(
                    init="random",
                    n_clusters=n_cluters,
                    n_init=10,
                    max_iter=300,
                    random_state=42)


    values = [a[2] for a in data]
    values = [[c] for c in values]

    kmeans.fit(values)
    labels = kmeans.labels_.tolist()
    clusters_labels = labels


    clusters_centers =  kmeans.cluster_centers_
    print('done')
    return clusters_labels, clusters_centers

