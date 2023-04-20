from sklearn.cluster._dbscan import DBSCAN
from sklearn.cluster._kmeans import KMeans

def getCluster(features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    cluster_id =  kmeans.labels_
    # relToClu = {}
    # for rel_id, clu_id in enumerate(cluster_id):
    #     relToClu[rel_id] = clu_id
    return cluster_id

