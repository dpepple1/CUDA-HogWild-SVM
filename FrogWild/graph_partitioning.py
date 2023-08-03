import cugraph
import cudf

num_clusters=10
graph= cudf.read_csv("webGoogle.csv", delimiter='\t', dtype=['int32', 'int32'])
print(graph.head())
g=cugraph.Graph()
g.from_cudf_edgelist(graph, source='FromNodeId', destination='ToNodeId')
df = cugraph.spectralBalancedCutClustering(g, num_clusters=num_clusters, num_eigen_vects=num_clusters-2)
print(cugraph.analyzeClustering_edge_cut(g,num_clusters,df))
df.to_csv("Cluster_Assignment.csv")

