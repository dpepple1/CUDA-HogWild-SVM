import cugraph
import cudf

graph= cudf.read_csv("webGoogle.csv", delimiter='\t', dtype=['int32', 'int32'])
print(graph.head())
g=cugraph.Graph()
g.from_cudf_edgelist(graph, source='FromNodeId', destination='ToNodeId')
df = cugraph.spectralBalancedCutClustering(g, 5)
print(df)
