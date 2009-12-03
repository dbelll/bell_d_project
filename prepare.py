import cuda_kmeans_tri as kmt
import numpy as np

kmt.VERBOSE = 1

#data = np.array([[.926, .381, .351, .633, .852, .942, .907, .343, .313, .998],
#                 [.796, .599, .635, .933, .340, .130, .956, .597, .912, .799]])
#clusters = np.array([[.2909, .4651, .8569],
#                     [.7381, .1116, .3702]])

nDim = 2
nPts = 10
nClusters = 4

np.random.seed(100)
data = np.random.rand(nDim, nPts)
clusters = np.random.rand(nDim, nClusters)

#(clusters2, labels) = kmt.trikmeans_gpu(data, clusters, 1)

kmt.run_tests1(1, nPts, nDim, nClusters, 1, 1)
