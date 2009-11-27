import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pycuda.reduction import ReductionKernel

import numpy as np
import math

import time

VERBOSE = 0
PRINT_TIMES = 0


from cpu_kmeans import kmeans_cpu
from cpu_kmeans import assign_cpu
from cpu_kmeans import calc_cpu

"""
#------------------------------------------------------------------------------------
#                               kmeans on the cpu
#------------------------------------------------------------------------------------

def kmeans_cpu(data, clusters, iterations):
    # kmeans_cpu(data, clusters, iterations) returns (clusters, labels)
    
    for i in range(iterations):
        assign = assign_cpu(data, clusters)
        clusters = calc_cpu(data, assign, clusters)
    return (clusters, assign)
    
def assign_cpu(data, clusters):
    # assign data to the nearest cluster, using cpu
    
    cpu_dist = np.sqrt(((data[:,:,np.newaxis]-clusters[:,np.newaxis,:])**2).sum(0))
    return np.argmin(cpu_dist, 1)

def calc_cpu(data, assign, clusters):
    # calculate new clusters for the data based on assignments

    # calc_cpu(data, assign, clusters)
    # clusters argument is the current clusters
    # returns the recalculated clusters
    
    (nDim, nPts) = data.shape
    nClusters = clusters.shape[1]
    
    c = np.arange(nClusters)
    assign.shape = (nPts, 1)
    c_counts = np.sum(assign == c, axis=0)
    cpu_new_clusters = np.sum(data[:,:,np.newaxis] * (assign==c)[np.newaxis,:,:], axis=1) / (c_counts + (c_counts == 0))
    cpu_new_clusters = cpu_new_clusters + clusters * (c_counts == 0)[np.newaxis:,]
    return cpu_new_clusters
"""

    
#------------------------------------------------------------------------------------
#                               kmeans on the gpu
#------------------------------------------------------------------------------------

def kmeans_gpu(data, clusters, iterations, return_times = 0):
    # kmeans_gpu(data, clusters, iterations) returns (clusters, labels)
    
    # kmeans using standard algorithm and cuda
    # input arguments are the data, intial cluster values, and number of iterations to repeat
    # The shape of data is (nDim, nPts) where nDim = # of dimensions in the data and
    # nPts = number of data points
    # The shape of clusters is (nDim, nClusters) 
    #
    # The return values are the updated clusters and labels for the data
    
    (nDim, nPts) = data.shape
    nClusters = clusters.shape[1]
    
    data = np.array(data).astype(np.float32)
    clusters = np.array(clusters).astype(np.float32)
    
    # block and grid sizes for the cluster_assign kernel
    threads_desired = max(nPts, nDim*nClusters)
    block_size_assign = min(256, threads_desired)
    grid_size_assign = 1 + (threads_desired - 1)/block_size_assign
    
    # block and grid sizes for the cluster_calc kernel
    for block_size_calc_x in range(32, 512, 32):
        #print "block_size_calc_x",block_size_calc_x
        if block_size_calc_x >= nClusters:
            break;
    block_size_calc_y = min(nDim, 512/block_size_calc_x)
    grid_size_calc_x = 1 + (nClusters-1)/block_size_calc_x
    shared_memory = cuda.Device(0).get_attributes()[cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK]

    # system uses 16 bytes of shared memory
    data_staging_memory = shared_memory/2 - 32 - 4* block_size_calc_x * block_size_calc_y  - 4 * block_size_calc_x
    data_chunk_size = data_staging_memory / 4 / block_size_calc_y
    data_reps = 1 + (nPts-1)/data_chunk_size
    
    # copy the data to the GPU
    t1 = time.time()
    gpu_data = gpuarray.to_gpu(data)
    gpu_clusters = gpuarray.to_gpu(clusters)
    
    gpu_distances = np.zeros((nPts,nClusters)).astype(np.float32)   #// **TODO**  dont bother with distances
    gpu_assignments = np.zeros(nPts).astype(np.int32)               #// **TODO**  leave assignments on the gpu

    pycuda.autoinit.context.synchronize()
    t2 = time.time()
    data_time = t2-t1
    
    # get the functions from the source modules
    t1 = time.time()
    mod_assign = get_assign_module(nDim, nPts, nClusters, block_size_assign)
    mod_calc = get_calc_module(nDim, nPts, nClusters, block_size_calc_x, block_size_calc_y, data_chunk_size, data_reps)

    cluster_assign = mod_assign.get_function("cluster_assign")
    cluster_calc = mod_calc.get_function("cluster_calc")
    pycuda.autoinit.context.synchronize()
    t2 = time.time()
    module_time = t2-t1
    
    assign_time = 0.
    calc_time = 0.
    
    

    for i in range(iterations):
        t1 = time.time()
        #print "cluster_assign blocksize", block_size_assign, 1, 1
        #print "cluster_assign gridsizze", grid_size_assign, 1
        cluster_assign(cuda.Out(gpu_assignments), cuda.Out(gpu_distances),
                 gpu_data, gpu_clusters,
                 block = (block_size_assign, 1, 1),
                 grid = (grid_size_assign,1))
        pycuda.autoinit.context.synchronize()
        t2 = time.time()
        assign_time += t2-t1
        
        t1 = time.time()
        #print "cluster_calc blocksize", block_size_calc_x, block_size_calc_y, 1
        #print "cluster_calc gridsizze", grid_size_calc_x, 1
        cluster_calc(cuda.In(gpu_assignments), gpu_data, gpu_clusters,
                        block = (block_size_calc_x, block_size_calc_y, 1),
                        grid = (grid_size_calc_x, 1));
        pycuda.autoinit.context.synchronize()
        t2 = time.time()
        calc_time += t2-t1

    if return_times:
        return gpu_clusters, gpu_assignments, data_time, module_time, assign_time/iterations, calc_time/iterations
    else:
        return gpu_clusters, gpu_assignments



#------------------------------------------------------------------------------------
#                                   source modules
#------------------------------------------------------------------------------------

def get_assign_module(nDim, nPts, nClusters, block_size_assign):

    return SourceModule("""

#define NCLUSTERS      """ + str(nClusters)                    + """
#define NPTS           """ + str(nPts)                         + """
#define NDIM           """ + str(nDim)                         + """
#define DATA_SIZE      """ + str(nPts * nDim)                  + """
#define CLUSTERS_SIZE  """ + str(nClusters*nDim)               + """
#define CLUSTER_CHUNKS """ + str(1 + (nClusters*nDim-1)/block_size_assign)  + """
#define THREADS        """ + str(block_size_assign)            + """

// calculate the distance from a data point to a cluster
__device__ float dc_dist(float *data, float *cluster)
{
    float dist = (data[0]-cluster[0]) * (data[0]-cluster[0]);
    for (int i=1; i<NDIM; i++) {
        float diff = data[i*NPTS] - cluster[i*NCLUSTERS];
        dist += diff*diff;
    }
    return sqrt(dist);
}


// **TODO**  need to loop through clusters if all of them don't fit into shared memory

// Assign data points to the nearest cluster
__global__ void cluster_assign(int *assignments, float *distances, float *data, float *clusters)
{
    // copy cluster to shared memory
    __shared__ float s_clusters[CLUSTERS_SIZE];
    int idx = threadIdx.x;
    for(int c = 0; c < CLUSTER_CHUNKS; c++, idx += THREADS){
        if(idx < CLUSTERS_SIZE){
            s_clusters[idx] = clusters[idx];
        }
    }
    __syncthreads();

    // calculate distance to each cluster
    idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= NPTS) return;
    float min_dist = 1e10;
    int closest = -1;
    for(int c=0; c<NCLUSTERS; c++){
        float d = dc_dist(data + idx, s_clusters + c);
        distances[idx * NCLUSTERS + c] = d;
        if(d < min_dist){
            min_dist = d;
            closest = c;
        }
    }
    assignments[idx] = closest;
}

""")



def get_calc_module(nDim, nPts, nClusters, block_size_calc_x, block_size_calc_y, data_chunk_size, data_reps):
    return SourceModule("""

#define NCLUSTERS       """ + str(nClusters)                            + """
#define NPTS            """ + str(nPts)                                 + """
#define NDIM            """ + str(nDim)                                 + """
#define DATA_SIZE       """ + str(nPts * nDim)                          + """
#define DATA_CHUNK_SIZE """ + str(data_chunk_size)                      + """
#define DATA_REPS       """ + str(data_reps)                            + """
#define CLUSTERS_SIZE   """ + str(nClusters*nDim)                       + """
#define CLUSTER_CHUNKS  """ + str(1 + (nClusters-1)/block_size_calc_x)  + """
#define THREADS         """ + str(min(block_size_calc_x,nClusters))     + """
#define DIMS            """ + str(block_size_calc_y)                    + """


// Calculate the new cluster centers
__global__ void cluster_calc(int *assignments, float *data, float *new_clusters)
{
    int idx = threadIdx.x;
    int cluster = threadIdx.x + blockDim.x*blockIdx.x;
    if(cluster >= NCLUSTERS) return;

    int idy = threadIdx.y;
    
    // allocate cluster_accum, cluster_count, and initialize to zero
    __shared__ float s_cluster_accum[NDIM * THREADS];
    __shared__ unsigned int s_cluster_count[THREADS];
    if(idy == 0) s_cluster_count[idx] = 0;
    for(int d = 0; d < NDIM; d+=DIMS){
        int dim = d + idy;
        if(dim >= NDIM) continue;
        s_cluster_accum[dim*THREADS + idx] = 0.0f;
    }
    __syncthreads();
    
/*
    // allocate a staging area for the data
    __shared__ float s_data[DIMS * DATA_CHUNK_SIZE];
    __syncthreads();
    

    // loop over all data points and update cluster_accum and cluster_count
    // for the cluster each point is assigned to
    
    // for each data rep
    for(int rep = 0; rep < DATA_REPS; rep++){
        int iData = rep * DATA_CHUNK_SIZE;    // index of the first data item in this chunk
        
        // copy the chunk into shared memory
        for(int add = 0; add < DATA_CHUNK_SIZE; add += THREADS){
            int iChunk = add + idx;     // index into the chunk in shared memory
            if(iChunk < DATA_CHUNK_SIZE && (iData + iChunk) < NPTS)
                s_data[idy*DATA_CHUNK_SIZE + iChunk] = data[dim*NPTS + iData + iChunk];
        }
        __syncthreads();
        
        // loop over data in shared memory and update cluster_count and cluster_accum
        for(int i=0; i<DATA_CHUNK_SIZE; i++){
            if((iData + i) >= NPTS) break;
            if(cluster == assignments[iData + i]){
                if(idy == 0) s_cluster_count[idx] += 1;
                s_cluster_accum[idy*THREADS + idx] += s_data[idy*DATA_CHUNK_SIZE + i];
            }
        }
        __syncthreads();
    }
*/

    // loop over all data and update cluster_count and cluster_accum
    for(int d = 0; d < NDIM; d += DIMS){
        int dim = d + idy;
        if(dim >=NDIM) continue;
        for(int i=0; i<NPTS; i++){
            if(i >= NPTS) break;
            if(cluster == assignments[i]){
                if(dim == 0) s_cluster_count[idx] += 1;
                s_cluster_accum[dim * THREADS + idx] += data[dim * NPTS + i];
            }
        }
    }
    __syncthreads();
    
    
    // divide the accum by the number of points and copy to the output area
    for(int d = 0; d < NDIM; d += DIMS){
        int dim = d + idy;
        if(dim >=NDIM) continue;
        if(s_cluster_count[idx] > 0){
            new_clusters[dim * NCLUSTERS + cluster] = s_cluster_accum[dim * THREADS + idx]
                                                    / s_cluster_count[idx];
        }
    }
}

""")



#--------------------------------------------------------------------------------------------
#                           testing functions
#--------------------------------------------------------------------------------------------
    
def run_tests(nTests, nPts, nDim, nClusters, nReps, verbose = VERBOSE, print_times = PRINT_TIMES):
    # run_tests(nTests, nPts, nDim, nClusters, nReps [, verbose [, print_times]]
    
    # Generate nPts random data elements with nDim dimensions and nCluster random clusters,
    # then run kmeans for nReps and compare gpu and cpu results.  This is repeated nTests times
    nErrors = 0
    nCalcErrors = 0
    cpu_time = 0.
    gpu_time = 0.
    
    gpu_data_time = 0.
    gpu_module_time = 0.
    gpu_assign_time = 0.
    gpu_calc_time = 0.

    np.random.seed(100);
    #data = np.random.rand(nDim, nPts).astype(np.float32)
    #clusters = np.random.rand(nDim, nClusters).astype(np.float32)

    data = np.array([[3., 4., 4., 9., 5., 6., 9., 5., 5., 7., 6.], \
                     [3., 3., 2., 2., 1., 2., 4., 2., 4., 4., 5.]]).astype(np.float32)
    clusters = np.array([[4.57142878, 7.75], \
                         [2.42857146, 3.75]]).astype(np.float32)
                         
    print "nPts =", nPts
    print "nDim =", nDim
    print "nClusters =", nClusters
    print "nReps =", nReps

    if verbose:
        print "data"
        print data
        print "\nclusters"
        print clusters
        
    for i in range(nTests):
        t1 = time.time()
        (cpu_clusters, cpu_assign) = kmeans_cpu(data, clusters, nReps)
        print cpu_assign.shape
        cpu_assign.shape = (nPts,)
        t2 = time.time()
        cpu_time += t2-t1
        if verbose:
            print "cpu assignments"
            print cpu_assign
            print "cpu clusters"
            print cpu_clusters
            print "cpu time = ", t2-t1
            
        t1 = time.time()
        (gpu_clusters, gpu_assign, data_time, module_time, assign_time, calc_time) = kmeans_gpu(data, clusters, nReps, 1)
        t2 = time.time()
        
        gpu_time += t2-t1
        gpu_data_time += data_time
        gpu_module_time += module_time
        gpu_assign_time += assign_time
        gpu_calc_time += calc_time
        
        if verbose:
            print "gpu assignments"
            print gpu_assign
            print "gpu clusters"
            print gpu_clusters
            print "gpu time = ", t2-t1
    
        # calculate the number of data points in each cluster
        c = np.arange(nClusters)
        c_counts = np.sum(cpu_assign.reshape(nPts,1) == c, axis=0)
        
        # verify results
        differences = sum(gpu_assign != cpu_assign)
        if(differences > 0):
            nErrors += 1
            if verbose:
                print "Test",i,"*** ERROR ***", differences, "differences"
                iDiff = np.arange(nPts)[gpu_assign != cpu_assign]
                print "iDiff", iDiff
                for ii in iDiff:
                    print "data point is", data[:,ii]
                    print "cpu assigned to", cpu_assign[ii]
                    print "   with center at (cpu)", cpu_clusters[:,cpu_assign[ii]]
                    print "   with center at (gpu)", gpu_clusters.get()[:,cpu_assign[ii]]
                    print "gpu assigned to", gpu_assign[ii]
                    print "   with center at (cpu)", cpu_clusters[:,gpu_assign[ii]]
                    print "   with center at (gpu)", gpu_clusters.get()[:, gpu_assign[ii]]
        else:
            if verbose:
                print "Cluster assignment OK"

        diff = np.max(np.abs(gpu_clusters.get() - cpu_clusters))

        if verbose:
            print "max error in cluster centers is", diff
            print "avg error in cluster centers is", 
            print np.mean(np.abs(gpu_clusters.get()-cpu_clusters))

        if diff > 1e-7 * max(c_counts) or math.isnan(diff):
            nCalcErrors += 1
            if verbose:
                print "Test",i,"*** ERROR *** max diff was", diff
                print 
        else:
            if verbose:
                print "Test", i, "OK"

    if print_times:
        print "\n---------------------------------------------"
        print "nPts      =", nPts
        print "nDim      =", nDim
        print "nClusters =", nClusters
        print "nReps     =", nReps
        print "Assignment errors  =", nErrors, "out of", nTests, "tests"
        print "Calculation errors =", nCalcErrors, "out of", nTests, "tests"
        print "average cpu time (ms) =", cpu_time/nTests*1000.
        print "average gpu time (ms) =", gpu_time/nTests*1000.
        print "       data time (ms) =", gpu_data_time/nTests*1000.
        print "     module time (ms) =", gpu_module_time/nTests*1000.
        print "     assign time (ms) =", gpu_assign_time/nTests*1000.        
        print "       calc time (ms) =", gpu_calc_time/nTests*1000.        
        print "---------------------------------------------"

    return nErrors + nCalcErrors


#----------------------------------------------------------------------------------------
#                           multi-tests
#----------------------------------------------------------------------------------------

def quiet_run(nTests, nPts, nDim, nClusters, nReps, ptimes = PRINT_TIMES):
    # quiet_run(nTests, nPts, nDim, nClusters, nReps [, ptimes]):
    print "[TEST]({0:3},{1:8},{2:5},{3:5}, {4:5})...".format(nTests, nPts, nDim, nClusters, nReps),
    try:
        if run_tests(nTests, nPts, nDim, nClusters, nReps, verbose = 0, print_times = ptimes) == 0:
            print "OK"
        else:
            print "*** ERROR ***"
    except cuda.LaunchError:
        print "launch error"
    
def quiet_runs(nTest_list, nPts_list, nDim_list, nClusters_list, nRep_list, print_it = PRINT_TIMES):
    # quiet_runs(nTest_list, nPts_list, nDim_list, nClusters_list [, print_it]):
    for t in nTest_list:
        for pts in nPts_list:
            for dim in nDim_list:
                for clst in nClusters_list:
                    for rep in nRep_list:
                        quiet_run(t, pts, dim, clst, rep, ptimes = print_it);

def run_all(pFlag = 1):
    quiet_run(10, 10, 4, 3, 1, ptimes = pFlag)
    quiet_run(10, 1000, 60, 20, 1, ptimes = pFlag)
    quiet_run(1, 100000, 60, 20, 1, ptimes = pFlag)
    quiet_run(1, 10000, 600, 5, 1, ptimes = pFlag)
    quiet_run(1, 10000, 5, 600, 1, ptimes = pFlag)
    quiet_run(10, 100, 5, 600, 1, ptimes = pFlag)
    quiet_run(10, 100, 600, 5, 1, ptimes = pFlag)
    quiet_run(10, 10, 20, 30, 1, ptimes = pFlag)
    quiet_run(10, 10, 4, 3, 10, ptimes = pFlag)
    quiet_run(10, 1000, 60, 20, 10, ptimes = pFlag)
    quiet_run(1, 10000, 60, 20, 10, ptimes = pFlag)
    quiet_run(1, 1000, 600, 5, 10, ptimes = pFlag)
    quiet_run(1, 1000, 5, 600, 10, ptimes = pFlag)
    quiet_run(10, 100, 5, 600, 10, ptimes = pFlag)
    quiet_run(10, 100, 600, 5, 10, ptimes = pFlag)
    quiet_run(10, 10, 20, 30, 10, ptimes = pFlag)
    
def timings():
    quiet_runs([1], [10, 100, 1000, 10000, 100000], [2, 16], [4, 16], [1, 5], print_it = 0)
