import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pycuda.reduction import ReductionKernel
from cpu_kmeans import kmeans_cpu
from cpu_kmeans import assign_cpu
from cpu_kmeans import calc_cpu

import numpy as np
import math
import time

VERBOSE = 0
PRINT_TIMES = 0



#------------------------------------------------------------------------------------
#                kmeans using triangle inequality algorithm on the gpu
#------------------------------------------------------------------------------------

def trikmeans_gpu(data, clusters, iterations, return_times = 0):
    # trikmeans_gpu(data, clusters, iterations) returns (clusters, labels)
    
    # kmeans using triangle inequality algorithm and cuda
    # input arguments are the data, intial cluster values, and number of iterations to repeat
    # The shape of data is (nDim, nPts) where nDim = # of dimensions in the data and
    # nPts = number of data points
    # The shape of clustrs is (nDim, nClusters) 
    #
    # The return values are the updated clusters and labels for the data
    
    
    (nDim, nPts) = data.shape
    nClusters = clusters.shape[1]
    
    data = np.array(data).astype(np.float32)
    clusters = np.array(clusters).astype(np.float32)
    
    # copy the data and clusters to the GPU, and allocate room for other arrays
    t1 = time.time()
    gpu_data = gpuarray.to_gpu(data)
    gpu_clusters = gpuarray.to_gpu(clusters)
    gpu_assignments = gpuarray.zeros((nPts,), np.int32)         # cluster assignment
    gpu_lower = gpuarray.zeros((nClusters, nPts), np.float32)   # lower bounds on distance between 
                                                                # point and each cluster
    gpu_upper = gpuarray.zeros((nPts,), np.float32)             # upper bounds on distance between
                                                                # point and any cluster
    gpu_ccdist = gpuarray.zeros((nClusters, nClusters), np.float32)    # cluster-cluster distances
    gpu_hdClosest = gpuarray.zeros((nClusters,), np.float32)    # half distance to closest
    gpu_hdClosest.fill(1.0e10)  # set to large value // **TODO**  get the acutal float max
    gpu_badUpper = gpuarray.zeros((nPts,), np.int32)   # flag to indicate upper bound needs recalc
    #gpu_badUpper.fill(1)
    gpu_clusters2 = gpuarray.zeros((nDim, nClusters), np.float32);
    gpu_cluster_movement = gpuarray.zeros((nClusters,), np.float32);
    pycuda.autoinit.context.synchronize()
    t2 = time.time()
    data_time = t2-t1
    
    
    # block and grid sizes for the ccdist kernel (also for hdclosest)
    blocksize_ccdist = min(512, 16*(1+(nClusters-1)/16))
    gridsize_ccdist = 1 + (nClusters-1)/blocksize_ccdist
    
    #block and grid sizes for the init module
    threads_desired = 16*(1+(max(nPts, nDim*nClusters)-1)/16)
    blocksize_init = min(512, threads_desired) 
    gridsize_init = 1 + (threads_desired - 1)/blocksize_init
    
    #block and grid sizes for the step3 module
    blocksize_step3 = blocksize_init
    gridsize_step3 = gridsize_init
    
    #block and grid sizes for the step4 module
    for blocksize_step4_x in range(32, 512, 32):
        if blocksize_step4_x >= nClusters:
            break;
    blocksize_step4_y = min(nDim, 512/blocksize_step4_x)
    gridsize_step4_x = 1 + (nClusters-1)/blocksize_step4_x
    
    #block and grid sizes for the step56 module
    blocksize_step56 = blocksize_init
    gridsize_step56 = gridsize_init
    
    # get the functions from the source modules
    t1 = time.time()
    mod_ccdist = get_ccdist_module(nDim, nPts, nClusters, blocksize_ccdist)
    mod_hdclosest = get_hdclosest_module(nClusters)
    mod_init = get_init_module(nDim, nPts, nClusters, blocksize_init)
    mod_step3 = get_step3_module(nDim, nPts, nClusters, blocksize_step3)
    mod_step4 = get_step4_module(nDim, nPts, nClusters, blocksize_step4_x, blocksize_step4_y)
    mod_step56 = get_step56_module(nDim, nPts, nClusters, blocksize_step56)
    ccdist = mod_ccdist.get_function("ccdist")
    calc_hdclosest = mod_hdclosest.get_function("calc_hdclosest")
    init = mod_init.get_function("init")
    step3 = mod_step3.get_function("step3")
    step4 = mod_step4.get_function("step4")
    step56 = mod_step56.get_function("step56")
    pycuda.autoinit.context.synchronize()
    t2 = time.time()
    module_time = t2-t1
    
    ccdist_time = 0.
    hdclosest_time = 0.
    init_time = 0.
    step3_time = 0.
    step4_time = 0.
    step56_time = 0.


    t1 = time.time()
    ccdist(gpu_clusters, gpu_ccdist, gpu_hdClosest,
             block = (blocksize_ccdist, 1, 1),
             grid = (gridsize_ccdist, 1))
    pycuda.autoinit.context.synchronize()
    t2 = time.time()
    ccdist_time += t2-t1
    
    t1 = time.time()
    calc_hdclosest(gpu_ccdist, gpu_hdClosest,
            block = (blocksize_ccdist, 1, 1),
            grid = (gridsize_ccdist, 1))
    pycuda.autoinit.context.synchronize()
    t2 = time.time()
    hdclosest_time += t2-t1
    
    t1 = time.time()
    init(gpu_data, gpu_clusters, gpu_ccdist, gpu_hdClosest, gpu_assignments, 
            gpu_lower, gpu_upper,
            block = (blocksize_init, 1, 1),
            grid = (gridsize_init, 1))
    pycuda.autoinit.context.synchronize()
    t2 = time.time()
    init_time += t2-t1

    for i in range(iterations):
    
        if i>0:
            t1 = time.time()
            ccdist(gpu_clusters, gpu_ccdist, gpu_hdClosest,
                     block = (blocksize_ccdist, 1, 1),
                     grid = (gridsize_ccdist, 1))
            pycuda.autoinit.context.synchronize()
            t2 = time.time()
            ccdist_time += t2-t1
            
            t1 = time.time()
            calc_hdclosest(gpu_ccdist, gpu_hdClosest,
                    block = (blocksize_ccdist, 1, 1),
                    grid = (gridsize_ccdist, 1))
            pycuda.autoinit.context.synchronize()
            t2 = time.time()
            hdclosest_time += t2-t1
            
        """
        print "Just before step 3=========================================="
        print "gpu_clusters"
        print gpu_clusters
        print "gpu_ccdist"
        print gpu_ccdist
        print "gpu_hdClosest"
        print gpu_hdClosest
        print "gpu_assignments"
        print gpu_assignments
        print "gpu_lower"
        print gpu_lower
        print "gpu_upper"
        print gpu_upper
        print "gpu_badUpper"
        print gpu_badUpper
        """
        
        t1 = time.time()
        step3(gpu_data, gpu_clusters, gpu_ccdist, gpu_hdClosest, gpu_assignments,
                gpu_lower, gpu_upper, gpu_badUpper,
                block = (blocksize_step3, 1, 1),
                grid = (gridsize_step3, 1))
        pycuda.autoinit.context.synchronize()
        t2 = time.time()
        step3_time += t2-t1
        
        """
        print "Just before step 4=========================================="
        print "gpu_assignments"
        print gpu_assignments
        print "gpu_lower"
        print gpu_lower
        print "gpu_upper"
        print gpu_upper
        print "gpu_badUpper"
        print gpu_badUpper
        """        
    
        t1 = time.time()
        step4(gpu_data, gpu_clusters, gpu_clusters2, gpu_assignments, gpu_cluster_movement,
                block = (blocksize_step4_x, blocksize_step4_y, 1),
                grid = (gridsize_step4_x, 1))
        pycuda.autoinit.context.synchronize()
        t2 = time.time()
        step4_time += t2-t1
        
        """
        print "Just before step 5=========================================="
        print "gpu_cluste_movement"
        print gpu_cluster_movement
        print "gpu_clusters"
        print gpu_clusters2
        """
    
        t1 = time.time()
        step56(gpu_data, gpu_assignments, gpu_lower, gpu_upper, gpu_cluster_movement, gpu_badUpper,
                block = (blocksize_step56, 1, 1),
                grid = (gridsize_step56, 1))
        pycuda.autoinit.context.synchronize()
        t2 = time.time()
        step56_time += t2-t1
        
        """
        print "Just after step 6=========================================="
        print "gpu_lower"
        print gpu_lower
        print "gpu_upper"
        print gpu_upper
        print "gpu_badUpper"
        print gpu_badUpper
        """
        
        # prepare for next iteration
        temp = gpu_clusters
        gpu_clusters = gpu_clusters2
        gpu_clusters2 = temp
        
    if return_times:
        return gpu_ccdist, gpu_hdClosest, gpu_assignments, gpu_lower, gpu_upper, \
                gpu_clusters.get(), gpu_cluster_movement, \
                data_time, module_time, init_time, \
                ccdist_time/iterations, hdclosest_time/iterations, \
                step3_time/iterations, step4_time/iterations, step56_time/iterations
    else:
        return gpu_clusters.get(), gpu_assignments.get()



#------------------------------------------------------------------------------------
#                                   source modules
#------------------------------------------------------------------------------------

def get_ccdist_module(nDim, nPts, nClusters, blocksize_ccdist):
    # module to calculate distances between each cluster and half distance to closest
    
    return SourceModule("""

#define NCLUSTERS      """ + str(nClusters)                    + """
#define NDIM           """ + str(nDim)                         + """
#define CLUSTERS_SIZE  """ + str(nClusters*nDim)               + """
#define CLUSTER_CHUNKS """ + str(1 + (nClusters*nDim-1)/blocksize_ccdist) + """
#define THREADS        """ + str(blocksize_ccdist) + """

// calculate the distance beteen two clusters
__device__ float calc_dist(float *clusterA, float *clusterB)
{
    float dist = (clusterA[0]-clusterB[0]) * (clusterA[0]-clusterB[0]);
    for (int i=1; i<NDIM; i++) {
        float diff = clusterA[i*NCLUSTERS] - clusterB[i*NCLUSTERS];
        dist += diff*diff;
    }
    return sqrt(dist);
}


// **TODO**  need to loop through clusters if all of them don't fit into shared memory

// Calculate cluster - cluster distances
__global__ void ccdist(float *clusters, float *cc_dists, float *hdClosest)
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

    // calculate distance between this cluster and all lower clusters
    // then store the distance in the table in two places: (this, lower) and (lower, this)
    idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= NCLUSTERS) return;
    for(int c=0; c<idx; c++){
        float d = 0.5f * calc_dist(s_clusters+c, s_clusters + idx); // store 1/2 distance
        cc_dists[c * NCLUSTERS + idx] = d;
        cc_dists[idx * NCLUSTERS + c] = d;
    }
}

""")


def get_hdclosest_module(nClusters):
    # module to calculate half distance to closest
    
    return SourceModule("""

#define NCLUSTERS      """ + str(nClusters)                    + """

// **TODO**  convert this kernel to a reduction to speed up

// Finish the determination of hdClosest (half
__global__ void calc_hdclosest(float *cc_dists, float *hdClosest)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for(int c=0; c<NCLUSTERS; c++){
        if(c == idx) continue;
        float d = cc_dists[c*NCLUSTERS + idx];      // cc_dists contains 1/2 distance
        if(d < hdClosest[idx]) hdClosest[idx] = d;
    }
}

""")


def get_init_module(nDim, nPts, nClusters, block_size_assign):
    # initial assignment of points to closest cluster
    
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
__global__ void init(float *data, float *clusters, 
                     float *ccdist, float *hdClosest, int *assignments, 
                     float *lower, float *upper)
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
    
    // start with cluster 0 as the closest
    float min_dist = dc_dist(data+idx, s_clusters);
    lower[idx] = min_dist;
    int closest = 0;
    
    for(int c=1; c<NCLUSTERS; c++){
    // **TODO**  see if this test to skip some calculations is really worth it on the gpu versus cpu
        if(min_dist + 0.000001f <= ccdist[closest * NCLUSTERS + c]) continue;
        float d = dc_dist(data + idx, s_clusters + c);
        lower[c*NPTS + idx] = d;
        if(d < min_dist){
            min_dist = d;
            closest = c;
        }
    }
    assignments[idx] = closest;
    upper[idx] = min_dist;
}

""")


def get_step3_module(nDim, nPts, nClusters, blocksize_step3):
    # determine nearest cluster to each point, and update distances between points and their
    # assigned cluster
    
    # threads are one-dimensional and cover the data points.

    return SourceModule("""

#define NCLUSTERS      """ + str(nClusters)                    + """
#define NPTS           """ + str(nPts)                         + """
#define NDIM           """ + str(nDim)                         + """
#define DATA_SIZE      """ + str(nPts * nDim)                  + """
#define CLUSTERS_SIZE  """ + str(nClusters*nDim)               + """
#define CLUSTER_CHUNKS """ + str(1 + (nClusters*nDim-1)/blocksize_step3)  + """
#define THREADS        """ + str(blocksize_step3)            + """

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

// Step 3 of the algorithm
__global__ void step3(float *data, float *clusters, 
                     float *ccdist, float *hdClosest, int *assignments, 
                     float *lower, float *upper, int *badUpper)
{
    // copy clusters to shared memory
    __shared__ float s_clusters[CLUSTERS_SIZE];
    int idx = threadIdx.x;
    for(int c = 0; c < CLUSTER_CHUNKS; c++, idx += THREADS){
        if(idx < CLUSTERS_SIZE){
            s_clusters[idx] = clusters[idx];
        }
    }
    __syncthreads();
    
    // idx ranges over the data points
    idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= NPTS) return;
    
    float ux = upper[idx];
    int cx = assignments[idx];
    float rx = badUpper[idx];

    if(ux <= hdClosest[cx]) return; // step 2 condition

    for(int c=0; c<NCLUSTERS; c++){
        // step 3 conditions...
        if(c == cx || ux <= lower[c*NPTS + idx] || ux <= ccdist[cx*NCLUSTERS + c])
             continue;
             
        // Step 3a: check if upper bound needs to be recalculated
        float d_x_cx;
        if(rx){
            // distance between point idx and its currently assigned center needs to be calculated
            d_x_cx = dc_dist(data+idx, s_clusters+cx);
            ux = d_x_cx;
            lower[c*NPTS + idx] = d_x_cx;
            rx = 0;
        }else{
            d_x_cx = ux;
        }
        
        // Step 3b: compute distance between x and c change x's assignment if necessary
        if(d_x_cx > lower[c*NPTS + idx] || d_x_cx > ccdist[cx*NCLUSTERS + c]){
            float d_x_c = dc_dist(data+idx, s_clusters+c);
            lower[c*NPTS + idx] = d_x_c;
            if(d_x_c < d_x_cx){
                // assign x to c
                ux = d_x_c;
                cx = c;
                rx = 0;
                // **TODO**  flag the clusters that have changed which is needed for later steps
            }
        }
    }
    upper[idx] = ux;
    assignments[idx] = cx;
    badUpper[idx] = rx;
}

""")


def get_step4_module(nDim, nPts, nClusters, blocksize_step4_x, blocksize_step4_y):
    # calculate new cluster values based on points assigned
    
    return SourceModule("""

#define NCLUSTERS       """ + str(nClusters)                            + """
#define NPTS            """ + str(nPts)                                 + """
#define NDIM            """ + str(nDim)                                 + """
#define THREADS         """ + str(min(blocksize_step4_x,nClusters))     + """
#define DIMS            """ + str(blocksize_step4_y)                    + """

// calculate the distance beteen two clusters
__device__ float calc_dist(float *clusterA, float *clusterB)
{
    float dist = (clusterA[0]-clusterB[0]) * (clusterA[0]-clusterB[0]);
    for (int i=1; i<NDIM; i++) {
        float diff = clusterA[i*NCLUSTERS] - clusterB[i*NCLUSTERS];
        dist += diff*diff;
    }
    return sqrt(dist);
}

// Calculate the new cluster centers and also the distance between old center and new one
__global__ void step4(float *data, float *clusters, float *new_clusters, int *assignments,
    float *cluster_movement)
{
    int idx = threadIdx.x;
    int cluster = threadIdx.x + blockDim.x*blockIdx.x;
    if(cluster >= NCLUSTERS) return;

    int idy = threadIdx.y;
    
    // allocate cluster_accum, cluster_count, and initialize to zero
    // also initialize the cluster_movement array to zero
    __shared__ float s_cluster_accum[NDIM * THREADS];
    __shared__ unsigned int s_cluster_count[THREADS];
    if(idy == 0){
        s_cluster_count[idx] = 0;
        cluster_movement[idx] = 0.f;
    }
    for(int d = 0; d < NDIM; d+=DIMS){
        int dim = d + idy;
        if(dim >= NDIM) continue;
        s_cluster_accum[dim*THREADS + idx] = 0.0f;
    }
    __syncthreads();
    

    // loop over all data and update cluster_count and cluster_accum
    for(int dim = idy; dim < NDIM; dim += DIMS){
        int dim1 = dim * THREADS + idx;
        int dim2 = dim * NPTS;
//        int dim = d + idy;
//        if(dim >=NDIM) continue;
        for(int i=0; i<NPTS; i++, dim2++){
            if(i >= NPTS) break;
            if(cluster == assignments[i]){
                if(dim == 0) s_cluster_count[idx] += 1;
                s_cluster_accum[dim1] += data[dim2];
            }
        }
    }
    __syncthreads();
    
    // divide the accum by the number of points and copy to the output area
    for(int d = 0; d < NDIM; d += DIMS){
        int dim = d + idy;
        if(dim >=NDIM) continue;
        int index1 = dim * NCLUSTERS + cluster;
        if(s_cluster_count[idx] > 0){
            new_clusters[index1] = s_cluster_accum[dim * THREADS + idx]
                                                    / s_cluster_count[idx];
        }else{
            new_clusters[index1] = clusters[index1];
        }
    }
    
    // calculate the distance between old and new clusters
    cluster_movement[cluster] = calc_dist(new_clusters + cluster, clusters + cluster);
}

""")


def get_step56_module(nDim, nPts, nClusters, blocksize_step56):
    # initial assignment of points to closest cluster
    
    return SourceModule("""

#define NCLUSTERS      """ + str(nClusters)                    + """
#define NPTS           """ + str(nPts)                         + """
#define NDIM           """ + str(nDim)                         + """
#define CLUSTER_CHUNKS """ + str(1 + (nClusters-1)/blocksize_step56)  + """
#define THREADS        """ + str(blocksize_step56)            + """

// **TODO**  need to loop through clusters if all of them don't fit into shared memory

// Assign data points to the nearest cluster
__global__ void step56(float *data, int *assignment, float *lower, float * upper, 
                        float *cluster_movement, int *badUpper)
{
    // copy cluster movement to shared memory
    __shared__ float s_cluster_movement[NCLUSTERS];
//    __shared__ int s_cluster_movement_flag[NCLUSTERS];    // CHANGE#2
    int idx = threadIdx.x;
    for(int c = 0; c < CLUSTER_CHUNKS; c++, idx += THREADS){
        if(idx < NCLUSTERS){
            s_cluster_movement[idx] = cluster_movement[idx];
//            if(s_cluster_movement[idx] > 0.f)
//                s_cluster_movement_flag[idx] = 1;
        }
    }
    __syncthreads();

    idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= NPTS) return;
    
    // loop through all clusters and update the lower bound
    for(int c=0; c < NCLUSTERS; c++){
        if(s_cluster_movement[c] > 0.0f){
//        if(s_cluster_movement_flag[c]){
            if(s_cluster_movement[c] < lower[c * NPTS + idx]){
                lower[c*NPTS + idx] -= s_cluster_movement[c];
            }else{
                lower[c*NPTS + idx] = 0.0f;
            }
        }
    }
    
//  CHANGE#1
    // update the upper bound for this data point
    if(s_cluster_movement[assignment[idx]] > 0.f){
        upper[idx] += s_cluster_movement[assignment[idx]];
        badUpper[idx] = 1;
    }
//*/
/*
    upper[idx] += s_cluster_movement[assignment[idx]];
    
    // reset the badUpper flag
    badUpper[idx] = 1;
*/
}

""")





#--------------------------------------------------------------------------------------------
#                           testing functions
#--------------------------------------------------------------------------------------------
    
def run_tests1(nTests, nPts, nDim, nClusters, nReps=1, verbose = VERBOSE, print_times = PRINT_TIMES):
    # run_tests(nTests, nPts, nDim, nClusters, nReps [, verbose [, print_times]]
    
    if nReps > 1:
        print "This method only runs test for nReps == 1"
        return 1
        
    # Generate nPts random data elements with nDim dimensions and nCluster random clusters,
    # then run kmeans for nReps and compare gpu and cpu results.  This is repeated nTests times
    cpu_time = 0.
    gpu_time = 0.
    
    gpu_data_time = 0.
    gpu_module_time = 0.
    gpu_ccdist_time = 0.
    gpu_hdclosest_time = 0.
    gpu_init_time = 0.
    gpu_step3_time = 0.
    gpu_step4_time = 0.
    gpu_step56_time = 0.

    np.random.seed(100)
    data = np.random.rand(nDim, nPts).astype(np.float32)
    clusters = np.random.rand(nDim, nClusters).astype(np.float32)

    if verbose:
        print "data"
        print data
        print "\nclusters"
        print clusters

    nErrors = 0

    # repeat this test nTests times
    for iTest in range(nTests):
    
        #run the gpu algorithm
        t1 = time.time()
        (gpu_ccdist, gpu_hdClosest, gpu_assignments, gpu_lower, gpu_upper, \
            gpu_clusters2, gpu_cluster_movement, \
            data_time, module_time, init_time, ccdist_time, hdclosest_time, \
            step3_time, step4_time, step56_time) = \
            trikmeans_gpu(data, clusters, nReps, 1)
        t2 = time.time()        
        gpu_time += t2-t1
        gpu_data_time += data_time
        gpu_module_time += module_time
        gpu_ccdist_time += ccdist_time
        gpu_hdclosest_time += hdclosest_time
        gpu_init_time += init_time
        gpu_step3_time += step3_time
        gpu_step4_time += step4_time
        gpu_step56_time += step56_time
        
        if verbose:
            print "------------------------ gpu results ------------------------"
            print "cluster-cluster distances"
            print gpu_ccdist
            print "half distance to closest"
            print gpu_hdClosest
            print "gpu time = ", t2-t1
            print "gpu_assignments"
            print gpu_assignments
            print "gpu_lower"
            print gpu_lower
            print "gpu_upper"
            print gpu_upper
            print "gpu_clusters2"
            print gpu_clusters2
            print "-------------------------------------------------------------"
            

        # check ccdist and hdClosest
        ccdist = np.array(gpu_ccdist.get())
        hdClosest = np.array(gpu_hdClosest.get())
        
        t1 = time.time()
        cpu_ccdist = 0.5 * np.sqrt(((clusters[:,:,np.newaxis]-clusters[:,np.newaxis,:])**2).sum(0))
        t2 = time.time()
        cpu_ccdist_time = t2-t1
        
        if verbose:
            print "cpu_ccdist"
            print cpu_ccdist
        
        error = np.abs(cpu_ccdist - ccdist)
        if np.max(error) > 1e-7 * nDim * 2:
            print "iteration", iTest,
            print "***ERROR*** max ccdist error =", np.max(error)
            nErrors += 1
        if verbose:
            print "average ccdist error =", np.mean(error)
            print "max ccdist error     =", np.max(error)
        
        t1 = time.time()
        cpu_ccdist[cpu_ccdist == 0.] = 1e10
        good_hdClosest = np.min(cpu_ccdist, 0)
        t2 = time.time()
        cpu_hdclosest_time = t2-t1
        
        if verbose:
            print "good_hdClosest"
            print good_hdClosest
        err = np.abs(good_hdClosest - hdClosest)
        if np.max(err) > 1e-7 * nDim:
            print "***ERROR*** max hdClosest error =", np.max(err)
            nErrors += 1
        if verbose:
            print "errors on hdClosest"
            print err
            print "max error on hdClosest =", np.max(err)
    
    
        # calculate cpu initial assignments
        t1 = time.time()
        cpu_assign = assign_cpu(data, clusters)
        t2 = time.time()
        cpu_assign_time = t2-t1
        
        if verbose:
            print "cpu assignments"
            print cpu_assign
            print "gpu assignments"
            print gpu_assignments
            print "gpu new clusters"
            print gpu_clusters2.get()
            
        differences = sum(gpu_assignments.get() - cpu_assign)
        if(differences > 0):
            nErrors += 1
            print differences, "errors in initial assignment"
        else:
            if verbose:
                print "initial cluster assignments match"
    
        # calculate the number of data points in each cluster
        c = np.arange(nClusters)
        c_counts = np.sum(cpu_assign.reshape(nPts,1) == c, axis=0)

        # calculate cpu new cluster values:
        t1 = time.time()
        cpu_new_clusters = calc_cpu(data, cpu_assign, clusters)
        t2 = time.time()
        cpu_calc_time = t2-t1
        
        if verbose:
            print "cpu new clusters"
            print cpu_new_clusters
        
        diff = np.max(np.abs(gpu_clusters2 - cpu_new_clusters))
        if diff > 1e-7 * max(c_counts) or math.isnan(diff):
            iDiff = np.arange(nClusters)[((gpu_clusters2 - cpu_new_clusters)**2).sum(0) > 1e-7]
            print "clusters that differ:"
            print iDiff
            nErrors += 1
            if verbose:
                print "Test",iTest,"*** ERROR *** max diff was", diff
                print 
        else:
            if verbose:
                print "Test", iTest, "OK"
        
        #check if the cluster movement values are correct
        cpu_cluster_movement = np.sqrt(((clusters - cpu_new_clusters)**2).sum(0))
        diff = np.max(np.abs(cpu_cluster_movement - gpu_cluster_movement.get()))
        if diff > 1e-7 * nDim:
            print "*** ERROR *** max cluster movement error =", diff
        if verbose:
            print "cpu cluster movements"
            print cpu_cluster_movement
            print "gpu cluster movements"
            print gpu_cluster_movement
            print "max diff in cluster movements is", diff
        
        cpu_time = cpu_assign_time + cpu_calc_time
    

    if print_times:
        print "\n---------------------------------------------"
        print "nPts      =", nPts
        print "nDim      =", nDim
        print "nClusters =", nClusters
        print "nReps     =", nReps
        print "average cpu time (ms) =", cpu_time/nTests*1000.
        print "     assign time (ms) =", cpu_assign_time/nTests*1000.
        if nReps == 1:
            print "       calc time (ms) =", cpu_calc_time/nTests*1000.
            print "average gpu time (ms) =", gpu_time/nTests*1000.
        else:
            print "       calc time (ms) ="
            print "average gpu time (ms) ="
        print "       data time (ms) =", gpu_data_time/nTests*1000.
        print "     module time (ms) =", gpu_module_time/nTests*1000.
        print "       init time (ms) =", gpu_init_time/nTests*1000.        
        print "     ccdist time (ms) =", gpu_ccdist_time/nTests*1000.        
        print "  hdclosest time (ms) =", gpu_hdclosest_time/nTests*1000.        
        print "      step3 time (ms) =", gpu_step3_time/nTests*1000.        
        print "      step4 time (ms) =", gpu_step4_time/nTests*1000.        
        print "     step56 time (ms) =", gpu_step56_time/nTests*1000.        
        print "---------------------------------------------"

    return nErrors


def verify_assignments(gpu_assign, cpu_assign, data, gpu_clusters, cpu_clusters, verbose = 0, iTest = -1): 
    # check that assignments are equal

    """
    print "verify_assignments"
    print "gpu_assign", gpu_assign, "is type", type(gpu_assign)
    print "gpu_assign", cpu_assign, "is type", type(cpu_assign)
    """
    differences = sum(gpu_assign != cpu_assign)
    # print "differences =", differences
    error = 0
    if(differences > 0):
        error = 1
        if verbose:
            if iTest >= 0:
                print "Test", iTest,
            print "*** ERROR ***", differences, "differences"
            iDiff = np.arange(gpu_assign.shape[0])[gpu_assign != cpu_assign]
            print "iDiff", iDiff
            for ii in iDiff:
                print "data point is", data[:,ii]
                print "cpu assigned to", cpu_assign[ii]
                print "   with center at (cpu)", cpu_clusters[:,cpu_assign[ii]]
                print "   with center at (gpu)", gpu_clusters[:,cpu_assign[ii]]
                print "gpu assigned to", gpu_assign[ii]
                print "   with center at (cpu)", cpu_clusters[:,gpu_assign[ii]]
                print "   with center at (gpu)", gpu_clusters[:, gpu_assign[ii]]
                print ""
                print "cpu calculated distances:"
                print "   from point", ii, "to:"
                print "      cluster", cpu_assign[ii], "is", np.sqrt(np.sum((data[:,ii]-cpu_clusters[:,cpu_assign[ii]])**2))
                print "      cluster", gpu_assign[ii], "is", np.sqrt(np.sum((data[:,ii]-cpu_clusters[:,gpu_assign[ii]])**2))
                print "gpu calculated distances:"
                print "   from point", ii, "to:"
                print "      cluster", cpu_assign[ii], "is", np.sqrt(np.sum((data[:,ii]-gpu_clusters[:,cpu_assign[ii]])**2))
                print "      cluster", gpu_assign[ii], "is", np.sqrt(np.sum((data[:,ii]-gpu_clusters[:,gpu_assign[ii]])**2))
    else:
        if verbose:
            if iTest >= 0:
                print "Test", iTest,
            print "Cluster assignment is OK"
    return error

def verify_clusters(gpu_clusters, cpu_clusters, cpu_assign, verbose = 0, iTest = -1):
    # check that clusters are equal
    error = 0
    
    # calculate the number of data points in each cluster
    nPts = cpu_assign.shape[0]
    nClusters = cpu_clusters.shape[1]
    c = np.arange(nClusters)
    c_counts = np.sum(cpu_assign.reshape(nPts,1) == c, axis=0)
    
    err = np.abs(gpu_clusters - cpu_clusters)
    diff = np.max(err)
    
    if verbose:
        print "max error in cluster centers is", diff
        print "avg error in cluster centers is", np.mean(err)
    
    allowable_diff = max(c_counts) * 1e-7
    if diff > allowable_diff or math.isnan(diff):
        error = 1
        iDiff = np.arange(nClusters)[((gpu_clusters - cpu_clusters)**2).sum(0) > allowable_diff]
        if verbose:
            print "clusters that differ:"
            print iDiff
            if iTest >= 0:
                print "Test",iTest,
            print "*** ERROR *** max diff was", diff
            print 
    else:
        if verbose:
            if iTest >= 0:
                print "Test", iTest,
            print "Clusters are OK"
        
    return error


def run_tests(nTests, nPts, nDim, nClusters, nReps=1, verbose = VERBOSE, print_times = PRINT_TIMES):
    # run_tests(nTests, nPts, nDim, nClusters, nReps [, verbose [, print_times]]
    
    # Generate nPts random data elements with nDim dimensions and nCluster random clusters,
    # then run kmeans for nReps and compare gpu and cpu results.  This is repeated nTests times
    cpu_time = 0.
    gpu_time = 0.
    
    gpu_data_time = 0.
    gpu_module_time = 0.
    gpu_ccdist_time = 0.
    gpu_hdclosest_time = 0.
    gpu_init_time = 0.
    gpu_step3_time = 0.
    gpu_step4_time = 0.
    gpu_step56_time = 0.

    np.random.seed(100)
    data = np.random.rand(nDim, nPts).astype(np.float32)
    clusters = np.random.rand(nDim, nClusters).astype(np.float32)

    if verbose:
        print "data"
        print data
        print "\nclusters"
        print clusters

    nErrors = 0

    # repeat this test nTests times
    for iTest in range(nTests):
    
        """
        #run the cpu algorithm
        t1 = time.time()
        (cpu_clusters, cpu_assign) = kmeans_cpu(data, clusters, nReps)
        cpu_assign.shape = (nPts,)
        t2 = time.time()
        cpu_time += t2-t1
        
        if verbose:
            print "------------------------ cpu results ------------------------"
            print "cpu_assignments"
            print cpu_assign
            print "cpu_clusters"
            print cpu_clusters
            print "-------------------------------------------------------------"
        """
        
        #run the gpu algorithm
        t1 = time.time()
        (gpu_ccdist, gpu_hdClosest, gpu_assign, gpu_lower, gpu_upper, \
            gpu_clusters, gpu_cluster_movement, \
            data_time, module_time, init_time, ccdist_time, hdclosest_time, \
            step3_time, step4_time, step56_time) = \
            trikmeans_gpu(data, clusters, nReps, 1)
        t2 = time.time()        
        gpu_time += t2-t1
        gpu_data_time += data_time
        gpu_module_time += module_time
        gpu_ccdist_time += ccdist_time
        gpu_hdclosest_time += hdclosest_time
        gpu_init_time += init_time
        gpu_step3_time += step3_time
        gpu_step4_time += step4_time
        gpu_step56_time += step56_time
        
        if verbose:
            print "------------------------ gpu results ------------------------"
            print "gpu_assignments"
            print gpu_assign
            print "gpu_clusters"
            print gpu_clusters
            print "-------------------------------------------------------------"
            

        """
        # calculate the number of data points in each cluster
        c = np.arange(nClusters)
        c_counts = np.sum(cpu_assign.reshape(nPts,1) == c, axis=0)

        # verify the results...
        nErrors += verify_assignments(gpu_assign.get(), cpu_assign, data, gpu_clusters, cpu_clusters, verbose, iTest)
        nErrors += verify_clusters(gpu_clusters, cpu_clusters, cpu_assign, verbose, iTest)
        """

    if print_times:
        print "\n---------------------------------------------"
        print "nPts      =", nPts
        print "nDim      =", nDim
        print "nClusters =", nClusters
        print "nReps     =", nReps
        #print "average cpu time (ms) =", cpu_time/nTests*1000.
        print "average cpu time (ms) = N/A"
        print "average gpu time (ms) =", gpu_time/nTests*1000.
        print "       data time (ms) =", gpu_data_time/nTests*1000.
        print "     module time (ms) =", gpu_module_time/nTests*1000.
        print "       init time (ms) =", gpu_init_time/nTests*1000.        
        print "     ccdist time (ms) =", gpu_ccdist_time/nTests*1000.        
        print "  hdclosest time (ms) =", gpu_hdclosest_time/nTests*1000.        
        print "      step3 time (ms) =", gpu_step3_time/nTests*1000.        
        print "      step4 time (ms) =", gpu_step4_time/nTests*1000.        
        print "     step56 time (ms) =", gpu_step56_time/nTests*1000.        
        print "---------------------------------------------"

    return nErrors


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
    # when number of tests is -1, it will be calculated based on the size of the problem
    for t in nTest_list:
        for pts in nPts_list:
            for dim in nDim_list:
                for clst in nClusters_list:
                    if clst > pts or clst * dim > 4000:
                        continue
                    for rep in nRep_list:
                        if t < 0:
                            tt = max(1, min(10, 10000000/(pts*dim*clst)))
                        else:
                            tt = t
                        quiet_run(tt, pts, dim, clst, rep, ptimes = print_it);

def run_all(pFlag = 1):
    quiet_run(1, 10, 4, 3, 1, ptimes = pFlag)
    quiet_run(1, 1000, 60, 20, 1, ptimes = pFlag)
    quiet_run(1, 100000, 60, 20, 1, ptimes = pFlag)
    quiet_run(1, 10000, 600, 5, 1, ptimes = pFlag)
    quiet_run(1, 10000, 5, 600, 1, ptimes = pFlag)
    quiet_run(1, 100, 5, 600, 1, ptimes = pFlag)
    quiet_run(1, 100, 600, 5, 1, ptimes = pFlag)
    quiet_run(1, 10, 20, 30, 1, ptimes = pFlag)
    quiet_run(1, 10, 4, 3, 10, ptimes = pFlag)
    quiet_run(1, 1000, 60, 20, 10, ptimes = pFlag)
    quiet_run(1, 10000, 60, 20, 10, ptimes = pFlag)
    quiet_run(1, 1000, 600, 5, 10, ptimes = pFlag)
    quiet_run(1, 1000, 5, 600, 10, ptimes = pFlag)
    quiet_run(1, 100, 5, 600, 10, ptimes = pFlag)
    quiet_run(1, 100, 600, 5, 10, ptimes = pFlag)
    quiet_run(1, 10, 20, 30, 10, ptimes = pFlag)


def run_reps(pFlag = 1):
    quiet_run(1, 10, 4, 3, 5, ptimes = pFlag)
    quiet_run(1, 1000, 60, 20, 5, ptimes = pFlag)
    quiet_run(1, 50000, 60, 20, 5, ptimes = pFlag)
    quiet_run(1, 10000, 600, 5, 5, ptimes = pFlag)
    quiet_run(1, 10000, 5, 600, 5, ptimes = pFlag)
    quiet_run(1, 100, 5, 600, 5, ptimes = pFlag)
    quiet_run(1, 100, 600, 5, 5, ptimes = pFlag)
    quiet_run(1, 10, 20, 30, 5, ptimes = pFlag)
    
def timings(t = 0):
    # run a bunch of tests with optional timing
    quiet_runs([1], [10, 100, 1000, 10000], [2, 8, 32, 600], [3, 9, 27, 600], [1], print_it = t)
    
def quickTimes():
    if quickRun() > 0:
        print "***ERROR***"
    else:
        quiet_run(3, 1000, 60, 20, 5, 1)
        quiet_run(3, 1000, 600, 2, 5, 1)
        quiet_run(3, 1000, 6, 200, 5, 1)
        quiet_run(3, 10000, 60, 20, 5, 1)
        quiet_run(3, 10000, 600, 2, 5, 1)
        quiet_run(3, 10000, 6, 200, 5, 1)
        quiet_run(3, 100000, 6, 20, 5, 1)

def quickRun():
    # run to make sure answers have not changed
    nErrors = run_tests1(1, 1000, 6, 2, 1)
    nErrors += run_tests1(1, 1000, 600, 2, 1)
    nErrors += run_tests1(1, 10000, 2, 600, 1)
    return nErrors