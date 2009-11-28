from pycuda.compiler import SourceModule

print "mods.py"
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

// Determination of hdClosest
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


texture<float, 2, cudaReadModeElementType>texData;


// calculate the distance from a data point to a cluster
//__device__ float dc_dist(float *data, float *cluster)
//{
//    float dist = (data[0]-cluster[0]) * (data[0]-cluster[0]);
//    for (int i=1; i<NDIM; i++) {
//        float diff = data[i*NPTS] - cluster[i*NCLUSTERS];
//        dist += diff*diff;
//    }
//    return sqrt(dist);
//}

__device__ float dc_dist_tex(int pt, float *cluster)
{
    float dist = (tex2D(texData, 0, pt)-cluster[0]) * (tex2D(texData, 0, pt)-cluster[0]);
    for(int i=1; i<NDIM; i++){
        float diff = tex2D(texData, i, pt) - cluster[i*NCLUSTERS];
        dist += diff * diff;
    }
    return sqrt(dist);
}


// **TODO**  need to loop through clusters if all of them don't fit into shared memory

// Assign data points to the nearest cluster
//__global__ void init(float *data, float *clusters, 
//__global__ void init(float *dataout, float *clusters, 
__global__ void init(float *clusters, 
                     float *ccdist, float *hdClosest, int *assignments, 
                     float *lower, float *upper)
{

//    int idx = threadIdx.x + blockDim.x * blockIdx.x;
//    if(idx >= NPTS) return;
//    for(int d = 0; d<NDIM; d++){
//        dataout[d*NPTS + idx] = tex2D(texData, d, idx);
//    }
    
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
//    float min_dist = dc_dist(data+idx, s_clusters);
    float min_dist = dc_dist_tex(idx, s_clusters);
    lower[idx] = min_dist;
    int closest = 0;
    
    for(int c=1; c<NCLUSTERS; c++){
    // **TODO**  see if this test to skip some calculations is really worth it on the gpu versus cpu
        if(min_dist + 0.000001f <= ccdist[closest * NCLUSTERS + c]) continue;
//        float d = dc_dist(data + idx, s_clusters + c);
        float d = dc_dist_tex(idx, s_clusters + c);
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
