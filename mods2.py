from pycuda.compiler import SourceModule

#------------------------------------------------------------------------------------
#                                   source modules
#------------------------------------------------------------------------------------

def get_ccdist_module(nDim, nPts, nClusters, blocksize_ccdist, blocksize_init, 
                        blocksize_step4_x, blocksize_step4_y, blocksize_step56,
                        useTextureForData):
    # module to calculate distances between each cluster and half distance to closest
    
    modString = """

#define NCLUSTERS      """ + str(nClusters)                    + """
#define NDIM           """ + str(nDim)                         + """
#define NPTS           """ + str(nPts)                         + """
#define CLUSTERS_SIZE  """ + str(nClusters*nDim)               + """
#define CLUSTER_CHUNKS """ + str(1 + (nClusters*nDim-1)/blocksize_ccdist) + """
#define THREADS        """ + str(blocksize_ccdist) + """

#define CLUSTER_CHUNKS2 """ + str(1 + (nClusters*nDim-1)/blocksize_init)  + """
#define THREADS2        """ + str(blocksize_init)            + """

#define CLUSTER_CHUNKS3 """ + str(1 + (nClusters*nDim-1)/blocksize_init)  + """
#define THREADS3        """ + str(blocksize_init)            + """

#define THREADS4         """ + str(min(blocksize_step4_x,nClusters))     + """
#define DIMS4            """ + str(blocksize_step4_y)                    + """

#define CLUSTER_CHUNKS5 """ + str(1 + (nClusters-1)/blocksize_step56)  + """
#define THREADS5        """ + str(blocksize_step56)            + """

texture<float, 2, cudaReadModeElementType>texData;


//-----------------------------------------------------------------------
//                          misc functions
//-----------------------------------------------------------------------

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

// calculate the distance from a data point in texture to a cluster
__device__ float dc_dist_tex(int pt, float *cluster)
{
    float dist = (tex2D(texData, 0, pt)-cluster[0]) * (tex2D(texData, 0, pt)-cluster[0]);
    for(int i=1; i<NDIM; i++){
        float diff = tex2D(texData, i, pt) - cluster[i*NCLUSTERS];
        dist += diff * diff;
    }
    return sqrt(dist);
}


//-----------------------------------------------------------------------
//                             ccdist
//-----------------------------------------------------------------------

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


//-----------------------------------------------------------------------
//                           calc_hdClosest
//-----------------------------------------------------------------------

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


//-----------------------------------------------------------------------
//                              init
//-----------------------------------------------------------------------

// **TODO**  need to loop through clusters if all of them don't fit into shared memory

// Assign data points to the nearest cluster

"""
    if useTextureForData:
        modString += "__global__ void init(float *clusters,\n"
    else:
        modString += "__global__ void init(float *data, float *clusters,\n"
    modString += """
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
    for(int c = 0; c < CLUSTER_CHUNKS2; c++, idx += THREADS2){
        if(idx < CLUSTERS_SIZE){
            s_clusters[idx] = clusters[idx];
        }
    }
    __syncthreads();

    // calculate distance to each cluster
    idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= NPTS) return;
    
    // start with cluster 0 as the closest
"""
    if useTextureForData:
        modString += "float min_dist = dc_dist_tex(idx, s_clusters);\n"
    else:
        modString += "float min_dist = dc_dist(data+idx, s_clusters);\n"
    modString += """
    lower[idx] = min_dist;
    int closest = 0;
    
    for(int c=1; c<NCLUSTERS; c++){
    // **TODO**  see if this test to skip some calculations is really worth it on the gpu versus cpu
//        if(min_dist + 0.000001f <= ccdist[closest * NCLUSTERS + c]) continue;
        if(min_dist <= ccdist[closest * NCLUSTERS + c]) continue;
"""
    if useTextureForData:
        modString += "float d = dc_dist_tex(idx, s_clusters + c);\n"
    else:
        modString += "float d = dc_dist(data + idx, s_clusters + c);\n"
    modString += """
        lower[c*NPTS + idx] = d;
        if(d < min_dist){
            min_dist = d;
            closest = c;
        }
    }
    assignments[idx] = closest;
    upper[idx] = min_dist;
}


//-----------------------------------------------------------------------
//                                step3
//-----------------------------------------------------------------------

// **TODO**  need to loop through clusters if all of them don't fit into shared memory


// Step 3 of the algorithm

//__global__ void step3(float *data, float *clusters, 
//__global__ void step3(float *clusters, 
"""
    if useTextureForData:
        modString += "__global__ void step3(float *clusters,\n"
    else:
        modString += "__global__ void step3(float *data, float *clusters,\n"
    modString += """
                                     float *ccdist, float *hdClosest, int *assignments, 
                                     float *lower, float *upper, int *badUpper, 
                                     int *cluster_changed)
{
    // copy clusters to shared memory
    __shared__ float s_clusters[CLUSTERS_SIZE];
    __shared__ int s_cluster_changed[NCLUSTERS];
    int idx = threadIdx.x;
    for(int c = 0; c < CLUSTER_CHUNKS3; c++, idx += THREADS3){
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
"""
    if useTextureForData:
        modString += "d_x_cx = dc_dist_tex(idx, s_clusters+cx);\n"
    else:
        modString += "d_x_cx = dc_dist(data+idx, s_clusters+cx);\n"
    modString += """
            ux = d_x_cx;
            lower[c*NPTS + idx] = d_x_cx;
            rx = 0;
        }else{
            d_x_cx = ux;
        }
        
        // Step 3b: compute distance between x and c change x's assignment if necessary
        if(d_x_cx > lower[c*NPTS + idx] || d_x_cx > ccdist[cx*NCLUSTERS + c]){
"""
    if useTextureForData:
        modString += "float d_x_c = dc_dist_tex(idx, s_clusters+c);\n"
    else:
        modString += "float d_x_c = dc_dist(data+idx, s_clusters+c);\n"
    modString += """
            lower[c*NPTS + idx] = d_x_c;
            if(d_x_c < d_x_cx){
                // assign x to c
                // mark both c and cx as having changed
                s_cluster_changed[c] = 1;
                s_cluster_changed[cx] = 1;
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
    
    __syncthreads();
    
    // update the global cluster-changed flag
    idx = threadIdx.x;
    for(int c = 0; c < CLUSTER_CHUNKS3; c++, idx += THREADS3){
        if(idx < CLUSTERS_SIZE && s_cluster_changed[idx]){
            cluster_changed[idx] = 1;
        }
    }
}

//-----------------------------------------------------------------------
//                                step4
//-----------------------------------------------------------------------

// Calculate the new cluster centers and also the distance between old center and new one
"""
    if useTextureForData:
        modString += "__global__ void step4(float *clusters,\n"
    else:
        modString += "__global__ void step4(float *data, float *clusters,\n"
    modString += """
                    float *new_clusters, int *assignments,
                    float *cluster_movement, float *cluster_changed)
{
    int idx = threadIdx.x;
    int cluster = threadIdx.x + blockDim.x*blockIdx.x;
    if(cluster >= NCLUSTERS) return;
    
    int idy = threadIdx.y;
    // allocate cluster_accum, cluster_count, and initialize to zero
    // also initialize the cluster_movement array to zero
    __shared__ float s_cluster_accum[NDIM * THREADS4];
    __shared__ unsigned int s_cluster_count[THREADS4];
    if(idy == 0){
        s_cluster_count[idx] = 0;
        cluster_movement[idx] = 0.f;
    }
    for(int d = 0; d < NDIM; d+=DIMS4){
        int dim = d + idy;
        if(dim >= NDIM) continue;
        s_cluster_accum[dim*THREADS4 + idx] = 0.0f;
    }

    
    __syncthreads();

"""

    if useTextureForData: 
        modString += """
    for(int i=0; i<NPTS; i++){
        if(i >= NPTS) break;
        if(cluster == assignments[i]){
            for(int d = idy; d < NDIM; d += DIMS4){
                if(d == 0) s_cluster_count[idx] += 1;
                s_cluster_accum[d * THREADS4 + idx] += tex2D(texData, d, i);
            }
        }
    }
"""
    else: 
        modString += """
    // loop over all data and update cluster_count and cluster_accum
    for(int dim = idy; dim < NDIM; dim += DIMS4){
        int dim1 = dim * THREADS4 + idx;
        int dim2 = dim * NPTS;
        for(int i=0; i<NPTS; i++, dim2++){
            if(i >= NPTS) break;
            if(cluster == assignments[i]){
                if(dim == 0) s_cluster_count[idx] += 1;
                s_cluster_accum[dim1] += data[dim2];
            }
        }
    }
"""
    modString += """
    __syncthreads();
    
    // divide the accum by the number of points and copy to the output area
    for(int d = 0; d < NDIM; d += DIMS4){
        int dim = d + idy;
        if(dim >=NDIM) continue;
        int index1 = dim * NCLUSTERS + cluster;
        if(s_cluster_count[idx] > 0){
            new_clusters[index1] = s_cluster_accum[dim * THREADS4 + idx]
                                                    / s_cluster_count[idx];
        }else{
            new_clusters[index1] = clusters[index1];
        }
    }
    
    // calculate the distance between old and new clusters
    cluster_movement[cluster] = calc_dist(new_clusters + cluster, clusters + cluster);
    
}


//-----------------------------------------------------------------------
//                                step56
//-----------------------------------------------------------------------
// **TODO**  need to loop through clusters if all of them don't fit into shared memory

// Assign data points to the nearest cluster
__global__ void step56(int *assignment, 
                        float *lower, float * upper, 
                        float *cluster_movement, int *badUpper)
{
    // copy cluster movement to shared memory
    __shared__ float s_cluster_movement[NCLUSTERS];
//    __shared__ int s_cluster_movement_flag[NCLUSTERS];    // CHANGE#2
    int idx = threadIdx.x;
    for(int c = 0; c < CLUSTER_CHUNKS5; c++, idx += THREADS5){
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

"""
    #print modString
    return SourceModule(modString)
