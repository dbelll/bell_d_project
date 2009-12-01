from pycuda.compiler import SourceModule
from looper import loop

#------------------------------------------------------------------------------------
#                                   source modules
#------------------------------------------------------------------------------------

def get_ccdist_module(nDim, nPts, nClusters, blocksize_ccdist, blocksize_init, 
                        blocksize_step4, seqcount_step4, gridsize_step4, 
                        blocksize_step4part2, blocksize_step56,
                        blocksize_calcm, useTextureForData):
    # module to calculate distances between each cluster and half distance to closest
    
    modString = """

#define NCLUSTERS      """ + str(nClusters)                                + """
#define NDIM           """ + str(nDim)                                     + """
#define NPTS           """ + str(nPts)                                     + """
#define CLUSTERS_SIZE  """ + str(nClusters*nDim)                           + """
#define CLUSTER_CHUNKS """ + str(1 + (nClusters*nDim-1)/blocksize_ccdist)  + """
#define THREADS        """ + str(blocksize_ccdist)                         + """

#define CLUSTER_CHUNKS2 """ + str(1 + (nClusters*nDim-1)/blocksize_init)   + """
#define THREADS2        """ + str(blocksize_init)                          + """

#define CLUSTER_CHUNKS3 """ + str(1 + (nClusters*nDim-1)/blocksize_init)   + """
#define THREADS3        """ + str(blocksize_init)                          + """

#define THREADS4        """ + str(blocksize_step4)                         + """
#define BLOCKS4         """ + str(gridsize_step4)                          + """
#define SEQ_COUNT4      """ + str(seqcount_step4)                          + """
#define RED_OUT_WIDTH   """ + str(gridsize_step4*nClusters)                + """
#define THREADS4PART2   """ + str(blocksize_step4part2)                    + """

#define THREADS4A       """ + str(blocksize_calcm)                         + """
#define CLUSTER_CHUNKS4A """ + str(1 + (nClusters*nDim-1)/blocksize_calcm) + """

#define CLUSTER_CHUNKS5 """ + str(1 + (nClusters-1)/blocksize_step56)      + """
#define THREADS5        """ + str(blocksize_step56)                        + """

texture<float, 2, cudaReadModeElementType>texData;


//-----------------------------------------------------------------------
//                          misc functions
//-----------------------------------------------------------------------

// calculate the distance beteen two clusters
__device__ float calc_dist(float *clusterA, float *clusterB)
{
    float dist = (clusterA[0]-clusterB[0]) * (clusterA[0]-clusterB[0]);
//    for (int i=1; i<NDIM; i++) {
//        float diff = clusterA[i*NCLUSTERS] - clusterB[i*NCLUSTERS];
//        dist += diff*diff;
//    }

""" + loop(1, nDim, 16, """ 
        dist += (clusterA[{0}*NCLUSTERS] - clusterB[{0}*NCLUSTERS])
                *(clusterA[{0}*NCLUSTERS] - clusterB[{0}*NCLUSTERS]);
"""        ) + """


    return sqrt(dist);
}

// calculate the distance from a data point to a cluster
__device__ float dc_dist(float *data, float *cluster)
{
    float dist = (data[0]-cluster[0]) * (data[0]-cluster[0]);
//    for (int i=1; i<NDIM; i++) {
//        float diff = data[i*NPTS] - cluster[i*NCLUSTERS];
//        dist += diff*diff;
//    }

""" + loop(1, nDim, 16, """ 
        dist += (data[{0}*NPTS] - cluster[{0}*NCLUSTERS])
                *(data[{0}*NPTS] - cluster[{0}*NCLUSTERS]);
"""        ) + """

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
//    __shared__ int s_cluster_changed[NCLUSTERS];
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
//                s_cluster_changed[c] = 1;
//                s_cluster_changed[cx] = 1;
                ux = d_x_c;
                cx = c;
                rx = 0;
                // **TODO**  flag the clusters that have changed which is needed for later steps
            }
        }
    }
    __syncthreads();
    upper[idx] = ux;
    if(cx != assignments[idx]){
        cluster_changed[cx] = 1;
        cluster_changed[assignments[idx]] = 1;
        assignments[idx] = cx;
    }
    badUpper[idx] = rx;
    
    __syncthreads();
    
//    // update the global cluster-changed flag
//    idx = threadIdx.x;
//    for(int c = 0; c < CLUSTER_CHUNKS3; c++, idx += THREADS3){
//        if(idx < CLUSTERS_SIZE && s_cluster_changed[idx]){
//            cluster_changed[idx] = 1;
//        }
//    }
}

//-----------------------------------------------------------------------
//                                step4
//-----------------------------------------------------------------------

// Calculate the new cluster centers
"""
    if useTextureForData:
        modString += "__global__ void step4(\n"
    else:
        modString += "__global__ void step4(float *data,\n"
    modString += """
                        int *cluster_changed, float *reduction_out,
                        int *reduction_counts, int *assignments)
{
    __shared__ float s_data[THREADS4];
    __shared__ int s_count[THREADS4];

    int idx = threadIdx.x;
    int iData = blockIdx.x * THREADS4 * SEQ_COUNT4 + idx;
    
    int dim = blockIdx.y;
    
    for(int c=0; c<NCLUSTERS; c++){
        if(cluster_changed[c]){
            float tot = 0.0f;
            int count = 0;
            for(int s=0; s<SEQ_COUNT4; s++){
                if(iData >= NPTS) break;
                if(assignments[iData] == c){
                    count += 1;
//                    tot += tex2D(texData, dim, iData);
"""
    if useTextureForData:
        modString += "tot += tex2D(texData, dim, iData);\n"
    else:
        modString += "tot += data[dim*NPTS + iData];\n"
    modString += """
                }
            }
            s_data[idx] = tot;
            s_count[idx] = count;
            __syncthreads();

            #if (THREADS4 >= 512) 
            if (idx < 256) { 
                s_data[idx] += s_data[idx + 256]; 
                s_count[idx] += s_count[idx + 256];
            }
            __syncthreads();
            #endif

            #if (THREADS4 >= 256) 
            if (idx < 128) { 
                s_data[idx] += s_data[idx+128]; 
                s_count[idx] += s_count[idx + 128];
            } 
            __syncthreads(); 
            #endif

            #if (THREADS4 >= 128) 
            if (idx < 64) { 
                s_data[idx] += s_data[idx + 64]; 
                s_count[idx] += s_count[idx + 64];
            } 
            __syncthreads(); 
            #endif

            if (idx < 32){
                if (THREADS4 >= 64){
                    s_data[idx] += s_data[idx + 32];
                    s_count[idx] += s_count[idx + 32];
                }
                if (THREADS4 >= 32){
                    s_data[idx] += s_data[idx + 16];
                    s_count[idx] += s_count[idx + 16];
                }
                if (THREADS4 >= 16){
                    s_data[idx] += s_data[idx + 8];
                    s_count[idx] += s_count[idx + 8];
                }
                if (THREADS4 >= 8){
                    s_data[idx] += s_data[idx + 4];
                    s_count[idx] += s_count[idx + 4];
                }
                if (THREADS4 >= 4){
                    s_data[idx] += s_data[idx + 2];
                    s_count[idx] += s_count[idx + 2];
                }
                if (THREADS4 >= 2){
                    s_data[idx] += s_data[idx + 1];
                    s_count[idx] += s_count[idx + 1];
                }
            }
        }

        if(idx == 0){
            reduction_out[dim * RED_OUT_WIDTH + blockIdx.x * NCLUSTERS + c] = s_data[0];
            reduction_counts[blockIdx.x * NCLUSTERS + c] = s_count[0];
        }
    }
}


//-----------------------------------------------------------------------
//                           step4part2
//-----------------------------------------------------------------------

// Calculate new cluster centers using reduction, part 2

__global__ void step4part2(int *cluster_changed, float *reduction_out, int *reduction_counts,
                            float *new_clusters, float *clusters)
{
    __shared__ float s_data[THREADS4PART2];
    __shared__ int s_count[THREADS4PART2];
    
    int idx = threadIdx.x;
    
    int dim = blockIdx.y;

    for(int c=0; c<NCLUSTERS; c++){
        s_data[idx] = 0.0f;
        s_count[idx] = 0;
        if(cluster_changed[c]){
//            s_data[idx] = 0.0f;
//            s_count[idx] = 0;
            if(idx < BLOCKS4){
                // straight copy of data into shared memory
                s_data[idx] = reduction_out[dim*RED_OUT_WIDTH + idx*NCLUSTERS + c];
                s_count[idx] = reduction_counts[idx*NCLUSTERS + c];
            }
            __syncthreads();
            
            // do the reduction
            #if (THREADS4PART2 >= 512) 
            if (idx < 256) { 
                s_data[idx] += s_data[idx + 256]; 
                s_count[idx] += s_count[idx + 256];
            }
            __syncthreads();
            #endif

            #if (THREADS4PART2 >= 256) 
            if (idx < 128) { 
                s_data[idx] += s_data[idx+128]; 
                s_count[idx] += s_count[idx + 128];
            } 
            __syncthreads(); 
            #endif

            #if (THREADS4PART2 >= 128) 
            if (idx < 64) { 
                s_data[idx] += s_data[idx + 64]; 
                s_count[idx] += s_count[idx + 64];
            } 
            __syncthreads(); 
            #endif

            if (idx < 32){
                if (THREADS4PART2 >= 64){
                    s_data[idx] += s_data[idx + 32];
                    s_count[idx] += s_count[idx + 32];
                }
                if (THREADS4PART2 >= 32){
                    s_data[idx] += s_data[idx + 16];
                    s_count[idx] += s_count[idx + 16];
                }
                if (THREADS4PART2 >= 16){
                    s_data[idx] += s_data[idx + 8];
                    s_count[idx] += s_count[idx + 8];
                }
                if (THREADS4PART2 >= 8){
                    s_data[idx] += s_data[idx + 4];
                    s_count[idx] += s_count[idx + 4];
                }
                if (THREADS4PART2 >= 4){
                    s_data[idx] += s_data[idx + 2];
                    s_count[idx] += s_count[idx + 2];
                }
                if (THREADS4PART2 >= 2){
                    s_data[idx] += s_data[idx + 1];
                    s_count[idx] += s_count[idx + 1];
                }
            }
        }

        // calculate the new cluster, or copy the old one has no values or didn't change
        if(idx == 0){
            if(s_count[0] == 0){
                new_clusters[dim * NCLUSTERS + c] = clusters[dim*NCLUSTERS + c];
            }else{
                new_clusters[dim * NCLUSTERS + c] = s_data[0] / s_count[0];
            }
        }
            
    }
}
    

//-----------------------------------------------------------------------
//                                calc movement
//-----------------------------------------------------------------------
__global__ void calc_movement(float *clusters, float *new_clusters, float *cluster_movement, 
                                int *cluster_changed)
{
    // copy clusters to shared memory
    __shared__ float s_clusters[CLUSTERS_SIZE];
    int idx = threadIdx.x;
    for(int c = 0; c < CLUSTER_CHUNKS4A; c++, idx += THREADS4A){
        if(idx < CLUSTERS_SIZE){
            s_clusters[idx] = clusters[idx];
        }
    }
    __syncthreads();
    
    int cluster = threadIdx.x + blockDim.x*blockIdx.x;
    if(cluster_changed[cluster])
        cluster_movement[cluster] = calc_dist(s_clusters + cluster, new_clusters + cluster);
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
/*
    upper[idx] += s_cluster_movement[assignment[idx]];
    
    // reset the badUpper flag
    badUpper[idx] = 1;
*/
}

"""
    #print modString
    return SourceModule(modString)
