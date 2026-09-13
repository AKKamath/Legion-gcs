#pragma once
#ifndef CACHE_IMPL_H
#define CACHE_IMPL_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <mutex>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include <functional>
#include <cstdlib>

#include <cstdint>
#include <sys/mman.h>

#include "client_gpu.cuh"
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/types.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

#define MIN_INTERVAL 0.01
#define CLS 64

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <bcht.hpp>
#include <cmd.hpp>
#include <gpu_timer.hpp>
#include <limits>
#include <perf_report.hpp>
#include <rkg.hpp>
#include <type_traits>

#include "system_config.cuh"

using pair_type = bght::pair<int32_t, int32_t>;
using index_pair_type = bght::pair<int32_t, char>;
using offset_pair_type = bght::pair<int32_t, int32_t>;

// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }



__global__ void GetEdgeMem(int32_t* edge_order, uint64_t* edge_mem, int32_t total_num_nodes, int64_t* csr_index){
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < total_num_nodes; thread_idx += gridDim.x * blockDim.x){
        int32_t id = edge_order[thread_idx];
        int64_t neighbor_count = csr_index[id + 1]- csr_index[id];
        edge_mem[thread_idx] = (sizeof(int64_t) + sizeof(int32_t) * neighbor_count);
    }
}


__global__ void aggregate_access(unsigned long long int* agg_access_time, unsigned long long int* new_access_time, int32_t total_num_nodes){
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < (total_num_nodes); thread_idx += blockDim.x * gridDim.x){
        agg_access_time[thread_idx] += new_access_time[thread_idx];
    }
}


__global__ void init_cache_order(int32_t* cache_order, int32_t total_num_nodes){
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < (total_num_nodes); thread_idx += blockDim.x * gridDim.x){
        cache_order[thread_idx] = thread_idx;
    }
}

__global__ void init_feature_cache(float** pptr, float* ptr, int dev_id){
    pptr[dev_id] = ptr;
}

__global__ void InitIndexPair(index_pair_type* pair, int32_t* QT, int32_t capacity, int32_t cache_expand, int32_t Kg, int32_t Ki){
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < capacity * cache_expand; thread_idx += gridDim.x * blockDim.x){
        pair[thread_idx].first = QT[thread_idx];
        pair[thread_idx].second = thread_idx % Kg + Ki * Kg;
    }
}

__global__ void InitOffsetPair(offset_pair_type* pair, int32_t* QT, int32_t capacity, int32_t cache_expand, int32_t Kg){
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < capacity * cache_expand; thread_idx += gridDim.x * blockDim.x){
        pair[thread_idx].first = QT[thread_idx];
        pair[thread_idx].second = thread_idx / Kg;
    }
}

//interleave cache
__global__ void InitPair(pair_type* pair, int32_t* QF, int32_t capacity, int32_t cache_expand, int32_t Kg){
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < capacity * cache_expand; thread_idx += gridDim.x * blockDim.x){
        pair[thread_idx].first = QF[thread_idx];
        pair[thread_idx].second = (thread_idx % Kg) * capacity + thread_idx / Kg;
    }
}


//cpu cache & gpu cache
__global__ void HybridInitPair(pair_type* pair, int32_t* QF, int32_t cpu_cache_capacity, int32_t gpu_cache_capacity){
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < (cpu_cache_capacity + gpu_cache_capacity); thread_idx += gridDim.x * blockDim.x){
        if(thread_idx < gpu_cache_capacity){
            pair[thread_idx].first = QF[thread_idx];
            pair[thread_idx].second = cpu_cache_capacity + thread_idx;
        }else{
            pair[thread_idx].first = QF[thread_idx];
            pair[thread_idx].second = thread_idx - gpu_cache_capacity;
        }
    }
}


__global__ void topo_cache_hit(char* partition_index, int32_t batch_size, int32_t* global_count){
    __shared__ int32_t local_count[1];
    if(threadIdx.x == 0){
        local_count[0] = 0;
    }
    __syncthreads();
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < (batch_size); thread_idx += blockDim.x * gridDim.x){
        int32_t offset = partition_index[thread_idx];
        if(offset >= 0){
            atomicAdd(local_count, 1);
        }
    }
    __syncthreads();
    if(threadIdx.x == 0){
        atomicAdd(global_count, local_count[0]);
    }
}


__global__ void feature_cache_hit(int32_t* cache_offset, int32_t batch_size, int32_t* global_count){
    __shared__ int32_t local_count[1];
    if(threadIdx.x == 0){
        local_count[0] = 0;
    }
    __syncthreads();
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < (batch_size); thread_idx += blockDim.x * gridDim.x){
        int32_t offset = cache_offset[thread_idx];
        if(offset >= 0){
            atomicAdd(local_count, 1);
        }
    }
    __syncthreads();
    if(threadIdx.x == 0){
        atomicAdd(global_count, local_count[0]);
    }
}

__global__ void CacheHitTimes(int32_t* sampled_node, int32_t* node_counter){
    __shared__ int32_t count[2];
    if(threadIdx.x == 0){
        count[0] = 0;
        count[1] = 0;
    }
    __syncthreads();
    int32_t num_nodes = node_counter[INTRABATCH_CON * 2 + 1];
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < num_nodes; thread_idx += gridDim.x * blockDim.x){
        int32_t cid = sampled_node[thread_idx];
        if(cid >= 0){
            atomicAdd(count, 1);
        }
    }
    __syncthreads();
    if(threadIdx.x == 0){
        atomicAdd(node_counter + INTRABATCH_CON * 2 + 2, count[0]);
    }
}

__global__ void FeatFillUp(int32_t capacity, int32_t float_feature_len, float* feature_cache, float* cpu_float_feature, int32_t* QF, int32_t Kg, int32_t Ki){
    for(int64_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < int64_t(capacity) * float_feature_len; thread_idx += gridDim.x * blockDim.x){
        int32_t id = QF[(thread_idx / float_feature_len) * Kg + Ki];
        feature_cache[thread_idx] = cpu_float_feature[int64_t(id) * float_feature_len + thread_idx % float_feature_len];
    }
}

__global__ void HotnessMeasure(int32_t* new_batch_ids, int32_t* node_counter, unsigned long long int* access_map){
    int32_t num_candidates = node_counter[INTRABATCH_CON * 2 + 1];
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < num_candidates; thread_idx += gridDim.x * blockDim.x){
        int32_t cid = new_batch_ids[thread_idx];
        if(cid >= 0){
            atomicAdd(access_map + cid, 1);
        }
    }
}



__global__ void feat_cache_lookup(
	float* cpu_float_feature, float* gpu_float_feature, int32_t float_feature_len,
	int32_t* sampled_ids, int32_t* cache_index,
    int32_t cpu_cache_capacity, int32_t gpu_cache_capacity,
	int32_t* node_counter, float* dst_float_buffer,
	int32_t op_id)
{
    int32_t node_off = 0;
	int32_t batch_size = 0;

    node_off   = node_counter[(op_id % INTRABATCH_CON) * 2];
    batch_size = node_counter[(op_id % INTRABATCH_CON) * 2 + 1];
    // if(threadIdx.x == 0 && blockIdx.x == 0){
    //     printf("%d %d\n", node_off, batch_size);
    // }
	if(float_feature_len > 0){
		for(int64_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < (int64_t(batch_size) * float_feature_len); thread_idx += blockDim.x * gridDim.x){
			int32_t gidx;//global cache index
            int32_t fidx;//local cache index
            int32_t didx;//device index
            int32_t foffset;
            gidx = (cache_index[thread_idx / float_feature_len]);
			foffset = thread_idx % float_feature_len;
			if(gidx < cpu_cache_capacity && gidx >= 0){/*cache in cpu*/
				fidx = gidx % cpu_cache_capacity;//sampled_ids[node_off + (thread_idx / float_feature_len)];
				dst_float_buffer[int64_t(int64_t((int64_t(node_off) * float_feature_len)) + thread_idx)] = cpu_float_feature[int64_t(int64_t(int64_t(fidx) * float_feature_len) + foffset)];
			}else if(gidx >= cpu_cache_capacity){/*cache in gpu*/
                // didx = (gidx - cpu_cache_capacity) / gpu_cache_capacity;//device idx in clique
                fidx = (gidx - cpu_cache_capacity) % gpu_cache_capacity;
				dst_float_buffer[int64_t(int64_t((int64_t(node_off) * float_feature_len)) + thread_idx)] = gpu_float_feature[int64_t(int64_t(int64_t(fidx) * float_feature_len) + foffset)];
			}
		}
	}
}



__global__ void multiGPU_feat_cache_lookup(
	float* cpu_float_features, float** gpu_float_feature, int32_t float_feature_len,
	int32_t* sampled_ids, int32_t* cache_index, int32_t cache_capacity,
	int32_t* node_counter, float* dst_float_buffer,
	int32_t total_num_nodes,
	int32_t dev_id,
	int32_t op_id)
{
    int32_t node_off = 0;
	int32_t batch_size = 0;

    node_off   = node_counter[(op_id % INTRABATCH_CON) * 2];
    batch_size = node_counter[(op_id % INTRABATCH_CON) * 2 + 1];
	int32_t gidx;//global cache index
	int32_t fidx;//local cache index
	int32_t didx;//device index
	int32_t foffset;
	if(float_feature_len > 0){
		for(int64_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < (int64_t(batch_size) * float_feature_len); thread_idx += blockDim.x * gridDim.x){
			gidx = (cache_index[thread_idx / float_feature_len]);
			didx = gidx / cache_capacity;//device idx in clique
			fidx = gidx % cache_capacity;
			foffset = thread_idx % float_feature_len;
			if(gidx < 0){/*cache miss*/
				fidx = sampled_ids[node_off + (thread_idx / float_feature_len)];
				if(fidx >= 0){
					dst_float_buffer[int64_t(int64_t((int64_t(node_off) * float_feature_len)) + thread_idx)] = cpu_float_features[int64_t(int64_t(int64_t(fidx%total_num_nodes) * float_feature_len) + foffset)];
				}
			}else{/*cache hit, find global position*/
				dst_float_buffer[int64_t(int64_t((int64_t(node_off) * float_feature_len)) + thread_idx)] = gpu_float_feature[didx][int64_t(int64_t(int64_t(fidx) * float_feature_len) + foffset)];
			}
		}
	}
}

#define GCS_MAX_WARPS_PER_BLOCK 32 // covers up to 1024 threads/block

__global__ void multiGPU_feat_cache_lookup_gcs_nobatch(
	GpuSharedBuffer* gcs_buf,
	float* cpu_float_features, float** gpu_float_feature, int32_t float_feature_len,
	int32_t* sampled_ids, int32_t* cache_index, int32_t cache_capacity,
	int32_t* node_counter, float* dst_float_buffer,
	int32_t total_num_nodes,
	int32_t dev_id,
	int32_t op_id)
{
    int32_t node_off = 0;
	int32_t batch_size = 0;

    node_off   = node_counter[(op_id % INTRABATCH_CON) * 2];
    batch_size = node_counter[(op_id % INTRABATCH_CON) * 2 + 1];
	int32_t gidx;//global cache index
	int32_t fidx;//local cache index
	int32_t didx;//device index

	const int warps_per_block = blockDim.x >> 5;
	const int global_warp     = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
	const int num_warps       = gridDim.x * warps_per_block;
	const size_t row_bytes    = (size_t)float_feature_len * sizeof(float);

	if(float_feature_len > 0){
		for(int64_t row = global_warp; row < batch_size; row += num_warps){
			gidx = (cache_index[row]);
			didx = gidx / cache_capacity;//device idx in clique
			fidx = gidx % cache_capacity;
			float* dst_row = dst_float_buffer + (size_t)(int64_t(node_off) + row) * float_feature_len;
			if(gidx < 0){/*cache miss*/
				fidx = sampled_ids[node_off + row];
				if(fidx >= 0){
					const float* src_row = cpu_float_features + (size_t)(fidx % total_num_nodes) * float_feature_len;
					gpuSysMemcpy_warp(gcs_buf, dst_row, (void*)src_row, row_bytes);
				}
			}else{/*cache hit, find global position*/
				const float* src_row = gpu_float_feature[didx] + (size_t)fidx * float_feature_len;
				performCopy<32>(dst_row, (void*)src_row, row_bytes);
			}
		}
	}
}

template <int RowsPerStep>
__global__ void multiGPU_feat_cache_lookup_gcs(
	GpuSharedBuffer* gcs_buf,
	float* cpu_float_features, float** gpu_float_feature, int32_t float_feature_len,
	int32_t* sampled_ids, int32_t* cache_index, int32_t cache_capacity,
	int32_t* node_counter, float* dst_float_buffer,
	int32_t total_num_nodes,
	int32_t dev_id, int32_t op_id)
{
	static_assert(RowsPerStep == 1 || RowsPerStep == 2 || RowsPerStep == 4 || RowsPerStep == 8,
	              "RowsPerStep must be a power of two in [1, 8]");
	if (float_feature_len <= 0) return;

	int32_t node_off   = node_counter[(op_id % INTRABATCH_CON) * 2];
	int32_t batch_size = node_counter[(op_id % INTRABATCH_CON) * 2 + 1];

	const int lane            = threadIdx.x & 31;
	const int warp_in_block   = threadIdx.x >> 5;
	const int warps_per_block = blockDim.x >> 5;
	const int global_warp     = blockIdx.x * warps_per_block + warp_in_block;
	const int num_warps       = gridDim.x * warps_per_block;

	// Compacted (dst_row, src_row) pairs for this warp's current RowsPerStep-row step.
	__shared__ int32_t s_dst_idx[GCS_MAX_WARPS_PER_BLOCK][RowsPerStep];
	__shared__ int32_t s_src_idx[GCS_MAX_WARPS_PER_BLOCK][RowsPerStep];
	int32_t* my_dst = s_dst_idx[warp_in_block];
	int32_t* my_src = s_src_idx[warp_in_block];

	const size_t row_bytes = (size_t)float_feature_len * sizeof(float);

	for (int32_t row_base = global_warp * RowsPerStep; row_base < batch_size; row_base += num_warps * RowsPerStep) {
		bool participates = lane < RowsPerStep;
		int32_t row = row_base + lane;
		bool valid = participates && (row < batch_size);
		int32_t gidx = valid ? cache_index[row] : 0;
		bool is_miss = valid && (gidx < 0);

		int32_t src_row_id = -1;
		if (is_miss) {
			src_row_id = sampled_ids[node_off + row];
			if (src_row_id < 0) is_miss = false; // matches multiGPU_feat_cache_lookup's fidx>=0 guard: skip, no write
		}

		// All 32 lanes execute this ballot unconditionally
		unsigned mask = __ballot_sync(0xFFFFFFFFu, is_miss);
		int32_t my_pos = __popc(mask & ((1u << lane) - 1)); // exclusive prefix count
		if (is_miss) {
			my_dst[my_pos] = node_off + row; // absolute row index into dst_float_buffer
			my_src[my_pos] = src_row_id % total_num_nodes;
		}
		__syncwarp();

		int32_t num_miss = __popc(mask);
		if (num_miss > 0) {
			gpuSysMemcpyGather_warp<RowsPerStep>(gcs_buf, dst_float_buffer, my_dst,
			                                      cpu_float_features, my_src,
			                                      row_bytes, num_miss);
		}
		__syncwarp();

		#pragma unroll
		for (int r = 0; r < RowsPerStep; r++) {
			int32_t hit_row = row_base + r;
			if (hit_row >= batch_size) break;
			int32_t gidx_r  = __shfl_sync(0xFFFFFFFFu, gidx, r);
			int32_t valid_r = __shfl_sync(0xFFFFFFFFu, (int32_t)valid, r);
			if (valid_r && gidx_r >= 0) {
				int32_t didx = gidx_r / cache_capacity;
				int32_t fidx = gidx_r % cache_capacity;
				const float* src_row = gpu_float_feature[didx] + (size_t)fidx * float_feature_len;
				float* dst_row = dst_float_buffer + (size_t)(node_off + hit_row) * float_feature_len;
				performCopy<32>(dst_row, (void*)src_row, row_bytes);
			}
		}
	}
}


void mmap_cache_read(std::string &cache_file, std::vector<int32_t>& cache_map){
    int64_t t_idx = 0;
    int32_t fd = open(cache_file.c_str(), O_RDONLY);
    if(fd == -1){
        std::cout<<"cannout open file: "<<cache_file<<std::endl;
    }
    // int64_t buf_len = lseek(fd, 0, SEEK_END);
    int64_t buf_len = int64_t(int64_t(cache_map.size()) * 4);
    const int32_t* buf = (int32_t *)mmap(NULL, buf_len, PROT_READ, MAP_PRIVATE, fd, 0);
    const int32_t* buf_end = buf + buf_len/sizeof(int32_t);
    int32_t temp;
    while(buf < buf_end){
        temp = *buf;
        cache_map[t_idx++] = temp;
        buf++;
    }
    close(fd);
    return;
}


#endif