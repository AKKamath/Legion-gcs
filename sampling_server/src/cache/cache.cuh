#ifndef CACHE_H
#define CACHE_H

#include "graph_storage.cuh"
#include "feature_storage.cuh"

#include <iostream>
#include <vector>

// Defined in shared.h (GPUCopyServerClient/src). Kept as a forward decl here
// so this widely-included header doesn't need that repo's include path
struct GpuSharedBuffer;

class CacheController{
public:
    virtual ~CacheController() = default;

    virtual void Initialize(
        int32_t dev_id,
        int32_t total_num_nodes) = 0;

    virtual void Finalize() = 0;

    virtual void FindFeat(
        int32_t* sampled_ids,
        int32_t* cache_offset,
        int32_t* node_counter,
        int32_t op_id,
        void* stream) = 0;

    virtual void FindTopo(int32_t* input_ids,
                        char* partition_index,
                        int32_t* partition_offset,
                        int32_t batch_size,
                        int32_t op_id,
                        void* strm_hdl,
                        int32_t device_id) = 0;

    virtual void CacheProfiling(
        int32_t* sampled_ids,
        int32_t* agg_src_id,
        int32_t* agg_dst_id,
        int32_t* agg_src_off,
        int32_t* agg_dst_off,
        int32_t* node_counter,
        int32_t* edge_counter,
        bool is_presc,
        void* stream) = 0;

    virtual void InitializeMap(int node_capacity, int edge_capacity) = 0;

    virtual void Insert(int32_t* QT, int32_t* QF, int32_t cache_expand, int32_t Kg) = 0;

    virtual void HybridInsert(int32_t* QF, int32_t cpu_cache_capacity, int32_t gpu_cache_capacity) = 0;

    virtual void AccessCount(
        int32_t* d_key,
        int32_t num_keys,
        void* stream) = 0;

    virtual unsigned long long int* GetNodeAccessedMap() = 0;

    virtual unsigned long long int* GetEdgeAccessedMap() = 0;

    virtual int32_t MaxIdNum() = 0;
};

CacheController* NewPreSCCacheController(int32_t train_step, int32_t device_count);

class UnifiedCache{
public:
    void Initialize(
        int64_t cache_memory,
        int32_t float_feature_len,
        int32_t train_step,
        int32_t device_count,
        int32_t cpu_cache_capacity,
        int32_t gpu_cache_capacity);

    void InitializeCacheController(
        int32_t dev_id,
        int32_t total_num_nodes);

    void Finalize(int32_t dev_id);

    //these api will change, find, update, clear
    void FindFeat(
        int32_t* sampled_ids,
        int32_t* cache_offset,
        int32_t* node_counter,
        int32_t op_id,
        void* stream,
        int32_t dev_id);

    void FindTopo(
        int32_t* input_ids,
        char* partition_index,
        int32_t* partition_offset,
        int32_t batch_size,
        int32_t op_id,
        void* strm_hdl,
        int32_t dev_id);

    void CacheProfiling(
        int32_t* sampled_ids,
        int32_t* agg_src_id,
        int32_t* agg_dst_id,
        int32_t* agg_src_off,
        int32_t* agg_dst_off,
        int32_t* node_counter,
        int32_t* edge_counter,
        void* stream,
        int32_t dev_id);

    void AccessCount(
        int32_t* d_key,
        int32_t num_keys,
        void* stream,
        int32_t dev_id);

    void CandidateSelection(int cache_agg_mode, FeatureStorage* feature, GraphStorage* graph);

    void CostModel(int cache_agg_mode, FeatureStorage* feature, GraphStorage* graph, std::vector<uint64_t>& counters, int32_t train_step);

    void FillUp(int cache_agg_mode, FeatureStorage* feature, GraphStorage* graph);

    void HybridInit(FeatureStorage* feature, GraphStorage* graph);

    int32_t MaxIdNum(int32_t dev_id);

    unsigned long long int* GetEdgeAccessedMap(int32_t dev_id);

    void FeatCacheLookup(int32_t* sampled_ids, int32_t* cache_index,
                         int32_t* node_counter, float* dst_float_buffer,
                         int32_t op_id, int32_t dev_id, cudaStream_t strm_hdl);

    // Whether the copy-server-backed lookup kernel should be used instead of
    // the direct-host-read one. Set once from Initialize() based on the
    // LEGION_GCS_ENABLED environment variable, so a/b runs (with vs. without
    // the copy server contending for PCIe) don't need a rebuild.
    bool GCSEnabled() const;

    // Registers dev_id's GpuSharedBuffer (from initClient(), called by the
    // host thread that owns that device's CUDA context) so FeatCacheLookup
    // can hand it to the GCS-backed kernel. Only called when GCSEnabled().
    void SetGCSBuffer(int32_t dev_id, GpuSharedBuffer* buf);

private:
    int32_t NodeCapacity(int32_t dev_id);

    int32_t CPUCapacity();

    int32_t GPUCapacity();

    float* Float_Feature_Cache(int32_t dev_id);//return all features

    float** Global_Float_Feature_Cache(int32_t dev_id);

    std::vector<bool> dev_ids_;/*valid device, indexed by device id, False means invalid, True means valid*/

    int32_t device_count_;

    std::vector<CacheController*> cache_controller_;

    std::vector<int32_t*> QF_;
    std::vector<int32_t*> QT_;
    std::vector<int32_t*> GF_;
    std::vector<int32_t*> GT_;
    std::vector<unsigned long long int*> AF_;
    std::vector<unsigned long long int*> AT_;
    int Kc_;
    int Kg_;

    std::vector<int32_t> node_capacity_;
    std::vector<int32_t> edge_capacity_;

    int32_t cpu_cache_capacity_;//for legion ssd
    int32_t gpu_cache_capacity_;//for legion ssd

    int64_t cache_memory_;
    std::vector<int32_t> sidx_;

    std::vector<int64_t*> int_feature_cache_;
    std::vector<float*> float_feature_cache_;
    std::vector<float**> d_float_feature_cache_ptr_;

    int32_t float_feature_len_;
    int32_t total_num_nodes_;
    float*  cpu_float_features_;

    bool is_presc_;

    // Perf-tuning knobs for the GCS-backed lookup kernels
    bool gcs_enabled_ = false;
    std::vector<GpuSharedBuffer*> gcs_buf_;
    bool gcs_nobatch_ = true;
    int gcs_merge_cap_ = 1;
    int gcs_threads_ = 384;
};



#endif