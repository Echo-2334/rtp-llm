#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"
#if USING_CUDA
#include <cuda_runtime.h>
#endif

namespace rtp_llm {
void CudaGraphRunner::replayDecode(int bs) {
    replayGraph(bs);
}

std::vector<int> CudaGraphRunner::getDecodeBatchSizesToCapture() {
    // If decode_capture_batch_sizes_ is provided from Python, use it directly
    if (!decode_capture_batch_sizes_.empty()) {
        RTP_LLM_LOG_INFO("Using decode capture batch sizes from Python: %zu sizes", decode_capture_batch_sizes_.size());
        // Sort in ascending order (from small to large)
        std::sort(decode_capture_batch_sizes_.begin(), decode_capture_batch_sizes_.end());
        return decode_capture_batch_sizes_;
    }

    // Otherwise, use default logic
    std::vector<int> capture_bs;
    int              max_generate_batch_size = max_bs_;
    RTP_LLM_LOG_INFO("max_generate_batch_size for cuda graph: %d", max_generate_batch_size);
    // Add key batch sizes up to 32
    for (int i : {1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32}) {
        if (i <= max_generate_batch_size) {
            capture_bs.push_back(i);
        }
    }
    // Add range from 48 to max_generate_batch_size, stepping by 16
    for (int i = 48; i <= max_generate_batch_size; i += 16) {
        capture_bs.push_back(i);
    }
    if (capture_bs[capture_bs.size() - 1] != max_generate_batch_size) {
        capture_bs.push_back(max_generate_batch_size);
    }
    return capture_bs;
}

void CudaGraphRunner::captureDecodeOneBatchSize(int bs, bool pool_mode, bool bootstrap_mode) {
    const char* key_type = pool_mode ? "pool batch size" :
                           bootstrap_mode ? "bootstrap batch size" :
                                            "exact batch size";
    captureOneGraphInstance(decodeGraphKey(bs, pool_mode, bootstrap_mode), key_type);
}

void CudaGraphRunner::captureDecode() {
    RTP_LLM_LOG_INFO("Capture Decode Start");
    // Pre-initialize all graph instances with keep_graph based on debug mode
    const bool capture_pool_variant = decodeIndexerPoolEnabled() && !is_target_verify_;
    for (int bs : capture_range_) {
        graph_instances_.try_emplace(decodeGraphKey(bs, false, false), enable_cuda_graph_debug_mode_);
        if (capture_pool_variant) {
            graph_instances_.try_emplace(decodeGraphKey(bs, false, true), enable_cuda_graph_debug_mode_);
            graph_instances_.try_emplace(decodeGraphKey(bs, true, false), enable_cuda_graph_debug_mode_);
        }
    }
    int capture_range_size = capture_range_.size();
    for (int i = capture_range_size - 1; i >= 0; i--) {
        int bs = capture_range_[i];
        for (int graph_mode = 0; graph_mode < 3; ++graph_mode) {
            const bool pool_mode      = graph_mode == 2;
            const bool bootstrap_mode = graph_mode == 1;
            if (graph_mode != 0 && !capture_pool_variant) {
                continue;
            }
            PyModelInputs inputs;
            // Prepare common inputs using shared function.
            prepareCaptureInputs(inputs, bs, bs * num_tokens_per_bs_);
            inputs.attention_inputs.indexer_pool_graph_mode           = pool_mode;
            inputs.attention_inputs.indexer_pool_bootstrap_graph_mode = bootstrap_mode;
            if (capture_pool_variant) {
                inputs.attention_inputs.decode_kv_length.fill_(decodeIndexerPoolMinKvLength() + 1);
            }

            // calculate context_total_kv_length
            int max_input_len  = inputs.attention_inputs.input_lengths.max().item<int>();
            int max_prefix_len = 0;
            if (inputs.attention_inputs.prefix_lengths.defined()
                && inputs.attention_inputs.prefix_lengths.numel() > 0) {
                max_prefix_len = inputs.attention_inputs.prefix_lengths.max().item<int>();
            }
            inputs.attention_inputs.context_total_kv_length = bs * (max_input_len + max_prefix_len);

            const int graph_key = decodeGraphKey(bs, pool_mode, bootstrap_mode);
            graph_instances_[graph_key].mem_hold_ = createCaptureMemoryHold(inputs, bs * num_tokens_per_bs_);
            graph_instances_[graph_key].mem_hold_.attn_pyobj_ =
                py_attn_pyobj_method_(graph_instances_[graph_key].mem_hold_.py_model_inputs_, true);
            try {
                captureDecodeOneBatchSize(bs, pool_mode, bootstrap_mode);
                const char* key_type = pool_mode ? "pool batch size" :
                                       bootstrap_mode ? "bootstrap batch size" :
                                                        "exact batch size";
                replayAndSyncCheck(graph_key, key_type);
                RTP_LLM_LOG_INFO("capture success for batch size: %d, pool mode: %d, bootstrap mode: %d",
                                 bs,
                                 pool_mode,
                                 bootstrap_mode);
            } catch (const std::exception& e) {
                RTP_LLM_LOG_ERROR("CUDA graph capture failed for decode batch size %d, pool mode %d, "
                                  "bootstrap mode %d: %s",
                                  bs,
                                  pool_mode,
                                  bootstrap_mode,
                                  e.what());
#if USING_CUDA
                cudaGetLastError();
                auto sync_err = cudaDeviceSynchronize();
                if (sync_err != cudaSuccess) {
                    RTP_LLM_LOG_ERROR("cudaDeviceSynchronize after capture failure returned: %s",
                                      cudaGetErrorString(sync_err));
                    cudaGetLastError();
                }
#endif
                throw;
            }
        }
    }
    RTP_LLM_LOG_INFO("Capture Decode End");
}
}  // namespace rtp_llm
