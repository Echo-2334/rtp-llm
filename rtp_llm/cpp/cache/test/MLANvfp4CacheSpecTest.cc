#include <gtest/gtest.h>
#include <set>

#include "rtp_llm/cpp/cache/MLAKVCacheSpec.h"
#include "rtp_llm/cpp/config/StaticConfig.h"

namespace rtp_llm {
namespace test {

class MLANvfp4CacheSpecTest: public ::testing::Test {
protected:
    void SetUp() override {
        old_core_dump_ = StaticConfig::user_ft_core_dump_on_exception;
        StaticConfig::user_ft_core_dump_on_exception = false;
        attention_.kv_cache_dtype = KvCacheDataType::NVFP4;
        attention_.use_mla = true;
        attention_.is_sparse = true;
        attention_.kv_lora_rank = 512;
        attention_.rope_head_dim = 64;
        attention_.indexer_head_dim = 128;
        attention_.tokens_per_block = 64;
        attention_.kernel_tokens_per_block = 64;
    }

    void TearDown() override {
        StaticConfig::user_ft_core_dump_on_exception = old_core_dump_;
    }

    bool old_core_dump_;
    AttentionConfigs attention_{};
    ParallelismConfig parallelism_{};
};

TEST_F(MLANvfp4CacheSpecTest, BlockIncludesCentersAndFormatIdentifiesCount) {
    std::set<std::string> formats;
    for (size_t centers : {0, 1, 2, 4}) {
        attention_.mla_nvfp4_cluster_centers = centers;
        MLAKVCacheSpec spec(attention_, parallelism_);
        spec.dtype = DataType::TYPE_UINT8;
        EXPECT_EQ(spec.block_size_bytes(), 64 * (432 + 16 * centers));
        EXPECT_EQ(spec.k_block_size_bytes(), spec.block_size_bytes());
        EXPECT_EQ(spec.v_block_size_bytes(), 0);
        EXPECT_EQ(spec.scale_block_size_bytes(), 64 * 132);
        EXPECT_TRUE(formats.insert(attention_.mlaCacheFormat()).second);
    }
    EXPECT_EQ(attention_.mlaCacheFormat(), "nvfp4_mla_cluster_v1_b64_k4_496");
    attention_.mla_nvfp4_cluster_centers = 0;
    EXPECT_EQ(attention_.mlaCacheFormat(), "nvfp4_mla_v1_432");
    attention_.kv_cache_dtype = KvCacheDataType::FP8;
    EXPECT_EQ(attention_.mlaCacheFormat(), "");
}

TEST_F(MLANvfp4CacheSpecTest, RejectsUnsupportedLayouts) {
    attention_.mla_nvfp4_cluster_centers = 3;
    EXPECT_THROW(MLAKVCacheSpec(attention_, parallelism_), std::runtime_error);
    attention_.mla_nvfp4_cluster_centers = 4;
    attention_.tokens_per_block = 128;
    EXPECT_THROW(MLAKVCacheSpec(attention_, parallelism_), std::runtime_error);
    attention_.tokens_per_block = 64;
    attention_.kernel_tokens_per_block = 32;
    EXPECT_THROW(MLAKVCacheSpec(attention_, parallelism_), std::runtime_error);
    attention_.kernel_tokens_per_block = 64;
    attention_.kv_cache_dtype = KvCacheDataType::FP8;
    EXPECT_THROW(MLAKVCacheSpec(attention_, parallelism_), std::runtime_error);
}

}
}
