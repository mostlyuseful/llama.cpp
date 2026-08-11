#include "llama.h"
#include "../src/llama-ext.h"
#include "arg.h"
#include "common.h"
#include "log.h"
#include "mtmd.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static void silent_log(ggml_log_level, const char *, void *) {}

struct memory_row {
    llama_memory_breakdown data = {};
    ggml_backend_dev_t dev = nullptr;
    uint64_t mmproj = 0;
};

static bool add_mmproj_memory(
        std::vector<memory_row> & rows,
        const std::map<ggml_backend_dev_t, size_t> & usage,
        std::string & error) {
    bool has_usable_data = false;

    for (const auto & entry : usage) {
        ggml_backend_dev_t dev = entry.first;
        const size_t size = entry.second;

        if (dev == nullptr) {
            error = "estimator reported an unknown backend device";
            return false;
        }
        if (size == 0) {
            continue;
        }

        has_usable_data = true;

        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            auto host = std::find_if(rows.begin(), rows.end(), [](const memory_row & row) {
                return !row.data.is_gpu;
            });
            if (host == rows.end()) {
                error = "text-model estimate has no Host row";
                return false;
            }
            host->mmproj += size;
            continue;
        }

        auto row = std::find_if(rows.begin(), rows.end(), [dev](const memory_row & candidate) {
            return candidate.dev == dev;
        });
        if (row != rows.end()) {
            row->mmproj += size;
            continue;
        }

        memory_row extra;
        extra.dev = dev;
        extra.mmproj = size;
        extra.data.is_gpu = true;

        size_t free = 0;
        size_t total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        extra.data.free = free;
        extra.data.total = total;

        const char * name = ggml_backend_dev_name(dev);
        const char * description = ggml_backend_dev_description(dev);
        snprintf(extra.data.name, sizeof(extra.data.name), "%s (%s)", name, description);
        rows.push_back(extra);
    }

    if (!has_usable_data) {
        error = "estimator returned no usable device data";
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MEMORY)) {
        return 1;
    }

    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    mparams.no_alloc  = true;
    mparams.load_mode = LLAMA_LOAD_MODE_NONE;

    llama_backend_init();
    llama_numa_init(params.numa);

    ggml_log_callback old_log_cb;
    void * old_log_data;
    llama_log_get(&old_log_cb, &old_log_data);
    llama_log_set(silent_log, nullptr);

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (!model) {
        llama_log_set(old_log_cb, old_log_data);
        llama_backend_free();
        fprintf(stderr, "error: failed to load model '%s'\n", params.model.path.c_str());
        return 1;
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_log_set(old_log_cb, old_log_data);
        llama_model_free(model);
        llama_backend_free();
        fprintf(stderr, "error: failed to create context\n");
        return 1;
    }

    llama_log_set(old_log_cb, old_log_data);

    constexpr size_t MAX_DEVS = 64;
    llama_memory_breakdown bk[MAX_DEVS];
    size_t n = MAX_DEVS;
    llama_memory_breakdown_get(ctx, bk, &n);

    std::vector<memory_row> rows;
    rows.reserve(n);
    int32_t i_dev = 0;
    const int32_t n_devs = llama_model_n_devices(model);
    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t dev = nullptr;
        if (bk[i].is_gpu) {
            if (i_dev >= n_devs) {
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                fprintf(stderr, "error: failed to associate text-model memory with backend devices\n");
                return 1;
            }
            dev = llama_model_get_device(model, i_dev++);
        }
        rows.push_back({bk[i], dev, 0});
    }
    if (i_dev != n_devs || std::none_of(rows.begin(), rows.end(), [](const memory_row & row) {
            return !row.data.is_gpu;
        })) {
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        fprintf(stderr, "error: incomplete text-model memory breakdown\n");
        return 1;
    }

    if (!params.mmproj.path.empty()) {
        mtmd_context_params mtmd_params = mtmd_context_params_default();
        mtmd_params.use_gpu          = params.mmproj_use_gpu;
        mtmd_params.print_timings    = false;
        mtmd_params.n_threads        = params.cpuparams.n_threads;
        mtmd_params.flash_attn_type  = params.flash_attn_type;
        mtmd_params.warmup           = params.warmup;
        mtmd_params.image_min_tokens = params.image_min_tokens;
        mtmd_params.image_max_tokens = params.image_max_tokens;
        mtmd_params.batch_max_tokens = params.mtmd_batch_max_tokens;

        const auto usage = mtmd_get_memory_usage(params.mmproj.path.c_str(), mtmd_params);
        std::string error;
        if (!add_mmproj_memory(rows, usage, error)) {
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            fprintf(stderr, "error: failed to estimate mmproj memory for '%s': %s\n",
                    params.mmproj.path.c_str(), error.c_str());
            return 1;
        }
    }

    for (const auto & row : rows) {
        const auto & d = row.data;
        uint64_t self = d.model + d.context + d.compute + row.mmproj;
        if (d.is_gpu) {
            printf("%-30s  total=%6llu MiB  free=%6llu MiB  "
                   "model=%6llu MiB  kv=%6llu MiB  compute=%6llu MiB  mmproj=%6llu MiB\n",
                   d.name,
                   (unsigned long long)(d.total       / (1024*1024)),
                   (unsigned long long)(d.free        / (1024*1024)),
                   (unsigned long long)(d.model       / (1024*1024)),
                   (unsigned long long)(d.context     / (1024*1024)),
                   (unsigned long long)(d.compute     / (1024*1024)),
                   (unsigned long long)(row.mmproj    / (1024*1024)));
        } else {
            printf("%-30s  self=%6llu MiB  "
                   "model=%6llu MiB  kv=%6llu MiB  compute=%6llu MiB  mmproj=%6llu MiB\n",
                   d.name,
                   (unsigned long long)(self          / (1024*1024)),
                   (unsigned long long)(d.model       / (1024*1024)),
                   (unsigned long long)(d.context     / (1024*1024)),
                   (unsigned long long)(d.compute     / (1024*1024)),
                   (unsigned long long)(row.mmproj    / (1024*1024)));
        }
    }

    uint64_t gpu_total  = 0;
    uint64_t host_total = 0;
    for (const auto & row : rows) {
        uint64_t self = row.data.model + row.data.context + row.data.compute + row.mmproj;
        if (row.data.is_gpu) gpu_total  += self;
        else                 host_total += self;
    }
    printf("\nTotal projected memory: GPU=%llu MiB  Host=%llu MiB\n",
           (unsigned long long)(gpu_total  / (1024*1024)),
           (unsigned long long)(host_total / (1024*1024)));

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
