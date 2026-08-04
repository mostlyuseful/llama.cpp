#include "llama.h"
#include "arg.h"
#include "common.h"
#include "log.h"

#include <cstdio>
#include <cstring>
#include <vector>

static void silent_log(ggml_log_level, const char *, void *) {}

int main(int argc, char ** argv) {
    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
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
        fprintf(stderr, "error: failed to load model '%s'\n", params.model.path.c_str());
        return 1;
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        fprintf(stderr, "error: failed to create context\n");
        return 1;
    }

    llama_log_set(old_log_cb, old_log_data);

    constexpr size_t MAX_DEVS = 64;
    llama_memory_breakdown bk[MAX_DEVS];
    size_t n = MAX_DEVS;
    llama_memory_breakdown_get(ctx, bk, &n);

    for (size_t i = 0; i < n; i++) {
        const auto & d = bk[i];
        uint64_t self = d.model + d.context + d.compute;
        if (d.is_gpu) {
            printf("%-30s  total=%6llu MiB  free=%6llu MiB  "
                   "model=%6llu MiB  kv=%6llu MiB  compute=%6llu MiB\n",
                   d.name,
                   (unsigned long long)(d.total   / (1024*1024)),
                   (unsigned long long)(d.free    / (1024*1024)),
                   (unsigned long long)(d.model   / (1024*1024)),
                   (unsigned long long)(d.context / (1024*1024)),
                   (unsigned long long)(d.compute / (1024*1024)));
        } else {
            printf("%-30s  self=%6llu MiB  "
                   "model=%6llu MiB  kv=%6llu MiB  compute=%6llu MiB\n",
                   d.name,
                   (unsigned long long)(self       / (1024*1024)),
                   (unsigned long long)(d.model   / (1024*1024)),
                   (unsigned long long)(d.context / (1024*1024)),
                   (unsigned long long)(d.compute / (1024*1024)));
        }
    }

    uint64_t gpu_total  = 0;
    uint64_t host_total = 0;
    for (size_t i = 0; i < n; i++) {
        uint64_t self = bk[i].model + bk[i].context + bk[i].compute;
        if (bk[i].is_gpu) gpu_total  += self;
        else               host_total += self;
    }
    printf("\nTotal projected memory: GPU=%llu MiB  Host=%llu MiB\n",
           (unsigned long long)(gpu_total  / (1024*1024)),
           (unsigned long long)(host_total / (1024*1024)));

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}