# llama-memory

A tool that reports the projected memory required to load a specific GGUF model at a given context size, KV-cache types, and GPU layer count — **without actually allocating any device memory**.

## Goal

```
./llama-memory -m ~/models/qwen3.5-35b-q4-k-l.gguf -c 20000 -ctk f16 -ctv q8_0
→ 17900 MiB  (GPU)  |  58271 MiB  (Host)
```

---

## How llama.cpp Computes Projected Memory

### The Three Memory Components

Every context is broken into three buckets (`struct llama_memory_breakdown_data`, `src/llama-context.h:26`):

```cpp
struct llama_memory_breakdown_data {
    size_t model   = 0;  // weight tensors (gguf buffers)
    size_t context = 0;  // KV-cache tensors
    size_t compute = 0;  // worst-case compute graph buffers (activations)
    size_t total() const { return model + context + compute; }
};
```

These are tracked per `ggml_backend_buffer_type_t` (one type per device + host), so the caller can distinguish GPU vs CPU memory.

---

### The Dry-Run Trick (`no_alloc = true`)

`llama_model_params.no_alloc` is the key flag (field visible in `src/llama-model.cpp:850`).

When set:
- `llama_model_load_from_file()` reads only metadata and *simulates* tensor allocation via `ggml_backend_alloc_ctx_tensors_from_buft_size()` instead of actually reserving VRAM/RAM.
- `llama_init_from_model()` follows suit: the KV-cache layers are tensor-graphed but not materialized, and `sched_reserve()` calls `ggml_backend_sched_reserve_size()` (dry-run) instead of `ggml_backend_sched_reserve()` (real alloc).

**Net effect**: a full model + context load in ~1 second with zero VRAM consumption, but with accurate size projections stored in `backend_buf_exp_size[]`.

---

### The Core Internal Helper: `llama_get_device_memory_data()`

**File**: `src/llama.cpp:55–142` (static, not in public API)

This function is the heart of `llama_params_fit`. Its logic:

```cpp
static std::vector<llama_device_memory_data>
llama_get_device_memory_data(path_model, mparams, cparams, devs, ...) {
    mparams_copy.no_alloc  = true;   // ← dry-run flag
    mparams_copy.use_mmap  = false;
    mparams_copy.use_mlock = false;

    llama_model * model = llama_model_load_from_file(path_model, mparams_copy);
    llama_context * ctx = llama_init_from_model(model, *cparams);
    // ↑ cparams carries n_ctx, type_k, type_v, n_seq_max, etc.

    // Collect per-buffer-type breakdown:
    auto memory_breakdown = ctx->memory_breakdown();

    // Map buffer types to devices:
    for (auto & [buft, mb] : memory_breakdown) {
        if (!ggml_backend_buft_is_host(buft)) {
            ret[device_index].mb.model   += mb.model;
            ret[device_index].mb.context += mb.context;
            ret[device_index].mb.compute += mb.compute;
        }
    }
    // Query physical free/total from the device driver:
    ggml_backend_dev_memory(model->devices[i].dev, &free, &total);
    ret[i].free  = free;
    ret[i].total = total;

    llama_free(ctx);
    llama_model_free(model);
    return ret;
}
```

---

### How `ctx->memory_breakdown()` Assembles All Three Components

**File**: `src/llama-context.cpp:2639`

```cpp
std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>
llama_context::memory_breakdown() const {

    // 1. MODEL WEIGHTS
    for (auto & [buft, size] : model.memory_breakdown())   // llama-model.cpp:8081
        ret[buft].model += size;                           // ggml_backend_alloc_ctx_tensors_from_buft_size()

    // 2. KV CACHE (context memory)
    if (memory)
        for (auto & [buft, size] : memory->memory_breakdown())  // llama-kv-cache.cpp:609
            ret[buft].context += size;                          // same dry-run size function

    // 3. COMPUTE BUFFERS
    if (model.hparams.no_alloc)
        // no_alloc path: populated by sched_reserve() → ggml_backend_sched_reserve_size()
        ret[buft].compute += backend_buf_exp_size[i];
    else
        // real path: populated after sched_reserve() → ggml_backend_sched_get_buffer_size()
        ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend);
    ...
}
```

**KV cache memory** is computed in `llama_kv_cache::memory_breakdown()` (`src/llama-kv-cache.cpp:609`):
- `n_ctx × n_heads_kv × n_embd_head × n_layers` × `(ggml_type_size(type_k) + ggml_type_size(type_v))`
- `type_k` and `type_v` come directly from `cparams.type_k` / `cparams.type_v` → set by `-ctk`/`-ctv`

**Compute buffers** are estimated via `llama_context::sched_reserve()` (`src/llama-context.cpp:400–620`):
- Builds a worst-case compute graph for `n_tokens = min(n_ctx, n_ubatch)` with `n_seqs` sequences
- Calls `ggml_backend_sched_reserve_size()` on that graph
- Stores estimated sizes per backend in `backend_buf_exp_size[]`
- Does **not** perform actual graph execution

---

### What `llama_params_fit` Reports

`llama_params_fit_impl()` calls `llama_get_device_memory_data()` and then logs:

```
llama_params_fit_impl: projected to use 19187 MiB of device memory vs. 24077 MiB of free device memory
```

The `19187 MiB` is `sum(dmd.mb.total())` across all GPU devices.
The breakdown visible from `llama_memory_breakdown_print()` (called inside `llama_get_device_memory_data` at line 140):

```
| CUDA0 (RTX 4090) | 24077 = 945 + (19187 = 17904 + 384 + 898) + 3945 |
                               free    total  model   kv  compute  unaccounted
```

---

## The Public API Gap

There is **no public function that returns numeric memory breakdown values**. The only relevant public entry points are:

| Function | File | What it does |
|---|---|---|
| `llama_params_fit()` | `include/llama.h:525` | Adjusts params to fit free memory; calls `llama_get_device_memory_data()` internally |
| `llama_memory_breakdown_print()` | `include/llama.h:1550` | Logs breakdown via `LLAMA_LOG_INFO` to stderr |
| `llama_model_size()` | `include/llama.h:617` | Returns raw model bytes (no KV/compute) |
| `ggml_backend_dev_memory()` | `ggml/include/ggml-backend.h:181` | Returns device free/total (no breakdown) |

`llama_memory_breakdown_print()` calls `LLAMA_LOG_INFO`, which goes to stderr through the registered log callback. There is no equivalent function that returns the numbers as a struct.

---

## Implementation Plan

### Step 1 — Add a new public API function to expose breakdown numerically

The cleanest solution is to add one new function to the public API (analogous to `llama_memory_breakdown_print` but returning structured data).

**`include/llama.h`** — add after `llama_memory_breakdown_print` (line 1550):

```c
// Per-device memory breakdown result
struct llama_memory_breakdown {
    char     name[128];   // device name, e.g. "CUDA0 (RTX 4090)" or "Host"
    uint64_t model;       // model weight bytes on this device
    uint64_t context;     // KV-cache bytes on this device
    uint64_t compute;     // compute buffer bytes on this device
    uint64_t free;        // device free memory (0 for host)
    uint64_t total;       // device total memory (0 for host)
    bool     is_gpu;      // true if dedicated GPU memory
};

// Fills `out` (caller-allocated, `*n_out` capacity) with per-device breakdown.
// Sets *n_out to the number of entries written (GPU devices + 1 host entry).
// Requires a fully-initialized llama_context (created with no_alloc=true is fine).
LLAMA_API void llama_memory_breakdown_get(
    const struct llama_context        * ctx,
          struct llama_memory_breakdown * out,
          size_t                        * n_out);
```

**`src/llama-context.cpp`** — implement it by reusing `ctx->memory_breakdown()` (same logic as `llama_memory_breakdown_print`, lines 3496–3638, but writing structs instead of log strings):

```cpp
void llama_memory_breakdown_get(const llama_context * ctx,
                                llama_memory_breakdown * out,
                                size_t * n_out) {
    const auto & devices = ctx->get_model().devices;
    auto breakdown = ctx->memory_breakdown();

    size_t idx = 0;

    // GPU devices
    for (size_t i = 0; i < devices.size() && idx < *n_out; i++) {
        // accumulate mb_dev[i] from breakdown map (same loop as breakdown_print)
        // call ggml_backend_dev_memory() for free/total
        // fill out[idx++] with is_gpu=true
    }

    // Host entry
    // accumulate mb_host from breakdown map
    // fill out[idx++] with is_gpu=false, free=0, total=0

    *n_out = idx;
}
```

> **Alternative without API changes**: install a custom `llama_log_set()` callback that captures `LLAMA_LOG_INFO` lines into a string buffer, call `llama_memory_breakdown_print()`, then parse the table. This avoids touching the public API but is fragile if the log format changes.

The implementation body is a straightforward refactor of `llama_memory_breakdown_print()`. Extract the accumulation loop into a shared private helper, then call it from both functions:

```
llama_memory_breakdown_print()  ─┐
                                  ├── llama_memory_breakdown_collect()  [new private helper]
llama_memory_breakdown_get()    ─┘        └── iterates ctx->memory_breakdown()
                                               └── calls ggml_backend_dev_memory()
```

This keeps the two public functions in sync without duplicating logic.

---

### Step 2 — Create `tools/llama-memory/`

#### `tools/llama-memory/CMakeLists.txt`

```cmake
set(TARGET llama-memory)
add_executable(${TARGET} llama-memory.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
```

#### Register in `tools/CMakeLists.txt`

Add alongside `fit-params`:

```cmake
add_subdirectory(llama-memory)
```

---

### Step 3 — Write `tools/llama-memory/llama-memory.cpp`

```cpp
#include "llama.h"
#include "arg.h"       // common_params_parse, LLAMA_EXAMPLE_COMMON
#include "common.h"    // common_model_params_to_llama, common_context_params_to_llama
#include "log.h"

#include <cstdio>
#include <cstring>
#include <vector>

// Silence all log output during the dry-run load
static void silent_log(ggml_log_level, const char *, void *) {}

int main(int argc, char ** argv) {
    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    // 1. Build model/context params (respects -m, -c, -ctk, -ctv, -ngl, -ts, -ot etc.)
    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    // 2. Dry-run: load metadata only, simulate all allocations
    mparams.no_alloc  = true;
    mparams.use_mmap  = false;
    mparams.use_mlock = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    // Suppress llama.cpp internal log noise during the dry-run load
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

    // 3. Restore log output
    common_init();  // re-registers the default common log callback

    // 4. Collect breakdown
    constexpr size_t MAX_DEVS = 64;
    llama_memory_breakdown bk[MAX_DEVS];
    size_t n = MAX_DEVS;
    llama_memory_breakdown_get(ctx, bk, &n);  // new API from Step 1

    // 5. Print results — detailed breakdown per device
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

    // Single-line summary: total projected GPU and Host memory needed
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
```

---

## File Map (all files to touch)

| File | Change |
|---|---|
| `include/llama.h` | Add `llama_memory_breakdown` struct + `llama_memory_breakdown_get()` declaration |
| `src/llama-context.h` | No change (internal `llama_memory_breakdown_data` stays private) |
| `src/llama-context.cpp` | Implement `llama_memory_breakdown_get()`; refactor `llama_memory_breakdown_print()` to share the accumulation helper |
| `tools/llama-memory/llama-memory.cpp` | New tool (~80 lines) |
| `tools/llama-memory/CMakeLists.txt` | New CMake target (modelled after `tools/fit-params/CMakeLists.txt`) |
| `tools/CMakeLists.txt` | Add `add_subdirectory(llama-memory)` |

---

## Key Parameters That Drive Memory Usage

| CLI flag | `common_params` field | `cparams`/`mparams` field | Effect on memory |
|---|---|---|---|
| `-c N` | `params.n_ctx` | `cparams.n_ctx` | KV cache size: O(N × layers × heads × head_dim) |
| `-ctk TYPE` | `params.cache_type_k` | `cparams.type_k` | K cache element type (f16=2B, q8_0=1B, q4_0=0.5B …) |
| `-ctv TYPE` | `params.cache_type_v` | `cparams.type_v` | V cache element type (same range) |
| `-ngl N` | `params.n_gpu_layers` | `mparams.n_gpu_layers` | How many layers land on GPU (rest → host RAM) |
| `-ub N` | `params.n_ubatch` | `cparams.n_ubatch` | Compute buffer size: O(N × hidden_dim) |
| `-np N` | `params.n_parallel` | `cparams.n_seq_max` | Parallel sequences → multiplies KV cache |
| `-ts ...` | `params.tensor_split` | `mparams.tensor_split` | Multi-GPU weight distribution |
| `-ot ...` | `params.tensor_buft_overrides` | `mparams.tensor_buft_overrides` | Per-tensor backend overrides (e.g. MoE experts → CPU) |

---

## Example Output

```
$ ./llama-memory -m ~/models/qwen3.5-35b-q4-k-l.gguf -c 20000 -ctk f16 -ctv q8_0 -ngl 99

CUDA0 (RTX 4090)               total= 24077 MiB  free=  4890 MiB  model= 17904 MiB  kv=   682 MiB  compute=  898 MiB
Host                           self=  58271 MiB  model= 58259 MiB  kv=     0 MiB  compute=   12 MiB

Total projected memory: GPU=19484 MiB  Host=58271 MiB
```

*(Exact numbers depend on quantisation, layer count, head dimensions of the specific model.)*

---

## Full Call Chain (reference)

```
llama-memory main()
 ├─ common_params_parse()           [common/arg.cpp]
 ├─ common_model_params_to_llama()  [common/common.cpp:1412]
 ├─ common_context_params_to_llama()[common/common.cpp:1451]
 │    └─ sets cparams.type_k, .type_v, .n_ctx, .n_seq_max …
 ├─ mparams.no_alloc = true         ← DRY-RUN flag
 ├─ llama_model_load_from_file()    [src/llama-model.cpp]
 │    └─ loads hparams, skips real tensor allocation
 │       uses ggml_backend_alloc_ctx_tensors_from_buft_size() to MEASURE sizes
 ├─ llama_init_from_model()         [src/llama-context.cpp:~240]
 │    ├─ llama_kv_cache ctor        [src/llama-kv-cache.cpp:86]
 │    │    └─ creates KV tensors proportional to n_ctx × type_k/v × n_heads × n_layers
 │    └─ sched_reserve()            [src/llama-context.cpp:400]
 │         └─ ggml_backend_sched_reserve_size() → writes backend_buf_exp_size[]
 ├─ llama_memory_breakdown_get()    [NEW: src/llama-context.cpp]
 │    └─ ctx->memory_breakdown()    [src/llama-context.cpp:2639]
 │         ├─ model.memory_breakdown()   → mb.model per buft
 │         ├─ memory->memory_breakdown() → mb.context per buft  (KV cache)
 │         └─ backend_buf_exp_size[]     → mb.compute per buft  (compute graphs)
 └─ ggml_backend_dev_memory()       [ggml/include/ggml-backend.h:181]
      └─ queries driver for physical free/total per device
```
