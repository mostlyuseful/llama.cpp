#pragma once

#include "common.h"
#include "sampling.h"

#include <cstddef>
#include <deque>
#include <string>

struct inline_tool_call {
    std::string name;
    std::string param;
};

struct inline_tool_token_span {
    llama_token token = LLAMA_TOKEN_NULL;
    llama_pos   pos   = 0;

    size_t byte_begin = 0;
    size_t byte_end   = 0;

    // Snapshot before accepting this generated token.
    common_sampler_ptr sampler_before;
};

struct inline_tool_match {
    bool matched = false;

    size_t byte_begin = 0;
    size_t byte_end   = 0;

    size_t token_begin = 0;
    size_t token_end   = 0;

    llama_pos pos_begin = 0;

    std::string replacement_text;

    common_sampler_ptr sampler_before;
};

std::string inline_tool_execute(const inline_tool_call & call);
std::string inline_tool_sanitize_replacement(std::string text);

class inline_tool_interceptor {
public:
    explicit inline_tool_interceptor(size_t max_pending_bytes = 2048);

    void push(llama_token token, llama_pos pos, const std::string & piece, common_sampler_ptr sampler_before);

    inline_tool_match poll();

    std::string flush_safe_prefix();

    void clear_after_match();
    void clear();

    const std::string & pending() const { return pending_text; }
    size_t pending_token_count() const { return pending_tokens.size(); }

private:
    std::string pending_text;
    std::deque<inline_tool_token_span> pending_tokens;
    size_t max_pending_bytes = 2048;

    std::string flush_bytes(size_t byte_end);
};
