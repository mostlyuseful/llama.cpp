#include "inline-tools.h"

#include <algorithm>
#include <ctime>
#include <regex>
#include <stdexcept>

static const std::regex INLINE_TOOL_RE(
    R"INLINE(<tool\s+name="([A-Za-z0-9_-]{1,64})"\s+param="([^"]{0,512})"\s*/?>)INLINE",
    std::regex::ECMAScript
);

std::string inline_tool_sanitize_replacement(std::string text) {
    string_replace_all(text, "<tool ", "&lt;tool ");
    string_replace_all(text, "<tool>", "&lt;tool>");
    return text;
}

std::string inline_tool_execute(const inline_tool_call & call) {
    if (call.name != "datetime") {
        return "<tool-error unknown tool>";
    }

    std::time_t now = std::time(nullptr);
    std::tm tm;
#if defined(_WIN32)
    localtime_s(&tm, &now);
#else
    localtime_r(&now, &tm);
#endif

    char buf[64];
    if (std::strftime(buf, sizeof(buf), "%a %b %d %H:%M:%S %Z %Y", &tm) == 0) {
        return "<tool-error datetime unavailable>";
    }

    return std::string(buf) + call.param;
}

inline_tool_interceptor::inline_tool_interceptor(size_t max_pending_bytes) :
    max_pending_bytes(max_pending_bytes) {
}

void inline_tool_interceptor::push(llama_token token, llama_pos pos, const std::string & piece, common_sampler_ptr sampler_before) {
    const size_t begin = pending_text.size();
    pending_text += piece;
    const size_t end = pending_text.size();

    pending_tokens.push_back({
        /* .token          = */ token,
        /* .pos            = */ pos,
        /* .byte_begin     = */ begin,
        /* .byte_end       = */ end,
        /* .sampler_before = */ std::move(sampler_before),
    });
}

inline_tool_match inline_tool_interceptor::poll() {
    std::smatch match;
    if (!std::regex_search(pending_text, match, INLINE_TOOL_RE)) {
        return {};
    }

    const size_t byte_begin = (size_t) match.position(0);
    const size_t byte_end   = byte_begin + (size_t) match.length(0);

    if (byte_end == byte_begin || pending_tokens.empty()) {
        throw std::runtime_error("inline tools: invalid empty regex match");
    }

    const size_t npos = (size_t) -1;
    size_t token_begin = npos;
    size_t token_end   = npos;

    for (size_t i = 0; i < pending_tokens.size(); ++i) {
        const auto & span = pending_tokens[i];
        if (span.byte_begin <= byte_begin && byte_begin < span.byte_end) {
            token_begin = i;
        }
        if (span.byte_begin < byte_end && byte_end - 1 < span.byte_end) {
            token_end = i + 1;
            break;
        }
    }

    if (token_begin == npos || token_end == npos) {
        throw std::runtime_error("inline tools: regex match does not map to token spans");
    }

    const auto & first = pending_tokens[token_begin];
    const auto & last  = pending_tokens[token_end - 1];

    const std::string prefix = pending_text.substr(first.byte_begin, byte_begin - first.byte_begin);
    const std::string suffix = pending_text.substr(byte_end, last.byte_end - byte_end);

    inline_tool_call call {
        /* .name  = */ match.str(1),
        /* .param = */ match.str(2),
    };

    inline_tool_match result;
    result.matched          = true;
    result.byte_begin       = byte_begin;
    result.byte_end         = byte_end;
    result.token_begin      = token_begin;
    result.token_end        = token_end;
    result.pos_begin        = first.pos;
    result.replacement_text = prefix + inline_tool_sanitize_replacement(inline_tool_execute(call)) + suffix;
    result.sampler_before   = std::move(pending_tokens[token_begin].sampler_before);

    return result;
}

std::string inline_tool_interceptor::flush_bytes(size_t byte_end) {
    if (byte_end == 0) {
        return {};
    }

    byte_end = std::min(byte_end, pending_text.size());

    size_t token_count = 0;
    while (token_count < pending_tokens.size() && pending_tokens[token_count].byte_end <= byte_end) {
        token_count++;
    }

    if (token_count == 0) {
        return {};
    }

    byte_end = pending_tokens[token_count - 1].byte_end;
    std::string out = pending_text.substr(0, byte_end);
    pending_text.erase(0, byte_end);

    for (size_t i = 0; i < token_count; ++i) {
        pending_tokens.pop_front();
    }

    for (auto & span : pending_tokens) {
        span.byte_begin -= byte_end;
        span.byte_end   -= byte_end;
    }

    return out;
}

std::string inline_tool_interceptor::flush_safe_prefix() {
    if (pending_text.empty()) {
        return {};
    }

    const size_t last_lt = pending_text.rfind('<');
    if (last_lt == std::string::npos) {
        return flush_bytes(pending_text.size());
    }

    std::string out = flush_bytes(last_lt);

    if (pending_text.size() > max_pending_bytes) {
        const size_t overflow = pending_text.size() - max_pending_bytes;
        out += flush_bytes(overflow);
    }

    return out;
}

void inline_tool_interceptor::clear_after_match() {
    clear();
}

void inline_tool_interceptor::clear() {
    pending_text.clear();
    pending_tokens.clear();
}
