#include "inline-tools.h"

#include <iostream>
#include <stdexcept>
#include <string>

static void require(bool cond, const std::string & msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

static void test_no_match_flushes_safe_text() {
    inline_tool_interceptor it;
    it.push(1, 0, "hello ", nullptr);
    it.push(2, 1, "world", nullptr);

    require(it.poll().matched == false, "unexpected match");
    require(it.flush_safe_prefix() == "hello world", "safe text was not flushed");
    require(it.pending().empty(), "pending text not empty");
}

static bool has_suffix(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static void test_split_tool_call_matches() {
    inline_tool_interceptor it;
    it.push(1, 0, "<tool name=\"", nullptr);
    it.push(2, 1, "datetime\" param=\"", nullptr);
    it.push(3, 2, "abc\" />", nullptr);

    auto m = it.poll();
    require(m.matched, "split tool call did not match");
    require(m.token_begin == 0, "bad token_begin");
    require(m.token_end == 3, "bad token_end");
    require(m.pos_begin == 0, "bad pos_begin");
    require(has_suffix(m.replacement_text, "abc"), "datetime param was not appended: " + m.replacement_text);
    require(m.replacement_text.find("<tool") == std::string::npos, "raw tool text leaked");
}

static void test_boundary_prefix_suffix_preserved() {
    inline_tool_interceptor it;
    it.push(1, 7, "foo<tool", nullptr);
    it.push(2, 8, " name=\"datetime\" param=\"abc\" />bar", nullptr);

    auto m = it.poll();
    require(m.matched, "boundary tool call did not match");
    require(m.token_begin == 0, "bad boundary token_begin");
    require(m.token_end == 2, "bad boundary token_end");
    require(m.pos_begin == 7, "bad boundary pos_begin");
    require(m.replacement_text.rfind("foo", 0) == 0, "boundary prefix not preserved: " + m.replacement_text);
    require(has_suffix(m.replacement_text, "abcbar"), "boundary suffix/param not preserved: " + m.replacement_text);
}

static void test_unknown_tool() {
    inline_tool_interceptor it;
    it.push(1, 0, "<tool name=\"nope\" param=\"x\">", nullptr);

    auto m = it.poll();
    require(m.matched, "unknown tool call did not match");
    require(m.replacement_text == "<tool-error unknown tool>", "bad unknown replacement");
}

static void test_sanitize_replacement() {
    require(inline_tool_sanitize_replacement("a <tool b") == "a &lt;tool b", "replacement not sanitized");
}

static void test_pending_overflow_flushes_malformed() {
    inline_tool_interceptor it(8);
    it.push(1, 0, "abc<", nullptr);
    it.push(2, 1, "defg", nullptr);
    it.push(3, 2, "hijk", nullptr);

    const std::string out = it.flush_safe_prefix();
    require(!out.empty(), "overflow did not flush malformed text");
    require(it.pending().size() <= 8, "pending overflow not bounded");
}

int main() {
    test_no_match_flushes_safe_text();
    test_split_tool_call_matches();
    test_boundary_prefix_suffix_preserved();
    test_unknown_tool();
    test_sanitize_replacement();
    test_pending_overflow_flushes_malformed();

    std::cout << "OK\n";
    return 0;
}
