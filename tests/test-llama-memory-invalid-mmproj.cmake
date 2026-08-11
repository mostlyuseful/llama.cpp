execute_process(
    COMMAND "${LLAMA_MEMORY}" -m "${MODEL}" --mmproj "${MMPROJ}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr)

set(output "${stdout}${stderr}")
if (result EQUAL 0)
    message(FATAL_ERROR "llama-memory accepted an invalid mmproj:\n${output}")
endif()
if (NOT output MATCHES "error: failed to estimate mmproj memory")
    message(FATAL_ERROR "llama-memory did not report an mmproj estimation error:\n${output}")
endif()
if (output MATCHES "Total projected memory")
    message(FATAL_ERROR "llama-memory printed an incomplete combined total:\n${output}")
endif()
