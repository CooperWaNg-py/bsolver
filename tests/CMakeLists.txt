file(GLOB BSOLVER_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_executable(bsolver_tests ${BSOLVER_TEST_SOURCES})
target_link_libraries(bsolver_tests PRIVATE bsolver_core Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(bsolver_tests DISCOVERY_MODE PRE_TEST)

# CLI smoke tests: drive the real bsolver binary through tests/cli/run_cli_test.cmake (exit code + output checks).
# ARGS separates arguments with '|'; <WORK_DIR> expands to the test's scratch directory.
if(TARGET bsolver)
  function(bsolver_cli_test name)
    cmake_parse_arguments(PARSE_ARGV 1 T "" "ARGS;EXIT;STDOUT;STDERR;FILE;FILE_REGEX" "")
    set(defs -DBSOLVER=$<TARGET_FILE:bsolver> "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/cli/${name}" "-DARGS=${T_ARGS}"
             -DEXPECT_EXIT=${T_EXIT})
    if(DEFINED T_STDOUT)
      list(APPEND defs "-DEXPECT_STDOUT=${T_STDOUT}")
    endif()
    if(DEFINED T_STDERR)
      list(APPEND defs "-DEXPECT_STDERR=${T_STDERR}")
    endif()
    if(DEFINED T_FILE)
      list(APPEND defs "-DEXPECT_FILE=${T_FILE}" "-DEXPECT_FILE_REGEX=${T_FILE_REGEX}")
    endif()
    add_test(NAME cli.${name} COMMAND ${CMAKE_COMMAND} ${defs} -P ${CMAKE_CURRENT_SOURCE_DIR}/cli/run_cli_test.cmake)
    set_tests_properties(cli.${name} PROPERTIES TIMEOUT 60 LABELS cli)
  endfunction()

  bsolver_cli_test(puzzles_row
    ARGS "puzzles|--id|71"
    EXIT 0
    STDOUT "71 +71 +1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU +no +unsolved +-")
  bsolver_cli_test(run_found
    ARGS "run|--puzzle|20|--backend|cpu|--threads|2|--no-checkpoint|--results|<WORK_DIR>/found.jsonl"
    EXIT 0
    STDOUT "FOUND puzzle #20 address 1HsMJxNiV7TLxmoF6uJNkydxPFDog4NQum privkey 0+d2c55 WIF K"
    FILE found.jsonl
    FILE_REGEX "\"privkey\":\"0+d2c55\"")
  bsolver_cli_test(run_exhausted
    ARGS "run|--puzzle|20|--range|80000:c0000|--backend|cpu|--threads|2|--no-checkpoint|--results|<WORK_DIR>/found.jsonl"
    EXIT 2
    STDERR "range exhausted without finding the key")
  bsolver_cli_test(run_range_address
    ARGS "run|--address|1HsMJxNiV7TLxmoF6uJNkydxPFDog4NQum|--range|0xd0000:d3fff|--backend|cpu|--threads|2|--no-checkpoint|--results|<WORK_DIR>/found.jsonl"
    EXIT 0
    STDOUT "privkey 0+d2c55")
  bsolver_cli_test(run_range_malformed
    ARGS "run|--puzzle|20|--range|80000|--no-checkpoint"
    EXIT 1
    STDERR "--range must be START:END \\(hex\\)")
  bsolver_cli_test(run_range_reversed
    ARGS "run|--puzzle|20|--range|c0000:80000|--no-checkpoint"
    EXIT 1
    STDERR "range end must be >= range start")
  bsolver_cli_test(bench_seconds_range
    ARGS "bench|--seconds|0.05"
    EXIT 105
    STDERR "--seconds: Value 0.05 not in range")
endif()
