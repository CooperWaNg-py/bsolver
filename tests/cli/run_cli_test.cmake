# Runs the bsolver binary once and checks its exit code and output.
#   cmake -DBSOLVER=<exe> -DWORK_DIR=<dir> -DARGS=<a|b|c> -DEXPECT_EXIT=<n>
#         [-DEXPECT_STDOUT=<regex>] [-DEXPECT_STDERR=<regex>] [-DEXPECT_FILE=<path relative to WORK_DIR>
#          -DEXPECT_FILE_REGEX=<regex>] -P run_cli_test.cmake
# ARGS separates arguments with '|' (a ';' list would be split by add_test). Placeholder <WORK_DIR> in ARGS expands
# to WORK_DIR. WORK_DIR is recreated empty and used as the working directory, so default output files stay inside.

foreach(var BSOLVER WORK_DIR ARGS EXPECT_EXIT)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "run_cli_test.cmake: ${var} is required")
  endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

string(REPLACE "<WORK_DIR>" "${WORK_DIR}" args "${ARGS}")
string(REPLACE "|" ";" args "${args}")

execute_process(
  COMMAND "${BSOLVER}" ${args}
  WORKING_DIRECTORY "${WORK_DIR}"
  RESULT_VARIABLE rc
  OUTPUT_VARIABLE out
  ERROR_VARIABLE err)

set(failures "")
if(NOT rc STREQUAL EXPECT_EXIT)
  string(APPEND failures "exit code ${rc}, expected ${EXPECT_EXIT}\n")
endif()
if(DEFINED EXPECT_STDOUT AND NOT out MATCHES "${EXPECT_STDOUT}")
  string(APPEND failures "stdout does not match '${EXPECT_STDOUT}'\n")
endif()
if(DEFINED EXPECT_STDERR AND NOT err MATCHES "${EXPECT_STDERR}")
  string(APPEND failures "stderr does not match '${EXPECT_STDERR}'\n")
endif()
if(DEFINED EXPECT_FILE)
  if(NOT EXISTS "${WORK_DIR}/${EXPECT_FILE}")
    string(APPEND failures "${EXPECT_FILE} was not written\n")
  else()
    file(READ "${WORK_DIR}/${EXPECT_FILE}" content)
    if(NOT content MATCHES "${EXPECT_FILE_REGEX}")
      string(APPEND failures "${EXPECT_FILE} does not match '${EXPECT_FILE_REGEX}'\n")
    endif()
  endif()
endif()

if(failures)
  message(FATAL_ERROR "bsolver ${args}\n${failures}--- stdout ---\n${out}\n--- stderr ---\n${err}")
endif()
file(REMOVE_RECURSE "${WORK_DIR}")
