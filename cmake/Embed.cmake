# bsolver_embed(OUTPUT <cpp> SYMBOL <name> MODE <binary|text> [DIR <dir>] [FILES <f...>] [DEPENDS <...>])
#
# binary + DIR : table `const Asset k<SYMBOL>[]` / `k<SYMBOL>Count` of every file under DIR
# binary alone : empty table (count 0)
# text + FILES : files concatenated into `const std::string_view k<SYMBOL>` (NUL-terminated)
set(BSOLVER_EMBED_SCRIPT ${CMAKE_CURRENT_LIST_DIR}/embed_files.cmake)

function(bsolver_embed)
  cmake_parse_arguments(E "" "OUTPUT;SYMBOL;MODE;DIR" "FILES;DEPENDS" ${ARGN})
  if(NOT E_OUTPUT OR NOT E_SYMBOL OR NOT E_MODE)
    message(FATAL_ERROR "bsolver_embed: OUTPUT, SYMBOL and MODE are required")
  endif()
  string(REPLACE ";" "|" files_arg "${E_FILES}")
  add_custom_command(
    OUTPUT ${E_OUTPUT}
    COMMAND ${CMAKE_COMMAND}
      "-DOUTPUT=${E_OUTPUT}" "-DSYMBOL=${E_SYMBOL}" "-DMODE=${E_MODE}"
      "-DDIR=${E_DIR}" "-DFILES=${files_arg}"
      -P ${BSOLVER_EMBED_SCRIPT}
    DEPENDS ${BSOLVER_EMBED_SCRIPT} ${E_FILES} ${E_DEPENDS}
    COMMENT "Embedding ${E_SYMBOL}"
    VERBATIM)
endfunction()
