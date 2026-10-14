# Builds webui/ with npm and embeds the output as kWebAssets. Produces BSOLVER_WEBUI_ASSETS_CPP.
set(BSOLVER_WEBUI_ASSETS_CPP ${CMAKE_BINARY_DIR}/generated/web_assets.cpp)

if(NOT BSOLVER_WEBUI)
  bsolver_embed(OUTPUT ${BSOLVER_WEBUI_ASSETS_CPP} SYMBOL WebAssets MODE binary)
  return()
endif()

find_program(NPM NAMES npm.cmd npm)
if(NOT NPM)
  message(FATAL_ERROR "npm not found: install Node.js >= 20 or configure with -DBSOLVER_WEBUI=OFF")
endif()

set(WEBUI_DIR ${CMAKE_SOURCE_DIR}/webui)
set(WEBUI_OUT ${CMAKE_BINARY_DIR}/webui-dist)
# Written by `npm ci`; inside node_modules, so deleting node_modules reruns the install.
set(WEBUI_CI_STAMP ${WEBUI_DIR}/node_modules/.package-lock.json)
set(WEBUI_BUILD_STAMP ${CMAKE_BINARY_DIR}/webui-build.stamp)

add_custom_command(
  OUTPUT ${WEBUI_CI_STAMP}
  COMMAND ${NPM} ci --no-audit --no-fund
  COMMAND ${CMAKE_COMMAND} -E touch ${WEBUI_CI_STAMP}
  WORKING_DIRECTORY ${WEBUI_DIR}
  DEPENDS ${WEBUI_DIR}/package-lock.json
  COMMENT "Installing web UI dependencies (npm ci)"
  VERBATIM)

file(GLOB_RECURSE WEBUI_SOURCES CONFIGURE_DEPENDS
  ${WEBUI_DIR}/src/*
  ${WEBUI_DIR}/index.html
  ${WEBUI_DIR}/vite.config.ts
  ${WEBUI_DIR}/tsconfig.json
  ${WEBUI_DIR}/package.json)

add_custom_command(
  OUTPUT ${WEBUI_BUILD_STAMP}
  COMMAND ${CMAKE_COMMAND} -E env BSOLVER_WEBUI_OUT=${WEBUI_OUT} ${NPM} run build
  COMMAND ${CMAKE_COMMAND} -E touch ${WEBUI_BUILD_STAMP}
  WORKING_DIRECTORY ${WEBUI_DIR}
  DEPENDS ${WEBUI_CI_STAMP} ${WEBUI_SOURCES}
  COMMENT "Building web UI (npm run build)"
  VERBATIM)

bsolver_embed(OUTPUT ${BSOLVER_WEBUI_ASSETS_CPP} SYMBOL WebAssets MODE binary
  DIR ${WEBUI_OUT} DEPENDS ${WEBUI_BUILD_STAMP})
