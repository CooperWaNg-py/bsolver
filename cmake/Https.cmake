# HTTPS client for the btcpuzzle.info pool mode: WinHTTP on Windows, libcurl elsewhere.
# Produces BSOLVER_HTTPS_DEFINITIONS / BSOLVER_HTTPS_LIBRARIES.
set(BSOLVER_HTTPS_DEFINITIONS "")
set(BSOLVER_HTTPS_LIBRARIES "")

if(NOT BSOLVER_POOL)
  message(STATUS "bsolver: pool client disabled (BSOLVER_POOL=OFF)")
  return()
endif()

if(WIN32)
  set(BSOLVER_HTTPS_DEFINITIONS BSOLVER_HTTPS_WINHTTP=1)
  set(BSOLVER_HTTPS_LIBRARIES winhttp)
  message(STATUS "bsolver: pool client enabled (WinHTTP)")
else()
  find_package(CURL)
  if(CURL_FOUND)
    set(BSOLVER_HTTPS_DEFINITIONS BSOLVER_HTTPS_CURL=1)
    set(BSOLVER_HTTPS_LIBRARIES CURL::libcurl)
    message(STATUS "bsolver: pool client enabled (libcurl ${CURL_VERSION_STRING})")
  else()
    message(STATUS "bsolver: libcurl not found; pool client disabled (install libcurl development files)")
  endif()
endif()
