file(GLOB BSOLVER_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_executable(bsolver_tests ${BSOLVER_TEST_SOURCES})
target_link_libraries(bsolver_tests PRIVATE bsolver_core Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(bsolver_tests DISCOVERY_MODE PRE_TEST)
