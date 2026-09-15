cmake_minimum_required(VERSION 3.20)

project(credit_default_cpp_knn LANGUAGES CXX)

option(CPP_KNN_NATIVE "Enable native-CPU optimizations for local benchmarking" OFF)

add_library(cpp_knn_core
    src/cpp_knn/data_io.cpp
    src/cpp_knn/knn.cpp
)
target_include_directories(cpp_knn_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(cpp_knn_core PUBLIC cxx_std_20)

add_executable(cpp_knn src/cpp_knn/main.cpp)
target_link_libraries(cpp_knn PRIVATE cpp_knn_core)
target_compile_features(cpp_knn PRIVATE cxx_std_20)

if(MSVC)
    target_compile_options(cpp_knn_core PRIVATE /W4 /permissive- /fp:precise)
    target_compile_options(cpp_knn PRIVATE /W4 /permissive- /fp:precise)
    if(CPP_KNN_NATIVE)
        target_compile_options(cpp_knn_core PRIVATE /O2 /arch:AVX2)
        target_compile_options(cpp_knn PRIVATE /O2 /arch:AVX2)
    endif()
else()
    target_compile_options(cpp_knn_core PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
    target_compile_options(cpp_knn PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
    if(CPP_KNN_NATIVE)
        target_compile_options(cpp_knn_core PRIVATE -O3 -march=native)
        target_compile_options(cpp_knn PRIVATE -O3 -march=native)
    endif()
endif()

if(CPP_KNN_NATIVE)
    set_property(TARGET cpp_knn_core cpp_knn PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    target_compile_definitions(cpp_knn_core PUBLIC CPP_KNN_NATIVE_BUILD=1)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(cpp_knn_tests tests/cpp_knn_tests.cpp)
    target_link_libraries(cpp_knn_tests PRIVATE cpp_knn_core)
    target_compile_features(cpp_knn_tests PRIVATE cxx_std_20)
    if(MSVC)
        target_compile_options(cpp_knn_tests PRIVATE /W4 /permissive- /fp:precise)
    else()
        target_compile_options(cpp_knn_tests PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
    endif()
    add_test(NAME cpp_knn_tests COMMAND cpp_knn_tests)
endif()
