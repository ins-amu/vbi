cmake_minimum_required(VERSION 3.16)
project(${module_name} CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "${cxx_flags_release}")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# pybind11 - try cmake package first, then fall back to python path
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND "${python_executable}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _PB11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    find_package(pybind11 CONFIG REQUIRED PATHS "${'${_PB11_DIR}'}")
endif()

find_package(OpenMP)

pybind11_add_module(${module_name}
    ${bindings_cpp_filename}
)

target_include_directories(${module_name} PRIVATE .)

if(OpenMP_CXX_FOUND)
    target_link_libraries(${module_name} PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS ${module_name} LIBRARY DESTINATION .)
