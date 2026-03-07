cmake_minimum_required(VERSION 3.14)
project(kudio_dsp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler optimisation flags
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /arch:AVX2 /fp:fast /GL")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math")
endif()

# Auto-detect Python
# Priority:
#   1. -DPython3_EXECUTABLE=... on cmake command line  (explicit override)
#   2. KUDIO_PYTHON environment variable
#   3. 'python' on PATH  (CMake default)
#
# Examples:
#   cmake ..                                             <- uses PATH python
#   cmake .. -DPython3_EXECUTABLE=C:\python3-14\python.exe
#   set KUDIO_PYTHON=C:\python3-14\python.exe && cmake ..

if(NOT Python3_EXECUTABLE)
    if(DEFINED ENV{KUDIO_PYTHON})
        set(Python3_EXECUTABLE "$ENV{KUDIO_PYTHON}")
        message(STATUS "Using KUDIO_PYTHON env: ${Python3_EXECUTABLE}")
    endif()
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python: ${Python3_EXECUTABLE}  version ${Python3_VERSION}")

# Auto-install pybind11 for this Python if missing
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11"
    RESULT_VARIABLE _pybind11_check
    OUTPUT_QUIET ERROR_QUIET
)
if(NOT _pybind11_check EQUAL 0)
    message(STATUS "Installing pybind11 for ${Python3_EXECUTABLE} ...")
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pip install pybind11 --quiet
    )
endif()

# Find pybind11
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
                "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _pb11_cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pb11_result
    )
    if(_pb11_result EQUAL 0)
        list(APPEND CMAKE_PREFIX_PATH "${_pb11_cmake_dir}")
        find_package(pybind11 CONFIG REQUIRED)
    else()
        message(FATAL_ERROR
            "pybind11 not found. Run:\n"
            "  ${Python3_EXECUTABLE} -m pip install pybind11\n"
            "then delete build/ and retry.")
    endif()
endif()
message(STATUS "pybind11 ${pybind11_VERSION} found")

# Build the extension
pybind11_add_module(kudio_dsp cpp/kudio_dsp.cpp)

target_include_directories(kudio_dsp PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include"
)

# Write .pyd directly into project root (next to main.py)
set_target_properties(kudio_dsp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
foreach(cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
    set_target_properties(kudio_dsp PROPERTIES
        "LIBRARY_OUTPUT_DIRECTORY_${cfg}" "${CMAKE_SOURCE_DIR}"
        "RUNTIME_OUTPUT_DIRECTORY_${cfg}" "${CMAKE_SOURCE_DIR}"
    )
endforeach()

message(STATUS "")
message(STATUS "  Output: ${CMAKE_SOURCE_DIR}/kudio_dsp*.pyd")
message(STATUS "  Run with: python main.py")
message(STATUS "")