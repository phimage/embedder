cmake_minimum_required(VERSION 3.12)
project(Embedder)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set ONNX Runtime version
set(ONNX_VERSION "1.22.0")
set(ONNX_BASE_URL "https://github.com/microsoft/onnxruntime/releases/download")

# Determine the correct ONNX Runtime directory name based on platform
if(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        set(ONNX_DIR_NAME "onnxruntime-osx-arm64-${ONNX_VERSION}")
        set(ONNX_DOWNLOAD_URL "${ONNX_BASE_URL}/v${ONNX_VERSION}/onnxruntime-osx-arm64-${ONNX_VERSION}.tgz")
    else()
        set(ONNX_DIR_NAME "onnxruntime-osx-x86_64-${ONNX_VERSION}")
        set(ONNX_DOWNLOAD_URL "${ONNX_BASE_URL}/v${ONNX_VERSION}/onnxruntime-osx-x86_64-${ONNX_VERSION}.tgz")
    endif()
elseif(WIN32)
    set(ONNX_DIR_NAME "onnxruntime-win-x64-${ONNX_VERSION}")
    set(ONNX_DOWNLOAD_URL "${ONNX_BASE_URL}/v${ONNX_VERSION}/onnxruntime-win-x64-${ONNX_VERSION}.zip")
else()
    set(ONNX_DIR_NAME "onnxruntime-linux-x64-${ONNX_VERSION}")
    set(ONNX_DOWNLOAD_URL "${ONNX_BASE_URL}/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz")
endif()

# Check if ONNX Runtime directory exists, if not download and extract
set(ONNX_LOCAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${ONNX_DIR_NAME}")
if(NOT EXISTS "${ONNX_LOCAL_DIR}")
    message(STATUS "ONNX Runtime directory not found: ${ONNX_LOCAL_DIR}")
    
    # Determine archive name and extraction command based on platform
    if(WIN32)
        set(ARCHIVE_NAME "${ONNX_DIR_NAME}.zip")
        set(EXTRACT_COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_CURRENT_SOURCE_DIR}/${ARCHIVE_NAME}")
    else()
        set(ARCHIVE_NAME "${ONNX_DIR_NAME}.tgz")
        set(EXTRACT_COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_CURRENT_SOURCE_DIR}/${ARCHIVE_NAME}")
    endif()
    
    set(ARCHIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${ARCHIVE_NAME}")
    
    # Check if archive file already exists locally
    if(EXISTS "${ARCHIVE_PATH}")
        message(STATUS "Found existing archive: ${ARCHIVE_NAME}")
    else()
        message(STATUS "Downloading ONNX Runtime from: ${ONNX_DOWNLOAD_URL}")
        
        # Download the archive
        file(DOWNLOAD "${ONNX_DOWNLOAD_URL}" "${ARCHIVE_PATH}"
             SHOW_PROGRESS
             STATUS DOWNLOAD_STATUS)
        
        # Check if download was successful
        list(GET DOWNLOAD_STATUS 0 DOWNLOAD_CODE)
        if(NOT DOWNLOAD_CODE EQUAL 0)
            list(GET DOWNLOAD_STATUS 1 DOWNLOAD_ERROR)
            message(FATAL_ERROR "Failed to download ONNX Runtime: ${DOWNLOAD_ERROR}")
        endif()
    endif()
    
    message(STATUS "Extracting ONNX Runtime archive...")
    execute_process(
        COMMAND ${EXTRACT_COMMAND}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE EXTRACT_RESULT
    )
    
    if(NOT EXTRACT_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to extract ONNX Runtime archive")
    endif()
    
    # Clean up the archive file
    file(REMOVE "${ARCHIVE_PATH}")
    message(STATUS "Successfully extracted ONNX Runtime to: ${ONNX_LOCAL_DIR}")
endif()

# Find ONNX Runtime
if(DEFINED ENV{ONNXRUNTIME_ROOT_PATH})
    set(ONNXRUNTIME_ROOT_PATH $ENV{ONNXRUNTIME_ROOT_PATH})
else()
    find_path(ONNXRUNTIME_ROOT_PATH
        NAMES include/onnxruntime_cxx_api.h
        HINTS 
            ${CMAKE_CURRENT_SOURCE_DIR}/${ONNX_DIR_NAME}
            /opt/homebrew  # Homebrew on Apple Silicon
            /usr/local     # Homebrew on Intel Mac
            /usr 
            /opt/onnxruntime
        PATH_SUFFIXES . include
    )
endif()

if(NOT ONNXRUNTIME_ROOT_PATH)
    message(FATAL_ERROR "ONNX Runtime not found. Please set ONNXRUNTIME_ROOT_PATH environment variable or install via homebrew")
endif()

message(STATUS "Using ONNX Runtime from: ${ONNXRUNTIME_ROOT_PATH}")

include_directories(${ONNXRUNTIME_ROOT_PATH}/include)

# Find ONNX Runtime library with platform-specific names
if(APPLE)
    set(LIB_NAMES onnxruntime libonnxruntime.dylib)
    set(LIB_SUFFIXES lib)
elseif(WIN32)
    set(LIB_NAMES onnxruntime libonnxruntime onnxruntime.lib)
    set(LIB_SUFFIXES lib)
else()
    set(LIB_NAMES onnxruntime libonnxruntime.so)
    set(LIB_SUFFIXES lib)
endif()

find_library(ONNXRUNTIME_LIB 
    NAMES ${LIB_NAMES}
    HINTS ${ONNXRUNTIME_ROOT_PATH}
    PATH_SUFFIXES ${LIB_SUFFIXES}
    NO_DEFAULT_PATH
)

if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime library not found in ${ONNXRUNTIME_ROOT_PATH}/lib")
endif()

message(STATUS "Using ONNX Runtime library: ${ONNXRUNTIME_LIB}")

add_executable(embedder
    main.cpp
    embedder.cpp
)

target_include_directories(embedder PRIVATE src)
target_link_libraries(embedder ${ONNXRUNTIME_LIB})

# Platform-specific post-build steps
if(WIN32)
    add_custom_command(TARGET embedder POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT_PATH}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:embedder>)
elseif(APPLE)
    # On macOS, set rpath to find the dylib
    set_target_properties(embedder PROPERTIES
        INSTALL_RPATH "${ONNXRUNTIME_ROOT_PATH}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE)
endif()