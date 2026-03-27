# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(CMAKE_CXX_STANDARD 14)

add_executable(build_disk_index build_disk_index.cpp)
if(MSVC)
	target_link_options(build_disk_index PRIVATE /MACHINE:x64 /DEBUG:FULL "/INCLUDE:_tcmalloc")
	target_link_libraries(build_disk_index debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/diskann_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
	target_link_libraries(build_disk_index optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/diskann_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
else()
	target_link_libraries(build_disk_index ${PROJECT_NAME} -ltcmalloc aio)
endif()

add_executable(search_disk_index search_disk_index.cpp
    ${PROJECT_SOURCE_DIR}/src/aux_utils.cpp)
if(MSVC)
	target_link_options(search_disk_index PRIVATE /MACHINE:x64 /DEBUG:FULL)
	target_link_libraries(search_disk_index debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/diskann_dll.lib)
	target_link_libraries(search_disk_index optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/diskann_dll.lib)
else()
	target_link_libraries(search_disk_index ${PROJECT_NAME} aio -ltcmalloc)
endif()

add_executable(pageann_update_demo test_update_experiment.cpp)
if(MSVC)
	target_link_options(pageann_update_demo PRIVATE /MACHINE:x64 /DEBUG:FULL)
	target_link_libraries(pageann_update_demo debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/diskann_dll.lib)
	target_link_libraries(pageann_update_demo optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/diskann_dll.lib)
else()
	target_link_libraries(pageann_update_demo ${PROJECT_NAME} aio -ltcmalloc)
endif()

add_executable(pageann_benchmark benchmark_adapter.cpp)
if(MSVC)
	target_link_options(pageann_benchmark PRIVATE /MACHINE:x64 /DEBUG:FULL)
	target_link_libraries(pageann_benchmark debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/diskann_dll.lib)
	target_link_libraries(pageann_benchmark optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/diskann_dll.lib)
else()
	target_link_libraries(pageann_benchmark ${PROJECT_NAME} aio -ltcmalloc)
endif()
