# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(CMAKE_CXX_STANDARD 14)

if(MSVC)
	add_subdirectory(dll)
else()
	#file(GLOB CPP_SOURCES *.cpp)
	# Keep the inherited merge-path sources until a deeper dependency audit proves
	# they are safe to remove from the current PageANN library build.
	set(CPP_SOURCES ann_exception.cpp aux_utils.cpp index.cpp
        linux_aligned_file_reader.cpp math_utils.cpp memory_mapper.cpp
        partition_and_pq.cpp pq_flash_index.cpp logger.cpp distance.cpp
		 utils.cpp pageann/graph_delta.cpp pageann/index_merger.cpp
        pageann/merge_insert.cpp pageann/inplace_backend.cpp pageann/inplace_graph_ops.cpp)
	add_library(${PROJECT_NAME} ${CPP_SOURCES})
	add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})
endif()
install()
