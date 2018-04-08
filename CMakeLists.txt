project("VisionUtils")
cmake_minimum_required(VERSION 3.0)

add_definitions(-std=c++14)
add_definitions(-DEIGEN_HAS_CUDA_FP16)

find_package(Caffe)
find_package(CUDA REQUIRED)

file(GLOB VisionUtilsHeaders
     "${PROJECT_SOURCE_DIR}/Include/*.h")

file(GLOB VisionUtilsSources
     "${PROJECT_SOURCE_DIR}/Src/*.cpp")	

set(IncludeDirs
    ${PROJECT_SOURCE_DIR}/Include)

if(${Caffe_FOUND})
	set(IncludeDirs ${IncludeDirs} ${Caffe_INCLUDE_DIRS})
endif()

cuda_add_library(VisionUtils
	             ${VisionUtilsHeaders}
	             ${VisionUtilsSources})

target_include_directories(VisionUtils PUBLIC ${IncludeDirs})


# -=-=-=-=- Installation -=-=-=-=-
set( VisionUtils_INCLUDE_DIR "{CMAKE_INSTALL_PREFIX}/include/" )
set( VisionUtils_INCLUDE_DIRS {VisionUtils_INCLUDE_DIR} )

configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/VisionUtilsConfig.cmake.in
				${CMAKE_CURRENT_BINARY_DIR}/VisionUtilsConfig.cmake
				@ONLY ESCAPE_QUOTES)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/Include/vu/
		DESTINATION include/vu/
		FILES_MATCHING PATTERN "*.h")

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/VisionUtilsConfig.cmake
		DESTINATION share/vu/)