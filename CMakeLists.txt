project("VisionUtils")
cmake_minimum_required(VERSION 3.0)

add_definitions(-std=c++14)
#add_definitions(-DEIGEN_HAS_CUDA_FP16)

find_package(Caffe)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin)

file(GLOB VisionUtilsHeaders
     "${PROJECT_SOURCE_DIR}/Include/*.h")

file(GLOB VisionUtilsSources
     "${PROJECT_SOURCE_DIR}/Src/*.cpp"
	 "${PROJECT_SOURCE_DIR}/Src/*/*.cpp")

set(IncludeDirs
    ${PROJECT_SOURCE_DIR}/Include
	${EIGEN3_INCLUDE_DIRS})

if(${Caffe_FOUND})

	set(IncludeDirs ${IncludeDirs} ${Caffe_INCLUDE_DIRS})

	set(Libraries ${Libraries} ${Caffe_LIBRARIES})

	add_definitions(-DHAS_CAFFE)

endif()

if(${Pangolin_FOUND})

	set(IncludeDirs ${IncludeDirs} ${Pangolin_INCLUDE_DIRS})

	set(Libraries ${Libraries} ${Pangolin_LIBRARIES})

	add_definitions(-DHAS_PANGOLIN)

endif()

cuda_add_library(VisionUtils SHARED
	             ${VisionUtilsHeaders}
	             ${VisionUtilsSources})

target_include_directories(VisionUtils PUBLIC ${IncludeDirs})
target_link_libraries(VisionUtils ${Libraries})

# -=-=-=-=- Installation -=-=-=-=-
set( VisionUtils_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include/" )
set( VisionUtils_INCLUDE_DIRS ${VisionUtils_INCLUDE_DIR} )
set( VisionUtils_LIBRARY_DIR "${CMAKE_INSTALL_PREFIX}/lib/" )
set( VisionUtils_LIBRARY_DIRS ${VisionUtils_LIBRARY_DIR} )
set( VisionUtils_LIBRARY VisionUtils )
set( VisionUtils_LIBRARIES ${VisionUtils_LIBRARY} )

configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/VisionUtilsConfig.cmake.in
				${CMAKE_CURRENT_BINARY_DIR}/VisionUtilsConfig.cmake
				@ONLY ESCAPE_QUOTES)

install(TARGETS VisionUtils
		DESTINATION lib)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/Include/vu/
		DESTINATION include/vu/
		FILES_MATCHING PATTERN "*.h")

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/VisionUtilsConfig.cmake
		DESTINATION share/vu/)