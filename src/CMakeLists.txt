# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(ACL_DET_YOLO)

# Compile options
add_compile_options(
    -std=c++11
    -fPIC
    -fstack-protector-all
    -D_FORTIFY_SOURCE=2 -O2
    )

# Specify target generation path
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../../../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall -s")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie")
set(CMAKE_SKIP_RPATH TRUE)

# Header path
set(OPENCV_INC $ENV{DDK_PATH}/atc/third_party/opencv/include/opencv4)
include_directories(
    $ENV{DDK_PATH}/acllib/include/
    ../inc/
    ${OPENCV_INC}
)

# add host lib path
if (target MATCHES "Simulator_Function" OR target MATCHES "Simulator_Instruction" OR target MATCHES "Simulator_Performance")
    set(ENV{NPU_HOST_LIB} $ENV{DDK_PATH}/toolkit/tools/sim/lib)
    set(OPENCV_LIB $ENV{DDK_PATH}/atc/third_party_lib)
else()
    set(ENV{NPU_HOST_LIB} $ENV{DDK_PATH}/acllib/lib64/stub)
    if (CMAKE_CXX_COMPILER MATCHES "aarch64-v01c01-linux-musl-gcc")
        set(OPENCV_LIB ${PROJECT_SOURCE_DIR}/../../../third_party/musl/opencv/lib)
    else()
        set(OPENCV_LIB ${PROJECT_SOURCE_DIR}/../../../third_party/glibc/opencv/lib)
    endif()
endif()
set(LIB_PATH $ENV{NPU_HOST_LIB})
MESSAGE(NPU_HOST_LIB=$ENV{NPU_HOST_LIB})
link_directories(${LIB_PATH})
FILE(GLOB SAMPLE_SRC_FILE *.cpp)

add_executable(main ${SAMPLE_SRC_FILE})

target_link_libraries(main
    ${OPENCV_LIB}/libopencv_core.so
    ${OPENCV_LIB}/libopencv_imgcodecs.so
    ${OPENCV_LIB}/libopencv_imgproc.so
)

if (target MATCHES "Simulator_Function")
    target_link_libraries(main funcsim)
elseif (target MATCHES "Simulator_Instruction" OR target MATCHES "Simulator_Performance")
    target_link_libraries(main instsim)
else()
    target_link_libraries(main svp_acl ss_mpi_sysmem ss_mpi_km pthread securec dl protobuf-c stdc++ m)
endif()

install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
