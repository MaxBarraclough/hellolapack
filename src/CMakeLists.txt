
cmake_minimum_required(VERSION 3.16.1)

project(
    helloblas
    LANGUAGES  CXX
    VERSION    0.1
)

set(CMAKE_CXX_STANDARD 17) # https://cmake.org/cmake/help/latest/guide/tutorial/index.html
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(LAPACK REQUIRED)
# find_package(BLAS REQUIRED) # BLA_VENDOR Generic)

# MESSAGE (${BLAS_LIBRARIES})
# MESSAGE (${LAPACK_LIBRARIES})

#configure_file(HelloBlasConfig.h.in HelloBlasConfig.h)


add_executable(
    helloblas
    "${CMAKE_SOURCE_DIR}/helloblas.cpp"
)

target_include_directories(
    helloblas PRIVATE "${CMAKE_SOURCE_DIR}"
)


# target_link_libraries(helloblas PRIVATE ${CMAKE_DL_LIBS} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} )
target_link_libraries(helloblas PRIVATE ${CMAKE_DL_LIBS} ${LAPACK_LIBRARIES} )

