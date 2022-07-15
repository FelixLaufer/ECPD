cmake_minimum_required(VERSION 3.14)

project("ECPD-Example")

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/bunny.xyz DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

include(FetchContent)
FetchContent_Declare(Eigen2Mat
  GIT_REPOSITORY https://github.com/FelixLaufer/Eigen2Mat
  GIT_TAG main
)
FetchContent_MakeAvailable(Eigen2Mat)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME}
  ECPD::ECPD
  Eigen2Mat::Eigen2Mat
)