find_package(GTest REQUIRED)
#find_package(OpenMP REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)
find_package(LAPACK REQUIRED)
aux_source_directory(../tests/test_data DIR_TEST_DATA)
aux_source_directory(../tests/test_layer DIR_TEST_LAYER)
aux_source_directory(../tests/test_net DIR_TEST_NET)
aux_source_directory(../tests/test_runtime DIR_TEST_RUNTIME)

set(link_lib glog::glog GTest::gtest)
if (!WIN32)
    set(link_lib "${link_lib} pthread")
endif ()
set(link_math_lib ${ARMADILLO_LIBRARIES} ${OpenBLAS_LIB} ${LAPACK_LIBRARIES})

add_executable(test_zinfer test_main.cpp ${DIR_TEST_DATA} ${DIR_TEST_LAYER} ${DIR_TEST_NET} ${DIR_TEST_RUNTIME})

target_link_libraries(test_zinfer ${link_lib} ${link_math_lib})
target_link_directories(test_zinfer PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(test_zinfer zinfer)

if (MSVC)
    add_custom_command(TARGET test_zinfer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE_DIR:zinfer>/zinfer.dll"
            $<TARGET_FILE_DIR:test_zinfer>)
endif ()

target_include_directories(test_zinfer PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(test_zinfer PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(test_zinfer PUBLIC ${Armadillo_INCLUDE_DIR})
add_test(
    NAME test_zinfer
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_zinfer
)
set_target_properties(test_zinfer PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
