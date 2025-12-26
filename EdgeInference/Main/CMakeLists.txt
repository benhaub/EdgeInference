target_sources(${PROJECT_NAME}${EXECUTABLE_SUFFIX}
PRIVATE FILE_SET headers TYPE HEADERS BASE_DIRS ${CMAKE_CURRENT_LIST_DIR} FILES
  EdgeInference.hpp
)

target_sources(${PROJECT_NAME}${EXECUTABLE_SUFFIX}
PRIVATE
  EdgeInference.cpp
)

if (ESP_PLATFORM)
  target_include_directories(__idf_main PRIVATE ${CMAKE_CURRENT_LIST_DIR})
  target_compile_definitions(__idf_main PRIVATE $<TARGET_PROPERTY:${PROJECT_NAME}${EXECUTABLE_SUFFIX},COMPILE_DEFINITIONS>)
endif()

target_include_directories(${PROJECT_NAME}${EXECUTABLE_SUFFIX} PRIVATE ${CMAKE_SOURCE_DIR}/TrainHandWritten0To9)