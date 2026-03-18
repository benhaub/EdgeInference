###############################################################################
#File: CMakeLists.txt                                                         #
#Authour: Ben Haubrich                                                        #
#Date: April 13th, 2025                                                       #
#Synopsis: A top-level CMake file that works well with the AbstractionLayer,  #
#          ccsToCMake, and buildScript                                        #
###############################################################################
cmake_minimum_required(VERSION 3.21.3)

#Use `cmake --help-policy <POLICY>` to learn more
cmake_policy(SET CMP0079 NEW)
cmake_policy(SET CMP0076 NEW)
cmake_policy(SET CMP0067 NEW)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_BUILD_PARALLEL_LEVEL 8)

set(MODEL_FEATURES "InferenceAbstractionTypes::ModelFeatures::QuanitizationConversion,
                    InferenceAbstractionTypes::ModelFeatures::Convolutional2dLayers,
                    InferenceAbstractionTypes::ModelFeatures::MaxPooling2dSpatialDimension,
                    InferenceAbstractionTypes::ModelFeatures::ReshapeManipulation,
                    InferenceAbstractionTypes::ModelFeatures::DenseLayers,
                    InferenceAbstractionTypes::ModelFeatures::SoftMaxActivation")

set(ARENA_SIZE_BYTES 35*1024)

#python3 apf.py build --toolchain ccsToCMake/toolchain.cmake --target CC32xx
if (Cc32xx)
  message(STATUS "Building for Cc32xx")

  project(edgeInference LANGUAGES C CXX)

  set(EXECUTABLE_SUFFIX ".elf")
  add_executable(${PROJECT_NAME}${EXECUTABLE_SUFFIX} ${CMAKE_CURRENT_LIST_DIR}/main.cpp)

  #Order matters!
  include(${CMAKE_SOURCE_DIR}/cc32xx_module_selection.cmake)
  include(${CMAKE_SOURCE_DIR}/AbstractionLayer/targetPlatform.cmake)
  include(${CMAKE_CURRENT_LIST_DIR}/ccsToCMake/Cc32xx/target.cmake)

  add_custom_command(TARGET ${PROJECT_NAME}${EXECUTABLE_SUFFIX} POST_BUILD
  COMMAND
  ${CMAKE_OBJCOPY} ARGS -O binary ${CMAKE_SOURCE_DIR}/Cc32xx_build/${PROJECT_NAME}${EXECUTABLE_SUFFIX} ${CMAKE_SOURCE_DIR}/Cc32xx_build/${PROJECT_NAME}.bin
    WORKING_DIRECTORY
      ${CMAKE_CURRENT_LIST_DIR}
    COMMENT
      "Stripping executable"
    VERBATIM
  )

  #There is only one partition. So the type could be anything.
  target_compile_definitions(${PROJECT_NAME}${EXECUTABLE_SUFFIX}
  PRIVATE
    MAIN_PARTITION_TYPE=FileSystemTypes::Implementation::Spiffs
    APP_PROCESSOR_CLOCK_FREQUENCY=0
    APP_EXTERNAL_CRYSTAL_FREQUENCY=0
    APP_DEFAULT_STACK_SIZE=12*1024
    #Sysconfig automatically sets pins, so these will be ignored by the driver.
    #However, if using an application which is platform agnostic, -1 pins will be rejected as invalid.
    APP_MAX_QUEUEABLE_EVENTS=10
    APP_MAX_QUEUEABLE_COMMANDS=10
  )

#python3 apf.py build --toolchain ccsToCMake/toolchain.cmake --target Tm4c123
elseif (Tm4c123)
  message(STATUS "Building for Tm4c123")

  project(edgeInference LANGUAGES C CXX)

  set(EXECUTABLE_SUFFIX ".elf")
  add_executable(${PROJECT_NAME}${EXECUTABLE_SUFFIX} ${CMAKE_CURRENT_LIST_DIR}/main.cpp)

  #Order matters!
  include(${CMAKE_CURRENT_LIST_DIR}/ccsToCMake/Tm4c123/target.cmake)
  include(${CMAKE_SOURCE_DIR}/tm4c123_module_selection.cmake)
  include(${CMAKE_SOURCE_DIR}/AbstractionLayer/targetPlatform.cmake)

  add_custom_command(TARGET ${PROJECT_NAME}${EXECUTABLE_SUFFIX} POST_BUILD
  COMMAND
  ${CMAKE_OBJCOPY} ARGS -O binary ${CMAKE_SOURCE_DIR}/Tm4c123_build/${PROJECT_NAME}${EXECUTABLE_SUFFIX} ${CMAKE_SOURCE_DIR}/Tm4c123_build/${PROJECT_NAME}.bin
    WORKING_DIRECTORY
      ${CMAKE_CURRENT_LIST_DIR}
    COMMENT
      "Stripping executable"
    VERBATIM
  )

  target_compile_definitions(${PROJECT_NAME}${EXECUTABLE_SUFFIX}
  PRIVATE
    MAIN_PARTITION_TYPE=FileSystemTypes::Implementation::Unknown
    APP_PROCESSOR_CLOCK_FREQUENCY=80E6
    APP_EXTERNAL_CRYSTAL_FREQUENCY=16E6
    APP_DEFAULT_STACK_SIZE=12*1024
    APP_HEAP_SIZE=12*1024
    APP_MAX_QUEUEABLE_EVENTS=10
    APP_MAX_QUEUEABLE_COMMANDS=10
  )

#python3 apf.py build --toolchain ccsToCMake/toolchain.cmake --target Tm4c129
elseif (Tm4c129)
  message(STATUS "Building for Tm4c129")

  project(edgeInference LANGUAGES C CXX)

  set(EXECUTABLE_SUFFIX ".elf")
  add_executable(${PROJECT_NAME}${EXECUTABLE_SUFFIX} ${CMAKE_CURRENT_LIST_DIR}/main.cpp)

  #Order matters!
  include(${CMAKE_CURRENT_LIST_DIR}/ccsToCMake/Tm4c129/target.cmake)
  include(${CMAKE_SOURCE_DIR}/tm4c129_module_selection.cmake)
  include(${CMAKE_SOURCE_DIR}/AbstractionLayer/targetPlatform.cmake)

  add_custom_command(TARGET ${PROJECT_NAME}${EXECUTABLE_SUFFIX} POST_BUILD
  COMMAND
  ${CMAKE_OBJCOPY} ARGS -O binary ${CMAKE_SOURCE_DIR}/Tm4c129_build/${PROJECT_NAME}${EXECUTABLE_SUFFIX} ${CMAKE_SOURCE_DIR}/Tm4c129_build/${PROJECT_NAME}.bin
    WORKING_DIRECTORY
      ${CMAKE_CURRENT_LIST_DIR}
    COMMENT
      "Stripping executable"
    VERBATIM
  )

  target_compile_definitions(${PROJECT_NAME}${EXECUTABLE_SUFFIX}
  PRIVATE
    MAIN_PARTITION_TYPE=FileSystemTypes::Implementation::Unknown
    APP_EXTERNAL_CRYSTAL_FREQUENCY=25E6
    APP_DEFAULT_STACK_SIZE=8*1024
    APP_HEAP_SIZE=32*1024
    APP_MAX_NUMBER_OF_THREADS=1
    APP_TFLITE_MODEL_FEATURES=${MODEL_FEATURES}
    APP_TFLITE_ALLOCATED_MEMORY_SIZE_BYTES=${ARENA_SIZE_BYTES}
    #Pin Number are set using sysconfig.
    APP_RIVERDI_LCD_CHIP_SELECT_PIN_NUMBER=-1
    APP_RIVERDI_LCD_CLOCK_PIN_NUMBER=-1
    APP_RIVERDI_LCD_POCI_PIN_NUMBER=-1
    APP_RIVERDI_LCD_PICO_PIN_NUMBER=-1
    APP_RIVERDI_LCD_PART_NUMBER=LcdFactoryTypes::PartNumber::Rvt35AHBNWC00
    APP_RIVERDI_LCD_SPI_PERIPHERAL_NUMBER=PeripheralNumber::One
    APP_PROCESSOR_CLOCK_FREQUENCY=120E6
    APP_MAX_QUEUEABLE_EVENTS=10
    APP_MAX_QUEUEABLE_COMMANDS=10
    APP_RIVERDI_LCD_POWERDOWN_PERIPHERAL_NUMBER=PeripheralNumber::Three
    APP_RIVERDI_LCD_POWERDOWN_PIN_NUMBER=1
    APP_RIVERDI_LCD_SPI_CLOCK_FREQUENCY=50E3
    BT81X_CHIP_SELECT_PERIPHERAL_NUMBER=PeripheralNumber::Three
    BT81X_CHIP_SELECT_PIN_NUMBER=0
  )

#idf.py build
elseif (ESP_PLATFORM)
  message(STATUS "Building for ESP")

  if (${RELEASE_BUILD})
    set(SDKCONFIG_DEFAULTS "sdkconfig.release")
  endif()

  #If not set, defaults to all components. Set this to reduce the amount of
  #components that are built at compile time. Required and linked components
  #for the ESP main component are located in the "main component" CMakeLists.txt file.
  set(COMPONENTS
    main
  )
  #Internal IDF cmake files set the target to end in .elf so we don't have much of a choice here.
  set(EXECUTABLE_SUFFIX ".elf")
  #include directive must come after the set(COMPONENTS*) directive
  include($ENV{IDF_PATH}/tools/cmake/project.cmake)
  #ESP projects require the project line to be declared after the include
  project(edgeInference)

  include(${CMAKE_SOURCE_DIR}/esp_module_selection.cmake)
  include(${CMAKE_SOURCE_DIR}/AbstractionLayer/targetPlatform.cmake)

  target_compile_definitions(${PROJECT_NAME}${EXECUTABLE_SUFFIX}
  PRIVATE
    MAIN_PARTITION_TYPE=FileSystemTypes::Implementation::KeyValue
    APP_PROCESSOR_CLOCK_FREQUENCY=0
    APP_MAX_QUEUEABLE_EVENTS=10
    APP_MAX_QUEUEABLE_COMMANDS=10
    APP_EXTERNAL_CRYSTAL_FREQUENCY=0
    APP_DEFAULT_STACK_SIZE=16*1024
    APP_MAX_QUEUEABLE_EVENTS=10
    APP_MAX_QUEUEABLE_COMMANDS=10
  )
  target_compile_definitions(__idf_main PRIVATE $<TARGET_PROPERTY:${PROJECT_NAME}${EXECUTABLE_SUFFIX},COMPILE_DEFINITIONS>)

#Even if cross-compiling, your host system name will still be present so cross compiled targets should be evaluated first.
#python3 apf.py build
elseif(${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Linux")
  message(STATUS "Building for Linux")

  project(edgeInference LANGUAGES C CXX)

  set(EXECUTABLE_SUFFIX ".elf")
  add_executable(${PROJECT_NAME}${EXECUTABLE_SUFFIX} ${CMAKE_CURRENT_LIST_DIR}/main.cpp)

  include(${CMAKE_SOURCE_DIR}/linux_module_selection.cmake)
  include(${CMAKE_SOURCE_DIR}/AbstractionLayer/targetPlatform.cmake)

  #The partition type when running on a desktop doesn't really matter. It could be set to anything.
  target_compile_definitions(${PROJECT_NAME}${EXECUTABLE_SUFFIX}
  PRIVATE
    MAIN_PARTITION_TYPE=FileSystemTypes::Implementation::Ntfs
    APP_PROCESSOR_CLOCK_FREQUENCY=0
    APP_EXTERNAL_CRYSTAL_FREQUENCY=0
    APP_DEFAULT_STACK_SIZE=12*1024
    APP_HOME_DIRECTORY="$ENV{HOME}"
    APP_MAX_NUMBER_OF_THREADS=1
    APP_TFLITE_MODEL_FEATURES=${MODEL_FEATURES}
    APP_TFLITE_ALLOCATED_MEMORY_SIZE_BYTES=${ARENA_SIZE_BYTES}
    APP_LCD_SPI_PERIPHERAL_NUMBER=PeripheralNumber::One
    APP_CHIP_SELECT_PIN_NUMBER=-1
    APP_CLOCK_PIN_NUMBER=-1
    APP_POCI_PIN_NUMBER=-1
    APP_PICO_PIN_NUMBER=-1
    APP_LCD_PART_NUMBER=LcdFactoryTypes::PartNumber::Rvt35AHBNWC00
    APP_LCD_SPI_PERIPHERAL_NUMBER=PeripheralNumber::One
    APP_MAX_QUEUEABLE_EVENTS=10
    APP_MAX_QUEUEABLE_COMMANDS=10
    BT81X_CHIP_SELECT_PERIPHERAL_NUMBER=PeripheralNumber::Three
    BT81X_CHIP_SELECT_PIN_NUMBER=0
    APP_LCD_POWERDOWN_PERIPHERAL_NUMBER=PeripheralNumber::Three
    APP_LCD_POWERDOWN_PIN_NUMBER=-1
  )

elseif(${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Darwin")
    message(STATUS "Building for Darwin")

  project(edgeInferece LANGUAGES C CXX)

  set(EXECUTABLE_SUFFIX ".Mach-O")
  add_executable(${PROJECT_NAME}${EXECUTABLE_SUFFIX} ${CMAKE_CURRENT_LIST_DIR}/main.cpp)

  include(${CMAKE_SOURCE_DIR}/darwin_module_selection.cmake)
  include(${CMAKE_SOURCE_DIR}/AbstractionLayer/targetPlatform.cmake)

  #The partition type when running on a desktop doesn't really matter. It could be set to anything.
  target_compile_definitions(${PROJECT_NAME}${EXECUTABLE_SUFFIX}
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/main_PARTITION_TYPE=FileSystemTypes::Implementation::Ntfs
    APP_PROCESSOR_CLOCK_FREQUENCY=0
    APP_EXTERNAL_CRYSTAL_FREQUENCY=0
    APP_DEFAULT_STACK_SIZE=12*1024
    APP_HOME_DIRECTORY="$ENV{HOME}"
    APP_MAX_QUEUEABLE_EVENTS=10
    APP_MAX_QUEUEABLE_COMMANDS=10
  )

#python3 apf.py build --toolchain ccsToCMake/toolchain.cmake --target Raspbian12Pi4ModelB
elseif(Raspbian12Pi4ModelB)
  message(STATUS "Building for Raspbian 12 on Pi 4 Model B")
  #https://www.raspberrypi.com/documentation/computers/getting-started.html#setting-up-your-raspberry-pi
  #https://datasheets.raspberrypi.com/rpi4/raspberry-pi-4-datasheet.pdf
  #https://datasheets.raspberrypi.com/bcm2711/bcm2711-peripherals.pdf

  project(edgeInference LANGUAGES C CXX)

  set(EXECUTABLE_SUFFIX ".elf")
  add_executable(${PROJECT_NAME}${EXECUTABLE_SUFFIX} ${CMAKE_CURRENT_LIST_DIR}/main.cpp)

  include(${CMAKE_SOURCE_DIR}/raspbian12Pi4ModelB_module_selection.cmake)
  include(${CMAKE_SOURCE_DIR}/AbstractionLayer/targetPlatform.cmake)

  #The partition type when running on a desktop doesn't really matter. It could be set to anything.
  target_compile_definitions(${PROJECT_NAME}${EXECUTABLE_SUFFIX}
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/main_PARTITION_TYPE=FileSystemTypes::Implementation::Ntfs
    APP_PROCESSOR_CLOCK_FREQUENCY=0
    APP_EXTERNAL_CRYSTAL_FREQUENCY=0
    APP_DEFAULT_STACK_SIZE=12*1024
    APP_HOME_DIRECTORY=$ENV{HOME}
    APP_MAX_QUEUEABLE_EVENTS=10
    APP_MAX_QUEUEABLE_COMMANDS=10
  )

else()
  message(FATAL_ERROR "Invalid or unsupported target specified")
endif()

#python3 app.py build --build-type Release
#or idf.py build -DRELEASE_BUILD=1
if (${RELEASE_BUILD})
  message(STATUS "Building release build")
elseif (${SANITIZE_BUILD})
  message(STATUS "Building ${SANITIZE_CHECK} sanitize build")
elseif (${PROFILE_BUILD})
  message(STATUS "Building profile build")
else ()
  #Default to debug build
  #or idf.py build -DRELEASE_BUILD=0
  message(STATUS "Building debug build")
endif()

target_compile_options(${PROJECT_NAME}${EXECUTABLE_SUFFIX}
PUBLIC
  -ffunction-sections
  -fdata-sections
  -Wall
  #https://cmake.org/cmake/help/latest/manual/cmake-generator-expressions.7.html#genex:condition
  $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
  $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>

  $<$<BOOL:${RELEASE_BUILD}>:-fomit-frame-pointer>
  $<$<BOOL:${RELEASE_BUILD}>:-O2>

  $<$<BOOL:${SANITIZE_BUILD}>:-O1>
  $<$<BOOL:${SANITIZE_BUILD}>:-g>
  $<$<BOOL:${SANITIZE_BUILD}>:-fsanitize=${SANITIZE_CHECK}>
  $<$<BOOL:${SANITIZE_BUILD}>:-fno-omit-frame-pointer>

  $<$<BOOL:${PROFILE_BUILD}>:-O2>
  $<$<BOOL:${PROFILE_BUILD}>:-g>
  $<$<AND:$<BOOL:${PROFILE_BUILD}>,$<STREQUAL:${CMAKE_HOST_SYSTEM_NAME},Linux>>:-pg>
  $<$<AND:$<BOOL:${PROFILE_BUILD}>,$<STREQUAL:${CMAKE_HOST_SYSTEM_NAME},Darwin>>:-fprofile-instr-generate>
  $<$<AND:$<BOOL:${PROFILE_BUILD}>,$<STREQUAL:${CMAKE_HOST_SYSTEM_NAME},Darwin>>:-fcoverage-mapping>
  $<$<AND:$<BOOL:${PROFILE_BUILD}>,$<STREQUAL:${CMAKE_HOST_SYSTEM_NAME},Darwin>>:-fno-omit-frame-pointer>

  # Debug build flags (default as in the case of a lack of any other build type specified).
  $<$<AND:$<NOT:$<BOOL:${RELEASE_BUILD}>>,$<NOT:$<BOOL:${SANITIZE_BUILD}>>,$<NOT:$<BOOL:${PROFILE_BUILD}>>>:-O0>
  $<$<AND:$<NOT:$<BOOL:${RELEASE_BUILD}>>,$<NOT:$<BOOL:${SANITIZE_BUILD}>>,$<NOT:$<BOOL:${PROFILE_BUILD}>>>:-g>
  $<$<AND:$<NOT:$<BOOL:${RELEASE_BUILD}>>,$<NOT:$<BOOL:${SANITIZE_BUILD}>>,$<NOT:$<BOOL:${PROFILE_BUILD}>>>:-gdwarf-3>
  $<$<AND:$<NOT:$<BOOL:${RELEASE_BUILD}>>,$<NOT:$<BOOL:${SANITIZE_BUILD}>>,$<NOT:$<BOOL:${PROFILE_BUILD}>>>:-gstrict-dwarf>
)

target_link_options(${PROJECT_NAME}${EXECUTABLE_SUFFIX}
PUBLIC
  $<$<BOOL:${SANITIZE_BUILD}>:-fsanitize=${SANITIZE_CHECK}>
  
  $<$<AND:$<BOOL:${PROFILE_BUILD}>,$<STREQUAL:${CMAKE_HOST_SYSTEM_NAME},Linux>>:-pg>
  $<$<AND:$<BOOL:${PROFILE_BUILD}>,$<STREQUAL:${CMAKE_HOST_SYSTEM_NAME},Darwin>>:-fprofile-instr-generate>
  $<$<AND:$<BOOL:${PROFILE_BUILD}>,$<STREQUAL:${CMAKE_HOST_SYSTEM_NAME},Darwin>>:-fcoverage-mapping>
)

target_compile_definitions(${PROJECT_NAME}${EXECUTABLE_SUFFIX}
PUBLIC
  $<$<BOOL:${RELEASE_BUILD}>:CLOUD_CLIENT_LOGGING_ENABLED=0>
  $<$<BOOL:${RELEASE_BUILD}>:CLOUD_CLIENT_VERBOSE_LOGGING_ENABLED=0>
  $<$<BOOL:${RELEASE_BUILD}>:DEBUG_BUILD=0>
  
  $<$<AND:$<BOOL:${PROFILE_BUILD}>,$<STREQUAL:${CMAKE_HOST_SYSTEM_NAME},Linux>>:-pg>
  $<$<AND:$<NOT:$<BOOL:${RELEASE_BUILD}>>,$<NOT:$<BOOL:${SANITIZE_BUILD}>>,$<NOT:$<BOOL:${PROFILE_BUILD}>>>:DEBUG_BUILD=1>
)

#Cross-platform edgeInference core code
add_subdirectory(EdgeInference/Main)
