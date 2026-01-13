cmake_minimum_required(VERSION 3.20)

project(CyberpunkCityDemo)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate compile_commands.json for VS Code IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add subdirectories
add_subdirectory(external/glfw)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/external/glm)
include_directories(${CMAKE_SOURCE_DIR}/external/tinygltf)
include_directories(${CMAKE_SOURCE_DIR}/external/miniaudio)
include_directories(${Vulkan_INCLUDE_DIRS})

# Find packages
find_package(Vulkan REQUIRED)

# Find slangc executable - check multiple possible locations
# First try the VULKAN_SDK environment variable
if(DEFINED ENV{VULKAN_SDK})
    find_program(SLANGC_EXECUTABLE 
        NAMES slangc slangc.exe
        PATHS "$ENV{VULKAN_SDK}/Bin"
        NO_DEFAULT_PATH
    )
endif()
# If not found, try common Vulkan SDK installation paths
if(NOT SLANGC_EXECUTABLE)
    file(GLOB VULKAN_SDK_DIRS "C:/VulkanSDK/*")
    foreach(SDK_DIR ${VULKAN_SDK_DIRS})
        if(IS_DIRECTORY "${SDK_DIR}/Bin")
            find_program(SLANGC_EXECUTABLE 
                NAMES slangc slangc.exe
                PATHS "${SDK_DIR}/Bin"
                NO_DEFAULT_PATH
            )
            if(SLANGC_EXECUTABLE)
                break()
            endif()
        endif()
    endforeach()
endif()
# Last resort: try system PATH
if(NOT SLANGC_EXECUTABLE)
    find_program(SLANGC_EXECUTABLE NAMES slangc slangc.exe)
endif()
if(NOT SLANGC_EXECUTABLE)
    message(FATAL_ERROR "slangc not found. Please ensure Vulkan SDK is installed and slangc is available. VULKAN_SDK: $ENV{VULKAN_SDK}")
else()
    message(STATUS "Found slangc: ${SLANGC_EXECUTABLE}")
endif()
find_package(OpenGL REQUIRED)

# Auto-VK library (disabled for now)
# add_subdirectory(external/auto-vk)

# Add executable
file(GLOB_RECURSE SOURCES "src/*.cpp")
add_executable(CyberpunkCityDemo ${SOURCES})

file(GLOB_RECURSE PRECOMPILED_HEADERS "include/pch/*.hpp")
target_precompile_headers(CyberpunkCityDemo PRIVATE ${PRECOMPILED_HEADERS})

# Link libraries
target_link_libraries(CyberpunkCityDemo PRIVATE glfw Vulkan::Vulkan)

# Configure dependencies
target_compile_definitions(CyberpunkCityDemo PRIVATE
    GLFW_INCLUDE_VULKAN
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    GLM_ENABLE_EXPERIMENTAL
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
)

# Compiler-specific options
if(MSVC)
    target_compile_options(CyberpunkCityDemo PRIVATE /W4)
    target_compile_definitions(CyberpunkCityDemo PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(CyberpunkCityDemo PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Copy assets to output directory - use POST_BUILD to handle multi-config generators
add_custom_command(TARGET CyberpunkCityDemo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/assets
    $<TARGET_FILE_DIR:CyberpunkCityDemo>/assets
    COMMENT "Copying assets to executable directory"
)

file(GLOB_RECURSE SHADER_FILES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.slang")

# Get all .slang files to use as dependencies (includes common headers, etc.)
file(GLOB_RECURSE ALL_SLANG_FILES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.slang")

foreach(SHADER_FILE ${SHADER_FILES})
    get_filename_component(SHADER_NAME_FULL ${SHADER_FILE} NAME)
    
    # Get the relative path from shaders directory to preserve folder structure
    file(RELATIVE_PATH SHADER_REL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/shaders ${SHADER_FILE})
    get_filename_component(SHADER_REL_DIR ${SHADER_REL_PATH} DIRECTORY)

    string(REGEX MATCH "\\.([^.]+)\\.slang$" MATCH_FOUND ${SHADER_NAME_FULL})

    if(MATCH_FOUND)
        set(SHADER_STAGE ${CMAKE_MATCH_1})

        if(${SHADER_STAGE} STREQUAL "frag")
            set(SHADER_STAGE "fragment")
        elseif(${SHADER_STAGE} STREQUAL "vert")
            set(SHADER_STAGE "vertex")
        endif()

        # Remove .slang from the shader name for the output file
        string(REPLACE ".slang" "" SHADER_NAME_WITHOUT_SLANG ${SHADER_NAME_FULL})

        # Create a build-time intermediate file path that preserves directory structure
        if(SHADER_REL_DIR STREQUAL "")
            set(SPIRV_FILE ${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME_WITHOUT_SLANG}.spv)
            set(SPIRV_OUTPUT_DIR ${CMAKE_BINARY_DIR}/shaders)
        else()
            set(SPIRV_FILE ${CMAKE_BINARY_DIR}/shaders/${SHADER_REL_DIR}/${SHADER_NAME_WITHOUT_SLANG}.spv)
            set(SPIRV_OUTPUT_DIR ${CMAKE_BINARY_DIR}/shaders/${SHADER_REL_DIR})
        endif()

        add_custom_command(
            OUTPUT ${SPIRV_FILE}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${SPIRV_OUTPUT_DIR}
            COMMAND ${SLANGC_EXECUTABLE}
                    -target spirv
                    -profile spirv_1_4
                    -emit-spirv-directly
                    -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
                    -fvk-use-entrypoint-name
                    -entry main
                    ${SHADER_FILE}
                    -o ${SPIRV_FILE}
            COMMAND ${CMAKE_COMMAND} -E touch ${SPIRV_FILE}
            DEPENDS ${SHADER_FILE} ${ALL_SLANG_FILES}
            COMMENT "Compiling ${SHADER_FILE}"
            VERBATIM
        )
        list(APPEND RAY_TRACING_SPV ${SPIRV_FILE})
    endif()
endforeach()

# Create shader compilation target
if(RAY_TRACING_SPV)
    list(LENGTH RAY_TRACING_SPV SHADER_COUNT)
    add_custom_target(CompileShaders ALL DEPENDS ${RAY_TRACING_SPV})
    add_dependencies(CyberpunkCityDemo CompileShaders)
    message(STATUS "Shader compilation configured: ${SHADER_COUNT} shaders will be compiled")
else()
    message(WARNING "No shaders found to compile!")
endif()

# Copy compiled SPIR-V files to the executable directory
# This depends on shaders being compiled first
add_custom_command(TARGET CyberpunkCityDemo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:CyberpunkCityDemo>/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/shaders $<TARGET_FILE_DIR:CyberpunkCityDemo>/shaders
    COMMENT "Copying compiled shaders to executable directory"
    DEPENDS ${RAY_TRACING_SPV}
)

# Copy assets directory to the executable directory
add_custom_command(TARGET CyberpunkCityDemo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets $<TARGET_FILE_DIR:CyberpunkCityDemo>/assets
    COMMENT "Copying assets to executable directory"
)

install(DIRECTORY ${CMAKE_BINARY_DIR}/shaders DESTINATION .)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/assets DESTINATION .)
