# Copyright: (C) Robotics Brain and Cognitive Sciences (RBCS) - Istituto Italiano di Tecnologia
# CopyPolicy: Released under the terms of the GNU GPL v2.0.

cmake_minimum_required(VERSION 3.12)

#### SET PYTHON VERSION AND ENV NAME ####
set(NAME_ENV ${PROJECT_NAME}_conda_env)

set(PROGRAM_PERMISSIONS_DEFAULT
		OWNER_WRITE OWNER_READ OWNER_EXECUTE
		GROUP_READ GROUP_EXECUTE
		WORLD_READ WORLD_EXECUTE)

# Find Conda.
find_program(CONDA conda)

if(NOT CONDA)
	message(FATAL_ERROR "Could not find `conda`, please install Miniconda or Anaconda.")
endif()

set(ICUB_APPLICATIONS_PREFIX "$ENV{ICUB_ROOT}" CACHE PATH "Application path prefix")

# Find python command in conda
execute_process(
	COMMAND ${CONDA} info --base
	OUTPUT_VARIABLE CONDA_BASE
	OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(CONDA_ENV_PATH "${CMAKE_CURRENT_BINARY_DIR}/${NAME_ENV}")

# Print which conda is being used
message(STATUS "BUILDING CONDA ENV AT: ${CONDA_ENV_PATH}")

# Create the conda environment from environment.yml
add_custom_command(
	OUTPUT ${CONDA_ENV_PATH}
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/environment.yml ${CMAKE_CURRENT_BINARY_DIR}/environment.yml
	COMMAND ${CONDA} env create -f ${CMAKE_CURRENT_BINARY_DIR}/environment.yml -p ${CONDA_ENV_PATH}
	COMMENT "Creating conda environment at ${CONDA_ENV_PATH} from environment.yml"
)

# Add a custom target to trigger environment creation
add_custom_target(build-python ALL
	DEPENDS ${CONDA_ENV_PATH}
)

# set PATH_TO_ENV so that the python modules can find the conda env
set(PATH_TO_ENV ${CONDA_ENV_PATH})
message(STATUS "PATH_TO_ENV: ${PATH_TO_ENV}")

add_subdirectory(valence_arousal_estimator)


