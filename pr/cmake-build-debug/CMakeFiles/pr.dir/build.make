# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/ylf9811/Desktop/IPCC2021/pr

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/ylf9811/Desktop/IPCC2021/pr/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/pr.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/pr.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pr.dir/flags.make

CMakeFiles/pr.dir/SLIC.cpp.o: CMakeFiles/pr.dir/flags.make
CMakeFiles/pr.dir/SLIC.cpp.o: ../SLIC.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/ylf9811/Desktop/IPCC2021/pr/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pr.dir/SLIC.cpp.o"
	/usr/local/Cellar/gcc@8/8.4.0_1/bin/gcc-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pr.dir/SLIC.cpp.o -c /Users/ylf9811/Desktop/IPCC2021/pr/SLIC.cpp

CMakeFiles/pr.dir/SLIC.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pr.dir/SLIC.cpp.i"
	/usr/local/Cellar/gcc@8/8.4.0_1/bin/gcc-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/ylf9811/Desktop/IPCC2021/pr/SLIC.cpp > CMakeFiles/pr.dir/SLIC.cpp.i

CMakeFiles/pr.dir/SLIC.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pr.dir/SLIC.cpp.s"
	/usr/local/Cellar/gcc@8/8.4.0_1/bin/gcc-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/ylf9811/Desktop/IPCC2021/pr/SLIC.cpp -o CMakeFiles/pr.dir/SLIC.cpp.s

# Object files for target pr
pr_OBJECTS = \
"CMakeFiles/pr.dir/SLIC.cpp.o"

# External object files for target pr
pr_EXTERNAL_OBJECTS =

pr: CMakeFiles/pr.dir/SLIC.cpp.o
pr: CMakeFiles/pr.dir/build.make
pr: CMakeFiles/pr.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/ylf9811/Desktop/IPCC2021/pr/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable pr"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pr.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pr.dir/build: pr

.PHONY : CMakeFiles/pr.dir/build

CMakeFiles/pr.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pr.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pr.dir/clean

CMakeFiles/pr.dir/depend:
	cd /Users/ylf9811/Desktop/IPCC2021/pr/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/ylf9811/Desktop/IPCC2021/pr /Users/ylf9811/Desktop/IPCC2021/pr /Users/ylf9811/Desktop/IPCC2021/pr/cmake-build-debug /Users/ylf9811/Desktop/IPCC2021/pr/cmake-build-debug /Users/ylf9811/Desktop/IPCC2021/pr/cmake-build-debug/CMakeFiles/pr.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pr.dir/depend

