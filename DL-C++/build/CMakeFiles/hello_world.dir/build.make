# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 4.0

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
CMAKE_COMMAND = /opt/homebrew/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/arkajyotisaha/Desktop/DL-Finanace/DL-C++/ch-2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/arkajyotisaha/Desktop/DL-Finanace/DL-C++/build

# Include any dependencies generated for this target.
include CMakeFiles/hello_world.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/hello_world.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/hello_world.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/hello_world.dir/flags.make

CMakeFiles/hello_world.dir/codegen:
.PHONY : CMakeFiles/hello_world.dir/codegen

CMakeFiles/hello_world.dir/example.cpp.o: CMakeFiles/hello_world.dir/flags.make
CMakeFiles/hello_world.dir/example.cpp.o: /Users/arkajyotisaha/Desktop/DL-Finanace/DL-C++/ch-2/example.cpp
CMakeFiles/hello_world.dir/example.cpp.o: CMakeFiles/hello_world.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/arkajyotisaha/Desktop/DL-Finanace/DL-C++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/hello_world.dir/example.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/hello_world.dir/example.cpp.o -MF CMakeFiles/hello_world.dir/example.cpp.o.d -o CMakeFiles/hello_world.dir/example.cpp.o -c /Users/arkajyotisaha/Desktop/DL-Finanace/DL-C++/ch-2/example.cpp

CMakeFiles/hello_world.dir/example.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/hello_world.dir/example.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/arkajyotisaha/Desktop/DL-Finanace/DL-C++/ch-2/example.cpp > CMakeFiles/hello_world.dir/example.cpp.i

CMakeFiles/hello_world.dir/example.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/hello_world.dir/example.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/arkajyotisaha/Desktop/DL-Finanace/DL-C++/ch-2/example.cpp -o CMakeFiles/hello_world.dir/example.cpp.s

# Object files for target hello_world
hello_world_OBJECTS = \
"CMakeFiles/hello_world.dir/example.cpp.o"

# External object files for target hello_world
hello_world_EXTERNAL_OBJECTS =

bin/hello_world: CMakeFiles/hello_world.dir/example.cpp.o
bin/hello_world: CMakeFiles/hello_world.dir/build.make
bin/hello_world: CMakeFiles/hello_world.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/arkajyotisaha/Desktop/DL-Finanace/DL-C++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bin/hello_world"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hello_world.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/hello_world.dir/build: bin/hello_world
.PHONY : CMakeFiles/hello_world.dir/build

CMakeFiles/hello_world.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/hello_world.dir/cmake_clean.cmake
.PHONY : CMakeFiles/hello_world.dir/clean

CMakeFiles/hello_world.dir/depend:
	cd /Users/arkajyotisaha/Desktop/DL-Finanace/DL-C++/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/arkajyotisaha/Desktop/DL-Finanace/DL-C++/ch-2 /Users/arkajyotisaha/Desktop/DL-Finanace/DL-C++/ch-2 /Users/arkajyotisaha/Desktop/DL-Finanace/DL-C++/build /Users/arkajyotisaha/Desktop/DL-Finanace/DL-C++/build /Users/arkajyotisaha/Desktop/DL-Finanace/DL-C++/build/CMakeFiles/hello_world.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/hello_world.dir/depend

