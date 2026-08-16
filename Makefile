PATH_TOOLCHAIN				=
PATH_VULKAN					=

COMPILER_C					= gcc
COMPILER_GLSL				= $(PATH_VULKAN)/Bin/glslc.exe

FILE_RESULT					= vulkan_square_rotating.exe

DIRICTORY_SOURCES			= ./*.c ./common/*.c

FILE_SHADER_VERTEX_SOURCE	= shaders/shader.vert
FILE_SHADER_FRAGMENT_SOURCE	= shaders/shader.frag
FILE_SHADER_VERTEX_OBJECT	= shaders/vertex.spv
FILE_SHADER_FRAGMENT_OBJECT	= shaders/fragment.spv

DIRICTORY_INCLUDE			= -I$(PATH_TOOLCHAIN)/include
DIRICTORY_LIBRARY			= -L$(PATH_TOOLCHAIN)/lib

FLAGS_WARNINGS	= -Wall -Wextra -Wpedantic -Werror -Wconversion -Wsign-conversion -Wshadow		\
					-Wcast-align -Wstrict-prototypes -Wmissing-prototypes -Wredundant-decls		\
					-Wmissing-declarations -Wold-style-definition -Wnull-dereference -Wcast-qual\
					-Wwrite-strings -Wpointer-arith -Wundef -Wno-float-equal -Wjump-misses-init \
					-Wformat-security -Wvla -Wdouble-promotion -Wmissing-include-dirs			\
					-Wmissing-field-initializers

FLAGS_SPEED				= -march=native -funroll-loops -fuse-linker-plugin -flto -O2
FLAGS_SAVE				= -fstack-protector-strong -fstack-clash-protection -D_FORTIFY_SOURCE=2

FLAGS_BASE				= -std=c17 -pthread $(DIRICTORY_INCLUDE)

FLAGS_DEBUG				= $(FLAGS_BASE) $(FLAGS_WARNINGS) -g -O0
FLAGS_RELEASE			= $(FLAGS_BASE) $(FLAGS_WARNINGS) $(FLAGS_SAVE) $(FLAGS_SPEED) -DNDEBUG


FLAGS_LIBRARIES_BASE	= $(DIRICTORY_LIBRARY) -lglfw3 -lopengl32 -lglew32 -lvulkan-1
FLAGS_LIBRARIES_DEBUG	= $(FLAGS_LIBRARIES_BASE)
FLAGS_LIBRARIES_RELEASE	= $(FLAGS_LIBRARIES_BASE) -mwindows

.PHONY: all debug release

debug: FLAGS_TO_COMPILE		= $(FLAGS_DEBUG)
debug: FLAGS_LIBRARIES		= $(FLAGS_LIBRARIES_DEBUG)

release: FLAGS_TO_COMPILE	= $(FLAGS_RELEASE)
release: FLAGS_LIBRARIES	= $(FLAGS_LIBRARIES_RELEASE)

all: release

$(FILE_SHADER_FRAGMENT_OBJECT): $(FILE_SHADER_FRAGMENT_SOURCE)
	$(COMPILER_GLSL) $< -o $@

$(FILE_SHADER_VERTEX_OBJECT): $(FILE_SHADER_VERTEX_SOURCE)
	$(COMPILER_GLSL) $< -o $@

debug release: $(FILE_SHADER_VERTEX_OBJECT) $(FILE_SHADER_FRAGMENT_OBJECT)
	$(COMPILER_C) $(DIRICTORY_SOURCES) -o $(FILE_RESULT) $(FLAGS_TO_COMPILE) $(FLAGS_LIBRARIES)

