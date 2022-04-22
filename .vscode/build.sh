#!/bin/bash
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )""/../"
location=$(pwd)'/'
regex='s:./:'$location':'

# -Weffc++ has false positives
# OpenGL specific after MEMTRACE
# -lXi, -lXmu unknown

g++ \
-g \
-std=c++17 \
-Wall \
-Wextra \
-pedantic \
-fsanitize=undefined \
-fsanitize-undefined-trap-on-error \
-fsanitize=address \
-lm \
-D MEMTRACE \
-lGL \
-lGLU \
-lglut \
-lGLEW \
-lXi \
-lXmu \
\
framework.cpp \
$(find . -name '*.cpp' | sed 's:./framework.cpp::' | sed '$regex' | tr '\n' ' ') \
\
-o \
main \
