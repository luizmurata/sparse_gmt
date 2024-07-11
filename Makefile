CC = nvcc
INCS = -I./include
LIBS = -lcusparse
CXXFLAGS = -std=c++20 ${INCS} #-DDEBUG
LDFLAGS = ${LIBS}

SRC = src/utils.cu src/coo.cu src/csr.cu src/main.cu
OBJ = ${SRC:src/%.cu=build/%.o}

all: options transpose

options:
	@echo build options:
	@echo "CFLAGS   = ${CXXFLAGS}"
	@echo "LDFLAGS  = ${LDFLAGS}"
	@echo "CC       = ${CC}"

build/%.o: src/%.cu
	@mkdir -p build
	${CC} ${CXXFLAGS} -c $< -o $@ ${LDFLAGS}

clean: build transpose
	@rm -rf build transpose

transpose: ${OBJ}
	${CC} -o $@ ${OBJ} ${LDFLAGS}