CC=nvcc
CFLAGS=--std=c++11 `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`
test: test.cpp morphology.cu
	$(CC) $(CFLAGS) $(LIBS) -o $@ $^
