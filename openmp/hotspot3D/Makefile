CC = gcc
CCFLAGS = -g -fopenmp -O3
OUTPUT = *.out
HOOKS_CFLAGS = -I../../common -DENABLE_RODINIA_HOOKS

3D:
	$(CC) -o 3D $(CCFLAGS) 3D.c -lm  $(HOOKS_CFLAGS)

clean:
	rm -f 3D $(OUTPUT)
