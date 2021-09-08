# C compiler
CC = g++
CC_FLAGS = -g -fopenmp -O2 -I../../../common

bfs: 
	$(CC) $(CC_FLAGS) -DENABLE_RODINIA_HOOKS srad.cpp -o srad 

clean:
	rm -f srad
