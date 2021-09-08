CC = g++
SRC = pathfinder.cpp
EXE = pathfinder
FLAGS = -fopenmp -I../../common -DENABLE_RODINIA_HOOKS

release:
	$(CC) $(SRC) $(FLAGS) -o $(EXE)

debug:
	$(CC) $(SRC) -g -Wall -o $(EXE)

clean:
	rm -f pathfinder


