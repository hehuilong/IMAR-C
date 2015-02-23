# Configuration #
# Compilator
CC		= g++

# Executables to build 
EXEC		= detfps 

# Sources files

# Source code directory
SRCDIRS 	= .

# Compilation and link flags
CFLAGS 		= `pkg-config --cflags opencv`
LDFLAGS 	= `pkg-config --libs opencv`

.PHONY: clean cleanall

all: $(EXEC)
$(EXEC): main.o
	$(CC) -Wall -o $@  $^ $(LDFLAGS)
main.o: main.cpp 
	$(CC) -Wall -o $@ -c $< $(CFLAGS)
clean:
	rm -f *~
cleanall: clean
	rm -f *.o $(EXEC)
