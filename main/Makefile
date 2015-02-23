# Configuration #
# Compilator
CC		= g++

# Executables to build 
EXEC		= nvteacher 

# Options (TRANSFER_TO_ROBOT_NAO)
OPT 		= TRANSFER_TO_ROBOT_NAO

# Sources files

# Source code directory
SRCDIRS 	= ../src

# Include directories
INCLUDEDIRS 	= ../include ../include/kmlocal ../include/densetrack

# Compilation and link flags
CFLAGS 		= $(patsubst %,-I%,$(subst :, ,$(INCLUDEDIRS))) -D$(OPT)
LDFLAGS 	= -lsvm -lkmeans -ltinyxml `pkg-config --libs opencv`

.PHONY: clean cleanall

all: $(EXEC)
$(EXEC): main.o naomngt.o naokmeans.o naosvm.o imconfig.o naodensetrack.o IplImageWrapper.o IplImagePyramid.o imbdd.o
	$(CC) -Wall -o $@  $^ -L../lib $(LDFLAGS)
main.o: main.cpp 
	$(CC) -Wall -o $@ -c $< $(CFLAGS)
naomngt.o: $(SRCDIRS)/naomngt.cpp 
	$(CC) -Wall -o $@ -c $< $(CFLAGS)
naokmeans.o: $(SRCDIRS)/naokmeans.cpp
	$(CC) -Wall -o $@ -c $< $(CFLAGS)
naosvm.o: $(SRCDIRS)/naosvm.cpp
	$(CC) -Wall -o $@ -c $< $(CFLAGS)
imconfig.o: $(SRCDIRS)/imconfig.cpp
	$(CC) -Wall -o $@ -c $< $(CFLAGS)
imbdd.o: $(SRCDIRS)/imbdd.cpp
	$(CC) -o $@ -c $< $(CFLAGS)
naodensetrack.o: $(SRCDIRS)/naodensetrack.cpp
	$(CC) -o $@ -c $< $(CFLAGS)
IplImageWrapper.o: $(SRCDIRS)/IplImageWrapper.cpp
	$(CC) -o $@ -c $< $(CFLAGS)
IplImagePyramid.o:$(SRCDIRS)/IplImagePyramid.cpp
	$(CC) -o $@ -c $< $(CFLAGS)
clean:
	rm -f *.o
cleanall: clean
	rm -f *.o $(EXEC)
