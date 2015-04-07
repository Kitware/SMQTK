CC=g++
CFLAGS=-c -O3 -Wall
LDFLAGS=-O3 -Wall
SOURCES=kernel_libSVMList.cpp libSVMListData.cpp kernelizerList.cpp
OBJS=$(SOURCES:.cpp=.o)
EXECUTABLE=kernel_libSVMList 

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $@
.cpp.o:
	$(CC) $(CFLAGS) $< -o $@


clean:
	rm -rf core *.o 
