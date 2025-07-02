CXX := g++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra
LDFLAGS := 

SRCS := main.cpp wolf.cpp SquareLattice.cpp
OBJS := $(SRCS:.cpp=.o)
TARGET := ising_simulation

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

main.o: main.cpp wolf.h SquareLattice.h
	$(CXX) $(CXXFLAGS) -c main.cpp

wolf.o: wolf.cpp wolf.h SquareLattice.h
	$(CXX) $(CXXFLAGS) -c wolf.cpp

SquareLattice.o: SquareLattice.cpp SquareLattice.h
	$(CXX) $(CXXFLAGS) -c SquareLattice.cpp

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)

clean-data:
	rm -f *.dat

clean-all: clean clean-data

.PHONY: all run clean clean-data clean-all