CXX := g++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra
LDFLAGS := 

SRC_DIR := src
SRCS := $(SRC_DIR)/main.cpp $(SRC_DIR)/wolf.cpp $(SRC_DIR)/SquareLattice.cpp
OBJS := $(SRCS:.cpp=.o)
TARGET := ising_simulation

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(SRC_DIR)/main.o: $(SRC_DIR)/main.cpp $(SRC_DIR)/wolf.h $(SRC_DIR)/SquareLattice.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(SRC_DIR)/wolf.o: $(SRC_DIR)/wolf.cpp $(SRC_DIR)/wolf.h $(SRC_DIR)/SquareLattice.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(SRC_DIR)/SquareLattice.o: $(SRC_DIR)/SquareLattice.cpp $(SRC_DIR)/SquareLattice.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)

clean-data:
	rm -f *.dat

clean-all: clean clean-data

.PHONY: all run clean clean-data clean-all
