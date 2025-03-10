CXX                = g++ -std=c++17
OPTFLAGS	   = -O3
AUTOFLAGS          = -march=native -funroll-loops -ffast-math
AVXFLAGS           = -march=native -funroll-loops -ffast-math
CXXFLAGS          += -Wall 
INCLUDES	   = -I. -I./include
LIBS               = #-pthread -fopenmp
SOURCES            = $(wildcard *.cpp)
TARGET             = $(SOURCES:.cpp=)

.PHONY: all clean cleanall runall

%: %.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTFLAGS) -o $@ $< $(LIBS)

# For files with _avx in their name, append flags to CXXFLAGS
%_avx: CXXFLAGS += ${AVXFLAGS}

# For files with _auto in their name, append flags to CXXFLAGS
%_auto: CXXFLAGS += ${AUTOFLAGS}

all: $(TARGET)

clean: 
	-rm -fr *.o *~
cleanall: clean
	-rm -fr $(TARGET)

runall: all
	./softmax_plain $(k)
	./softmax_auto $(k) 
	./softmax_avx $(k) 

runall_dbg: all
	./softmax_plain $(k) 1
	./softmax_auto $(k) 1
	./softmax_avx $(k) 1
