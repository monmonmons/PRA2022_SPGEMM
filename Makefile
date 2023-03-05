ROCM_PATH?= $(wildcard /opt/rocm)

HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
    HIP_PATH=../../..
endif

ROCSPARSE_INCLUDE = ${ROCM_PATH}/rocsparse/include
ROCSPARSE_LIB = ${ROCM_PATH}/rocsparse/lib/

HIPCC=$(HIP_PATH)/bin/hipcc

gpu = 0
ifeq ($(gpu),0)
	EXECUTABLE = Csrsparse
	CXXFLAGS = -O2 -I ./ 
	LDFLAGS = 
else ifeq ($(gpu),1)
	EXECUTABLE = Csrsparse_rocsparse
	CXXFLAGS = -Dgpu -O2 -I ./ -I ${ROCSPARSE_INCLUDE}
	LDFLAGS = -L ${ROCSPARSE_LIB} -lrocsparse
endif

OBJECTS = main.o

all:$(EXECUTABLE) 

$(EXECUTABLE): $(OBJECTS)
	$(HIPCC) $(OBJECTS) $(LDFLAGS) -o $@

%.o:%.cpp
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o
	rm -f ${EXECUTABLE}
