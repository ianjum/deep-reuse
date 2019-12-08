CCPP = g++
NVCC = nvcc

COMFLAGS += -I/usr/local/cuda/include
COMFLAGS += -I/usr/local/include
COMFLAGS += -I/usr/local
COMFLAGS += -L/usr/local
COMFLAGS += -L/usr/local/cuda/lib64
COMFLAGS += -std=c++11 -D GOOGLE_CUDA=1 -D_GLIBCXX_USE_CXX11_ABI=0
TF_LFLAGS = -L/home/ianjum/.local/lib/python2.7/site-packages/tensorflow
TF_CFLAGS += -I/home/ianjum/.local/lib/python2.7/site-packages/tensorflow/include
EXFLAGS = -fPIC -w
CPPFLAGS = -shared 

NVFLAGS = --expt-relaxed-constexpr -DNDEBUG
NVFLAGS += -x cu -Xcompiler
NVLINKFLAGS = -lcudart -lcublas -ltensorflow_framework 

TARGET = user_conv.so

all : $(TARGET)

objs = user_conv.cu.o

$(TARGET) : $(objs) user_conv.cc
	$(CCPP) $(CPPFLAGS) -o $(TARGET) $^ $(COMFLAGS) $(EXFLAGS) $(NVLINKFLAGS) ${TF_CFLAGS} ${TF_LFLAGS} -O2

user_conv.cu.o : user_conv.cu.cc
	$(NVCC) -c $< -o $@ $(COMFLAGS) $(NVFLAGS) $(EXFLAGS) ${TF_CFLAGS} ${TF_LFLAGS}

clean : 
	-rm -rf user_conv.cu.o user_conv.so
