# 定义变量
CC = gcc
CXX = g++
CFLAGS = -fPIC -I./header
CXXFLAGS = -fPIC -I./header
LDFLAGS = -shared -L./lib
TARGET = libawnn_t527.so

SRCS_C = $(wildcard src/*.c)
SRCS_CXX = $(wildcard src/*.cpp)
OBJS_C = $(SRCS_C:.c=.o)
OBJS_CXX = $(SRCS_CXX:.cpp=.o)
OBJS = $(OBJS_C) $(OBJS_CXX)
LIBS = $(wildcard lib/*.so)
HEADERS = $(wildcard header/*.h)
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)
	cp $(TARGET) ../

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 清理规则
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean install uninstall