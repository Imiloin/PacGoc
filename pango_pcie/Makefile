# Python扩展模块的名称
Target = pcie$(shell python3-config --extension-suffix)
# 编译器
CC = g++
# 存放中间文件的路径
build_dir = build
# 存放源文件的文件夹
src_dir = sources
# 存放头文件的文件夹
inc_dir = includes

pybind11_include_path = $(shell python3 -m pybind11 --includes)
python_ldflags = $(shell python3-config --ldflags)

sources = $(foreach dir,$(src_dir),$(wildcard $(dir)/*.cpp))
objects = $(patsubst %.cpp,$(build_dir)/%.o,$(notdir $(sources)))
includes = $(foreach dir,$(inc_dir),$(wildcard $(dir)/*.h))
CFLAGS = $(patsubst %, -I%, $(inc_dir)) $(pybind11_include_path) -fPIC -std=c++11 -O3
LDFLAGS = $(python_ldflags) -shared

# 编译工程
$(build_dir)/$(Target) : $(objects)  | create_build
	$(CC) $^ $(LDFLAGS) -o $@ -Wall

# 链接工程
$(build_dir)/%.o : $(src_dir)/%.cpp $(includes)  | create_build
	$(CC) -c $(CFLAGS) $< -o $@

.PHONY:clean check create_build cp

clean:
	rm -rf $(build_dir)

check:
	@echo $(CFLAGS)
	@echo $(CURDIR)
	@echo $(src_dir)
	@echo $(sources)
	@echo $(objects)

create_build:
	@mkdir -p $(build_dir)
cp:	
	cp ./build/$(Target) ../pacgoc/pcie_api/
