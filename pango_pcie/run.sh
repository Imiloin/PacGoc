echo "**************************编译开始***************************"
# 尝试运行 python3 -m pybind11 --includes 并检查结果
pybind_includes=$(python3 -m pybind11 --includes 2>&1)
status=$?

if [ $status -ne 0 ]; then
    echo "警告: 执行 'python3 -m pybind11 --includes' 失败。请确保 pybind11 已正确安装。"
    exit
else
    echo "pybind11 includes: $pybind_includes"
fi
make
make cp
echo "**************************编译完成***************************"
