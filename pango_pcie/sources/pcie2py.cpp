#include "pcie2py.h"

namespace py = pybind11;

// 模块名称（example）作为第一个宏参数给出（不应用引号引起来）。
// 第二个参数（m）定义一个类型的 py::module_ 变量，该变量是创建绑定的主接口。
PYBIND11_MODULE(pcie, m) {
    py::class_<PCIE>(m, "PCIE")
        .def(py::init<>())
        .def("transfer", &PCIE::transfer)
        .def("fetch_pack", &PCIE::fetch_pack);
}
