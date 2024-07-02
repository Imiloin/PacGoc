#include "pcie.h"

namespace py = pybind11;

PCIE::PCIE() {
    pci_driver_fd = -1;
    pci_driver_fd = open_pci_driver();
    if (pci_driver_fd == 0) {
        throw std::runtime_error("PCIe Device Open Fail !!!");
    }
    init_dma(&dma_operation);
}

PCIE::~PCIE() {
    close(pci_driver_fd);
}

void PCIE::transfer() {
    dma_transfer(pci_driver_fd, &dma_operation);
}

// std::vector<unsigned char> PCIE::fetch_pack() {
//     std::vector<unsigned char> vec(dma_operation.data.read_buf, dma_operation.data.read_buf + DMA_PACKET_SIZE);
//     return vec;
// }

py::array_t<unsigned char> PCIE::fetch_pack() {
    // 假设DMA_PACKET_SIZE是数据包的大小
    return py::array_t<unsigned char>({DMA_PACKET_SIZE},           // NumPy数组的形状
                                      {sizeof(unsigned char)},     // NumPy数组的步长
                                      dma_operation.data.read_buf  // 数据
    );
}
