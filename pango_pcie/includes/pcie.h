#ifndef PCIE_H
#define PCIE_H

#include <pybind11/numpy.h>
#include <stdexcept>
#include <vector>
#include "audio.h"
#include "config.h"
#include "dma.h"

namespace py = pybind11;

class PCIE {
   public:
    PCIE();
    ~PCIE();
    void transfer();
    // std::vector<unsigned char> fetch_pack();
    py::array_t<unsigned char> fetch_pack();

   private:
    int pci_driver_fd;
    DMA_OPERATION dma_operation;
};

#endif  // PCIE_H
