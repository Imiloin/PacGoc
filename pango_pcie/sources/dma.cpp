#include "dma.h"


// 函数声明在 dma.h 中
// dma_oper->current_len 单位为四字节

int open_pci_driver(void) {
    int fd;
    fd = open(PCIE_DRIVER_FILE_PATH, O_RDWR);
    if (fd < 0) {
        perror("open fail\n");
        return -1;
    }
    return fd;
}

void print_data(DMA_OPERATION* dma_oper) {
    unsigned int i = 0;
    for (i = 0; i < dma_oper->current_len; i++) {
        printf(
            "dw_cnt = %d; read_data = "
            "0x%02x%02x%02x%02x\n",
            i + 1, dma_oper->data.read_buf[i * 4], dma_oper->data.read_buf[i * 4 + 1],
            dma_oper->data.read_buf[i * 4 + 2], dma_oper->data.read_buf[i * 4 + 3]);
    }
}

void dma_transfer(int pci_driver_fd, DMA_OPERATION* dma_oper) {
    // 清空数据缓存
    // memset(dma_oper->data.read_buf, 0, DMA_MAX_PACKET_LEN);
    // 地址映射,以及数据缓存申请
    ioctl(pci_driver_fd, PCI_MAP_ADDR_CMD, dma_oper);
    // 将数据从设备读出到内核（DMA写）
    ioctl(pci_driver_fd, PCI_DMA_WRITE_CMD, dma_oper);
    // 等待数据写入完成
    usleep(100);  ////// change the delay time here
    // 将数据从内核读出
    ioctl(pci_driver_fd, PCI_READ_FROM_KERNEL_CMD, dma_oper);
    // 释放数据缓存
    ioctl(pci_driver_fd, PCI_UMAP_ADDR_CMD, dma_oper);
    // print_data(dma_oper);
}

void init_dma(DMA_OPERATION* dma_oper) {
    dma_oper->offset_addr = 0;
    dma_oper->current_len = DMA_PACKET_LEN;
}
