#ifndef __DMA_H__
#define __DMA_H__

#include "audio.h"
#include "config.h"

#define DMA_PACKET_SIZE ((PACK_HEADER_SIZE + PACK_DATA_SIZE) * BYTES_PER_SAMPLE) /* 数据包字节数 */
#define DMA_PACKET_LEN DMA_PACKET_SIZE / 4 + 1 /* 一次传输的数据包大小（4字节） */

static_assert(DMA_PACKET_SIZE >= DMA_MIN_PACKET_SIZE && DMA_MAX_PACKET_SIZE <= DMA_MAX_PACKET_SIZE,
              "DMA_PACKET_SIZE is not valid.");

int open_pci_driver(void);
void print_data(DMA_OPERATION* dma_oper);
void dma_transfer(int pci_driver_fd, DMA_OPERATION* dma_oper);
void init_dma(DMA_OPERATION* dma_oper);

#endif /* __DMA_H__ */
