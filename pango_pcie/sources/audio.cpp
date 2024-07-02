#include "audio.h"

bool dump_pcm(DMA_OPERATION* dma_oper, FILE* file) {
    uint16_t audio_data =
        (static_cast<uint16_t>(dma_oper->data.read_buf[1]) << 8) | static_cast<uint16_t>(dma_oper->data.read_buf[0]);

    // cout << hex << audio_data << endl;
    static int write_cnt = 0;  // 记录写入包的个数

    if (audio_data == PACK_HEADER) {  // 检测到包头
        // 打印包的内容
        if (write_cnt < MAX_PACK_COUNT) {
            std::cout << "bag begin" << std::endl;
            // 一行打印8个16位数据，分为16行打印
            for (int i = 1; i <= LINE_NUM; i++) {
                for (int j = 0; j < LINE_LEN; j++) {
                    uint16_t audio_data =
                        (static_cast<uint16_t>(dma_oper->data.read_buf[i * LINE_NUM + j * BYTES_PER_SAMPLE + 1]) << 8) |
                        static_cast<uint16_t>(dma_oper->data.read_buf[i * LINE_NUM + j * BYTES_PER_SAMPLE]);
                    printf("0x%04x ", audio_data);
                }
                std::cout << std::endl;
            }
            std::cout << "bag end" << std::endl;
        }
        // 写入文件
        if (file != NULL) {
            if (write_cnt < MAX_PACK_COUNT) {
                unsigned char* array = &(dma_oper->data.read_buf[PACK_HEADER_SIZE * BYTES_PER_SAMPLE]);
                fwrite(array, sizeof(unsigned char), PACK_DATA_SIZE * BYTES_PER_SAMPLE, file);
                write_cnt++;
                return true;
            } else if (write_cnt == MAX_PACK_COUNT) {
                write_cnt = 0;
                printf("writing file end.\n");
                return false;
            }
        } else {
            printf("Unable to open file for writing.\n");
            return false;
        }
    }
    return true;  // 未检测到包头，继续等待
}
