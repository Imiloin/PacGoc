#ifndef AUDIO_H
#define AUDIO_H

#include <cstdint>
#include <iostream>
#include "config.h"

#define SAMPLE_RATE 48000
#define CHANNELS 1
#define BITS_PER_SAMPLE 16  // 16 bits per sample
#define BYTES_PER_SAMPLE (BITS_PER_SAMPLE / 8)

#define PACK_HEADER 0xaaaa  // 包头
#define PACK_HEADER_SIZE 8  // 包头中的样本数
#define PACK_DATA_SIZE 128       // 一个数据包中的样本数

#endif  // AUDIO_H
