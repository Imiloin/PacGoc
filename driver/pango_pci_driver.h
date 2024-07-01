#ifndef PANGO_PCI_DRIVER_H
#define PANGO_PCI_DRIVER_H

#include <linux/pci.h>
#include <linux/interrupt.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/spinlock.h>
#include <linux/fs.h>
#include <linux/types.h>
#include <linux/kdev_t.h>
#include <linux/cdev.h>
#include <asm/uaccess.h>
#include <linux/uaccess.h>
#include <linux/version.h>
#include <linux/dma-mapping.h>
#include <linux/delay.h>
#include <linux/string.h>
#include <linux/ctype.h>



#define TYPE 'S'
#define PCI_READ_DATA_CMD    			_IOWR(TYPE, 0, int)							/* 读数据指令 */
#define PCI_WRITE_DATA_CMD    			_IOWR(TYPE, 1, int)							/* 写数据指令 */
#define PCI_MAP_ADDR_CMD    			_IOWR(TYPE, 2, int)							/* DMA总线地址映射 */
#define PCI_WRITE_TO_KERNEL_CMD    		_IOWR(TYPE, 3, int)							/* 写内核数据操作 */
#define PCI_DMA_READ_CMD    			_IOWR(TYPE, 4, int)							/* DMA读操作 */
#define PCI_DMA_WRITE_CMD    			_IOWR(TYPE, 5, int)							/* DMA写操作 */
#define PCI_READ_FROM_KERNEL_CMD    	_IOWR(TYPE, 6, int)							/* 读取内核数据操作 */
#define PCI_UMAP_ADDR_CMD    			_IOWR(TYPE, 7, int)							/* 释放映射地址 */
#define PCI_PERFORMANCE_START_CMD		_IOWR(TYPE, 8, int)							/* 性能测试开始操作 */
#define PCI_PERFORMANCE_END_CMD			_IOWR(TYPE, 9, int)							/* 性能测试结束操作 */

#define MAX_BLOCK_SIZE   				1024										/* 位流分块缓存区最大值 */
#define BAR_OFFSET_1					0x00										/* 读取PCI状态地址偏移地址 */
#define BAR_OFFSET_2					0x10										/* 配置PCI加载偏移地址 */
#define BAR_OFFSET_3					0x20										/* 加载位流偏移地址 */
#define DMA_MAX_PACKET_SIZE   			4096										/* DMA包最大值 */
#define CMD_REG_OFFSET 					0x100										/* 读写控制寄存器偏移量 */
#define RW_ADDR_LO_OFFSET 				0x110										/* MEM访问地址低30位 */
#define RW_ADDR_HI_OFFSET 				0x120										/* MEM访问地址高32位 */

struct PciDriverDevInfo
{
	dev_t        _dev;
	unsigned int _dev_firstminor;
	unsigned int _dev_count;
	char        *_dev_name;
};

struct PangoPciDriver
{
	int               _pci_bar;
	int               _pci_io_size;
	void             *_pci_io;
	void             *_pci_io_buff;
	struct pci_driver _pci_driver;
};

struct PciPango
{
	struct cdev           _cdev;
	struct PangoPciDriver _pango_pci_driver;
	struct semaphore      _sem;
	struct class         *_cdev_class;
};

typedef struct _BAR_INFO_
{
	unsigned long bar_base;
	unsigned long bar_len;
}BAR_BASE_INFO;

typedef struct _CAP_INFO_
{
	unsigned char flag;
	unsigned char id;
	unsigned char addr_offset;
	unsigned char next_offset;
}CAP_INFO;

typedef struct _CAP_LIST_
{
	unsigned char cap_status;
	unsigned char cap_error;
	CAP_INFO cap_buf[256];
}CAP_LIST;

typedef struct _PCI_INFO_
{
	unsigned int vendor_id;
	unsigned int device_id;
	unsigned int cmd_reg;
	unsigned int status_reg;
	unsigned int revision_id;
	unsigned int class_prog;
	unsigned int class_device;
	BAR_BASE_INFO bar[6];
	unsigned int min_gnt;
	unsigned int max_lat;
	unsigned int link_speed;
	unsigned int link_width;
	unsigned int mps;
	unsigned int mrrs;
	unsigned int data[1024];
}PCI_DEVICE_INFO;

typedef struct _LOAD_DATA_
{
	unsigned int num_words;
	unsigned int block_words[MAX_BLOCK_SIZE];	
}LOAD_DATA_INFO;

typedef struct _PCI_LOAD_
{
	unsigned char link_status;
	unsigned int  crc;
	unsigned char axi_direction;
	unsigned char load_status;
	unsigned int total_num_words;
	LOAD_DATA_INFO data_block;
}PCI_LOAD_INFO;


typedef struct _COMMAND_
{
	unsigned char w_r;
	unsigned char step;
	unsigned int addr;
	unsigned int data;
	unsigned int cnt;
	unsigned int delay;
	PCI_DEVICE_INFO get_pci_dev_info;
	CAP_LIST cap_info;
	PCI_LOAD_INFO load_info;
}COMMAND_OPERATION;


COMMAND_OPERATION command_operation;

typedef struct _CONFIG_
{
	unsigned int addr;
	unsigned int data;
}CONFIG_OPERATION;

COMMAND_OPERATION config_operation;

typedef struct _DMA_DATA_
{
	unsigned char read_buf[DMA_MAX_PACKET_SIZE];
	unsigned char write_buf[DMA_MAX_PACKET_SIZE];
}DMA_DATA;

typedef struct _DMA_OPERATION_
{
	unsigned int current_len;
	unsigned int offset_addr;
	unsigned int cmd;
	DMA_DATA data;
}DMA_OPERATION;

DMA_OPERATION dma_operation;

typedef union _DMA_CMD_
{
    struct _cmd1_
    {
        unsigned short length    : 10;
        unsigned char  reserved1 : 6;
        unsigned char  addr_type : 1;
        unsigned char  reserved2 : 7;
        unsigned char  op_type   : 1;
        unsigned char  reserved3 : 7;
    } data;

    unsigned int value;
}DMA_CMD;

typedef struct _DMA_ADDR_
{
	char addr_size;
	dma_addr_t addr;
	void* data_buf;
	spinlock_t lock;
}DMA_ADDR;

typedef struct _DMA_INFO_
{
	DMA_CMD cmd;
	DMA_ADDR addr_r;
	DMA_ADDR addr_w;
}DMA_INFO;

DMA_INFO dma_info;

typedef union _DMA_PERFORMANCE_CMD_
{
    struct _cmd2_
    {
        unsigned short length    : 10;
        unsigned char  reserved1 : 6;
        unsigned char  addr_type : 1;
        unsigned char  reserved2 : 7;
        unsigned char  op_type   : 2;
        unsigned char  reserved3 : 6;
    } data;

    unsigned int value;
}DMA_PERFORMANCE_CMD;

typedef struct _DMA_PERFORMANCE_CONFIG_
{
	DMA_PERFORMANCE_CMD cmd;
	DMA_ADDR addr;
}DMA_PERFORMANCE_CONFIG;

DMA_PERFORMANCE_CONFIG performance_config;

typedef struct _PERFORMANCE_OPERATION_
{
	unsigned int current_len;
	unsigned int cmd;
	unsigned char cmp_flag;
}PERFORMANCE_OPERATION;

PERFORMANCE_OPERATION performance_operation;

//cdev file operation
loff_t  pango_cdev_llseek(struct file *filp, loff_t off, int whence);
ssize_t pango_cdev_read(struct file *filp, char __user *buf, size_t count, loff_t *f_pos);
ssize_t pango_cdev_write(struct file *filp, const char __user *buf, size_t count, loff_t *f_pos);
int     pango_cdev_open(struct inode *inode, struct file *filp);
long 	pango_cdev_ioctl(struct file *file, unsigned int cmd, unsigned long arg);
int     pango_cdev_release(struct inode *inode, struct file *filp);

//pci driver functions
int  pci_driver_probe(struct pci_dev *dev, const struct pci_device_id *device_id);
void pci_driver_remove(struct pci_dev *dev);

int init_pango_cdev(struct cdev *pango_cdev);
int init_pango_pci_driver(struct pci_driver *pango_pci_driver);
int init_pango_cdev_class(struct class *pango_cdev_class);

void exit_pango_cdev(struct cdev *pango_cdev);
void exit_pango_pci_driver(struct pci_driver *pango_pci_driver);
void exit_pango_cdev_class(struct class *pango_cdev_class);

#endif // PANGO_PCI_DRIVER_H
