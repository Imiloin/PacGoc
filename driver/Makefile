CONFIG_MODULE_SIG=n
ifeq ($(KERNELRELEASE),) 
# ifeq 后面一定要加空格
# 非命令行前面不能以 tab 开头，必须以空格开头
 KERNELDIR ?= /lib/modules/$(shell uname -r)/build  
 M=$(PWD) modules
 PWD := $(shell pwd)
modules:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules
modules_install:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules_install
clean:
	rm -rf *.o *~ core .depend .*.cmd *.ko *.mod *.mod.c .tmp_versions Module* modules*
.PHONY: modules modules_install clean
else
 obj-m :=pango_pci_driver.o
endif
