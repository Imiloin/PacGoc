#!/bin/sh
# 装载 PCIe 驱动程序

driver="pango_pci_driver"
target="app"
root_name="root"
temp_name=$(whoami)
if [ "$root_name" = "$temp_name" ]; then   # 判断操作用户是否为 root
	if [ $(lsmod | grep -o "$driver") ]; then # 判断 PCIe 驱动是否已经装载
		echo "****************************PCIe 驱动已装载***************************"
	else
		echo "*************************开始编译 PCIe 驱动程序************************"
		make clean
		make
		echo "***************************开始装载 PCIe 驱动**************************"
		insmod $driver.ko  # 装载驱动
		if [ $(lsmod | grep -o "$driver") ]; then  # 判断 PCIe 驱动是否已经装载成功
			echo "***************************PCIe 驱动装载成功**************************"
		else
			echo "***************************PCIe 驱动装载失败**************************"
			exit
		fi
	fi
else
	echo "该脚本默认管理员用户：$root_name"
	echo "控制台当前操作用户：$temp_name"
	echo "请将操作用户切换为 $root_name（管理员）"
	exit
fi
