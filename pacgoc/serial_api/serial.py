import serial
import serial.tools.list_ports
import time


class Serial:
    BAUD_RATE = 115200

    def __init__(
        self,
        port: str = "",
        baudrate: int = 115200,
        bytesize: int = 8,
        parity: str = "E",
        stopbits: int = 1,
        timeout: float = 0.5,
    ):
        """
        Initialize serial port.
        """

        if port == "":
            # get available serial ports
            port_list = list(serial.tools.list_ports.comports())
            print(port_list)
            if len(port_list) == 0:
                print("No serial port found")
            else:
                # find Silicon Labs CP210x USB to UART Bridge
                for port_info in port_list:
                    print(port_info.description)
                    if (
                        "CP210x USB to UART Bridge" in port_info.description
                        or "CP2102 USB to UART Bridge" in port_info.description
                    ):
                        port_name = port_info.device
                        print(
                            f"Found Silicon Labs CP210x USB to UART Bridge on {port_name}"
                        )
                        break
                else:
                    print("Silicon Labs CP210x USB to UART Bridge not found")
        else:
            port_name = port

        self.ser = serial.Serial(
            port=port_name,
            baudrate=baudrate,
            bytesize=bytesize,
            parity=parity,
            stopbits=stopbits,
            timeout=timeout,
        )
        if self.ser.isOpen():
            print("Open serial port successfully.")
            print(self.ser.name)
        else:
            print("Failed to open serial port.")
            raise Exception("Failed to open serial port.")

    def __del__(self):
        """
        Close serial port when object is deleted.
        """
        if self.ser.isOpen():
            self.ser.close()
            print("Serial port closed.")

    def write(self, data: str | int | float | bytes | bytearray):
        """
        Write data to serial port.
        """
        if isinstance(data, str):
            data = data.encode("utf-8")
        elif isinstance(data, int):
            data = str(data).encode("utf-8")
        elif isinstance(data, float):
            data = str(data).encode("utf-8")
        elif not isinstance(data, (bytes, bytearray)):
            raise TypeError("Unsupported data type")
        self.ser.write(data)

    def read(self, size: int = 1, timeout: float = 0.5) -> str:
        """
        Read data from serial port.
        """
        start_time = time.time()
        wait_time = timeout / 10
        while True:
            if self.ser.in_waiting() >= size:
                return self.ser.read(size).decode("utf-8")
            elif time.time() - start_time > timeout:
                print("Timeout")
                return ""
            else:
                time.sleep(wait_time)
