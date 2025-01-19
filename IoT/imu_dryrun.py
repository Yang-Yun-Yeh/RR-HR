from serial import Serial
from godirect import gdx
import time
import math

dev = gdx.gdx()

dev.open_usb()
dev.select_sensors([2])
dev.start(period=50)

last_ts = None

try:
    while True:
        val = dev.read()[0]
        if not math.isnan(val):
            current_timestamp = time.time()
            print(f'RR: {val}, Ts: {current_timestamp}', end='')
            
            if last_ts is not None:
                duration = current_timestamp - last_ts
                print(f', Duration: {duration:.4f} seconds')
            else:
                print() 
                
            last_ts = current_timestamp

except KeyboardInterrupt:
    print("\nMonitoring stopped by user")
    dev.stop()
    dev.close()
except Exception as e:
    print(f"Error during monitoring: {e}")
    dev.stop()
    dev.close()
