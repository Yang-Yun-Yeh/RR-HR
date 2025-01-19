from serial import Serial
import csv
from datetime import datetime
from godirect import gdx
import time


client = Serial("/dev/ttyUSB0", 921600)
gdx = gdx.gdx()

gdx.open_usb()
gdx.select_sensors([1, 2])  # Force, RR
gdx.start(period=50)

print("Warming up!")
time.sleep(5)
print(f"Lift off...")

start_time = time.time()
duration = 180
now = datetime.now().strftime("%m%d_%H%M")

#with open(f"csv/sit_{now}.csv", "a") as f:
with open(f"csv/try_{now}.csv", "a") as f:
    w = csv.writer(f)
    w.writerow(["Timestamp",
                "imu1_q_x", "imu1_q_y", "imu1_q_z", "imu1_q_w",
                "imu2_q_x", "imu2_q_y", "imu2_q_z", "imu2_q_w",
                "Force", "RR"])

    try:
        while time.time() - start_time < duration:
            raw = client.readline()
            resp = gdx.read()
            data = raw.decode('utf-8').strip()
            if data.startswith("$"):
                item = data[1:].strip().replace("#", ",")
                results = [float(x) for x in item.split(',')]
                results = results + resp
                if results:
                    t = datetime.now().isoformat()
                    row = [t] + results
                    w.writerow(row)
                    f.flush()  
    finally:
        client.close()
        gdx.stop()
        gdx.close()

print("Exiting...")
