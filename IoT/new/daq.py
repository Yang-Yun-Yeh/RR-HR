from serial import Serial
import csv
from datetime import datetime
from godirect import gdx
import time


client = Serial("/dev/ttyUSB0", 921600) # imu1, imu2
gdx = gdx.gdx() # rr gt

gdx.open_usb()
gdx.select_sensors([1, 2])  # Force, RR
gdx.start(period=50) # 50

print("Warming up!")
time.sleep(5)
print(f"Lift off...")

start_time = time.time()
duration = 625 # 315
now = datetime.now().strftime("%m%d_%H%M")
start_pt = 100

folder = "csv/3_27"
activity = "test"

import os
if not os.path.exists(folder):
    os.makedirs(folder)

with open(f"{folder}/{activity}_{now}.csv", "a") as f:
    w = csv.writer(f)
    w.writerow(["Timestamp",
                "imu1_q_x", "imu1_q_y", "imu1_q_z", "imu1_q_w",
                "imu2_q_x", "imu2_q_y", "imu2_q_z", "imu2_q_w",
                "Force", "RR"])

    try:
        count = 0 # wait for quaternions to stabilize
        while time.time() - start_time < duration:
            # print(time.time() - start_time)
            raw = client.readline()
            resp = gdx.read()
            data = raw.decode('utf-8').strip()

            if data.startswith("$"):
                item = data[1:].strip().replace("#", ",")
                results = [float(x) for x in item.split(',')]
                results = results + resp
                # if count >= 300 and results:
                if count >= start_pt and results: # start record: 5 sec
                    if count >= start_pt + 300:
                        print(time.time() - start_time)
                    t = datetime.now().isoformat()
                    row = [t] + results
                    w.writerow(row)
                    f.flush()  
                count += 1
    finally:
        client.close()
        gdx.stop()
        gdx.close()

print("Exiting...")
