import numpy as np
import pandas as pd
import os
import cv2
from skinematics import imus


def convert_gps(gps_file_path: str):
    df = pd.read_csv(gps_file_path)
    print(f"{df.columns=}")
    time = df['cts'] / 1000.  # ms to s

    lat = df['GPS (Lat.) [deg]']
    lon = df['GPS (Long.) [deg]']
    alt = df['GPS (Alt.) [m]']
    speed = df['GPS (2D speed) [m/s]']

    # convert to vista gps format and save
    df_gps = pd.DataFrame({'time': time,
                           'latitude': lat,
                           'longitude': lon,
                           'altitude': alt,
                           'variance_latitude': [0.] * len(time),
                           'variance_longitude': [0.] * len(time),
                           'variance_altitude': [0.] * len(time)})
    df_gps.to_csv(os.path.join(os.path.dirname(gps_file_path), 'gps.csv'), index=False)

    # convert to vista speed (2s) format and save
    df_speed = pd.DataFrame({'time': time,
                             'speed': speed})
    df_speed.to_csv(os.path.join(os.path.dirname(gps_file_path), 'speed.csv'), index=False)
    print("Gps conversion finished!")


def convert_imu(gyro_file_path: str, acc_file_path: str):
    df_gyro = pd.read_csv(gyro_file_path)
    df_acc = pd.read_csv(acc_file_path)
    print(f"{df_acc.columns=}")
    print(f"{df_gyro.columns=}")
    time = df_gyro['cts'] / 1000.  # ms to s, same timestamps for gyro and acc
    time = np.asarray(time)
    print(f"Read {len(time)} data.")
    rate = (len(time) - 1) / (time[-1] - time[0])  # Hz

    ax = df_acc['Accelerometer (x) [m/s2]']
    ay = df_acc['Accelerometer (y) [m/s2]']
    az = df_acc['Accelerometer (z) [m/s2]']
    rx = df_gyro['Gyroscope (x) [rad/s]']
    ry = df_gyro['Gyroscope (y) [rad/s]']
    rz = df_gyro['Gyroscope (z) [rad/s]']

    # integrate to get camera orientation in quaternion format
    q, pos, vel = imus.analytical(omega=np.column_stack((rx, ry, rz)),
                                  accMeasured=np.column_stack((ax, ay, az)),
                                  rate=rate)

    df_imu = pd.DataFrame({'time': time,
                           'ax': ax,
                           'ay': ay,
                           'az': az,
                           'rx': rx,
                           'ry': ry,
                           'rz': rz,
                           'qx': q[:, 1],
                           'qy': q[:, 2],
                           'qz': q[:, 3],
                           'qw': q[:, 0]})
    df_imu.to_csv(os.path.join(os.path.dirname(gyro_file_path), 'imu.csv'), index=False)
    print("Imu conversion finished!")


def convert_video(video_file_path: str):
    cap = cv2.VideoCapture(video_file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video fps is {fps}")
    timestamps = []
    while cap.isOpened():
        frame_exists, curr_frame = cap.read()
        if frame_exists:
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.)
        else:
            break
    cap.release()
    print(f"Captured {len(timestamps)} frames.")

    df_frame = pd.DataFrame({'#frame_num': range(len(timestamps)),
                             'ros_time': timestamps})
    df_frame.to_csv(os.path.join(os.path.dirname(video_file_path), 'camera_front.csv'), index=False)
    print("Camera frame timestamps saved!")

    df_master_clock = pd.DataFrame({'timestamp': timestamps,
                                    'camera_front': range(len(timestamps))})
    df_master_clock.to_csv(os.path.join(os.path.dirname(video_file_path), 'master_clock.csv'), index=False)
    print("Master clock csv saved!")


if __name__ == '__main__':
    gps_path = 'vista_traces/boat_wabash/GL010362_Hero7 Black-GPS5.csv'
    gyro_path = 'vista_traces/boat_wabash/GL010362_Hero7 Black-GYRO.csv'
    acc_path = 'vista_traces/boat_wabash/GL010362_Hero7 Black-ACCL.csv'
    video_path = 'vista_traces/boat_wabash/camera_front.mp4'

    # convert_gps(gps_path)
    # convert_imu(gyro_path, acc_path)
    convert_video(video_path)


