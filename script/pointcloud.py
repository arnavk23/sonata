import pyzed.sl as sl
import numpy as np
import os

def main():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera")
        exit(1)

    print("ZED Camera opened successfully!")

    runtime_params = sl.RuntimeParameters()
    point_cloud = sl.Mat()

    output_dir = "pointclouds_csv"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_count = 0
    max_frames = 5

    while frame_count < max_frames:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve point cloud (XYZRGBA)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            
            # Get point cloud data as numpy array
            pc_data = point_cloud.get_data()  # shape: (height, width, 4) (X,Y,Z,RGBA)
            
            # Reshape to 2D array: (num_points, 4)
            height, width, _ = pc_data.shape
            pc_data = pc_data.reshape(-1, 4)
            
            # Filter out invalid points (where Z == 0)
            valid_points = pc_data[pc_data[:,2] != 0]
            
            # Save to CSV with columns: X, Y, Z, R, G, B, A
            csv_filename = os.path.join(output_dir, f"pointcloud_{frame_count + 1}.csv")
            
            # Split RGB from RGBA float into separate channels (0-255)
            # The last value is a float packed RGBA, so we convert it accordingly:
            # Actually, the ZED SDK packs RGBA as floats, so it's best to just save XYZ here or handle color separately.
            # For simplicity, save only XYZ here:
            
            xyz_only = valid_points[:, :3]  # X, Y, Z
            
            # Save as CSV
            np.savetxt(csv_filename, xyz_only, delimiter=",", header="X,Y,Z", comments='')
            
            print(f"Captured point cloud {frame_count + 1}, saved to: {os.path.abspath(csv_filename)}")
            frame_count += 1
        else:
            print("Frame grab failed.")

    zed.close()
    print("Camera closed.")

if __name__ == "__main__":
    main()

