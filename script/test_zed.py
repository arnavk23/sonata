import pyzed.sl as sl
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

    output_dir = "captured_frames"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    frame_count = 0
    max_frames = 5

    while frame_count < max_frames:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            
            filename = os.path.join(output_dir, f"frame_{frame_count + 1}.png")
            image.write(filename)  # Save the image to disk
            
            print(f"Captured frame {frame_count + 1}, saved to: {os.path.abspath(filename)}")
            frame_count += 1
        else:
            print("Frame grab failed.")

    zed.close()
    print("Camera closed.")

if __name__ == "__main__":
    main()

