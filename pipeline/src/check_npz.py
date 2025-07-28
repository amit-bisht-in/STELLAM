import numpy as np

# 1. Set the path to your .npz file
file_path = r"D:\Projects\STELLA\pipeline\outputs\keypoints_full.npz"

try:
    # 2. Load the data from the file
    data = np.load(file_path)

    # 3. Print the names of the arrays stored inside
    print(f"Arrays found in file: {data.files}")

    # 4. Access the 'body' keypoints and check its shape
    if 'body' in data.files:
        body_data = data['body']
        print(f"Shape of the 'body' array: {body_data.shape}")

        # 5. Check if the entire 'body' array is composed of zeros
        is_all_zeros = np.all(body_data == 0)
        print(f"Is the 'body' array all zeros? -> {is_all_zeros}")

        # You can do the same for other parts
        # face_data = data['face']
        # print(f"Is the 'face' array all zeros? -> {np.all(face_data == 0)}")

except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")