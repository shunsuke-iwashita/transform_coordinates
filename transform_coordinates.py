import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import os
import random


def tracking_to_court_coordinates(video_num):
    # Load tracking data
    tracking_file = f'asset/tracking/{video_num}/1-1.txt'
    if not os.path.exists(tracking_file):
        return None
    tracking_data = pd.read_csv(tracking_file, sep=' ', header=None, names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z', 'class'])

    # Load homography matrix
    matrix_file = f'asset/homography_matrix/homography_matrix_{video_num}_1-1.npy'
    homography_matrix = np.load(matrix_file)

    # Get player's coordinates
    tracking_data['x_center'] = tracking_data['bb_left'] + tracking_data['bb_width'] / 2
    tracking_data['y_center'] = tracking_data['bb_top'] + tracking_data['bb_height']

    # Convert tracking coordinates to court coordinates
    for i in range(homography_matrix.shape[0]):
        frame_data = tracking_data[tracking_data['frame'] == i+1]
        frame_points = np.array([frame_data['x_center'].values, frame_data['y_center'].values, np.ones(frame_data.shape[0])])

        converted_points = np.dot(homography_matrix[i], frame_points).T
        converted_points = converted_points / converted_points[:, 2].reshape(-1, 1)

        tracking_data.loc[tracking_data['frame'] == i+1, 'x_court'] = converted_points[:, 0]
        tracking_data.loc[tracking_data['frame'] == i+1, 'y_court'] = converted_points[:, 1]

    # Fill NaN values
    tracking_data['x_court'] = tracking_data['x_court'].fillna(0)
    tracking_data['y_court'] = tracking_data['y_court'].fillna(0)

    # Drop unnecessary columns
    tracking_data = tracking_data.drop(columns=['x_center', 'y_center'])

    return tracking_data


def create_field():
    # Define the field size
    field_height, field_width = 3400, 3600
    # Create a field image
    field_image = np.ones((field_height, field_width, 3), dtype=np.uint8) * 255

    # Draw vertical lines
    vertical_lines = [
        (0, 0, 0, 2800),
        (3000, 0, 3000, 2800),
        (1010, 2800, 1010, 1640),
        (1990, 2800, 1990, 1640),
        (180, 2800, 180, 2200),
        (2820, 2800, 2820, 2200),
        (1250, 2560, 1250, 2485),
        (1750, 2560, 1750, 2485)
    ]

    for line in vertical_lines:
        cv2.line(field_image, (line[0] + 300, line[1] + 300), (line[2] + 300, line[3] + 300), (0, 0, 0), 2)

    # Draw horizonral lines
    horizontal_lines = [
        (0, 0, 3000, 0),
        (0, 2800, 3000, 2800),
        (1010, 1640, 1990, 1640)
    ]

    for line in horizontal_lines:
        cv2.line(field_image, (line[0] + 300, line[1] + 300), (line[2] + 300, line[3] + 300), (0, 0, 0), 2)

    # Draw arcs
    arcs = [
        ((1800, 1940), (360, 360), 0, 180, 360),
        ((1800, 2785), (250, 250), 0, 180, 360),
        ((1800, 2785), (1350, 1350), 0, 192, 348),
        ((1800, 300), (360, 360), 0, 0, 180)
    ]

    for arc in arcs:
        center, axes, angle, start_angle, end_angle = arc
        cv2.ellipse(field_image, center, axes, angle, start_angle, end_angle, (0, 0, 0), 2)

    return field_image


def plot_on_field(data, field_image, video_num):
    # Define the field size
    field_height, field_width = field_image.shape[0], field_image.shape[1]

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    frame_size = (field_width, field_height)
    output_video_file = f'result/video/{video_num}/plot.mp4'
    if os.path.exists(os.path.dirname(output_video_file)):
        os.makedirs(os.path.dirname(output_video_file), exist_ok=True)

    # Initialize the video writer
    out = cv2.VideoWriter(output_video_file, fourcc, fps, frame_size)

    # Get the maximum frame number
    max_frame = int(data['frame'].max())

    def draw_object(field, row):
        # Calculate the position on the field
        x, y = int(row['x_court'] * 200 + 300), int(row['y_court'] * 200 + 300)

        # Draw a circle at the position
        cv2.circle(field, (x, y), 30, (0, 0, 255), -1)

        # Add id
        cv2.putText(field, str(int(row['id'])), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

    # Iterate through each frame
    for current_frame in tqdm(range(1, max_frame+1)):
        # Filter the data for the current frame
        frame_data = data[data['frame'] == current_frame]

        # Copy the field image
        field = field_image.copy()

        # Apply the draw_object function to each row in the frame_data
        frame_data.apply(lambda row: draw_object(field, row), axis=1)

        # Add the frame number
        cv2.putText(field, f'Frame: {current_frame}', (300, 250), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 2)

        # Write the frame to the video
        out.write(field)

    # Release the video writer and close all OpenCV windows
    out.release()
    cv2.destroyAllWindows()


def court_coordinates_to_video_coordinates(court_coordinates, video_num):
    # Load court coordinates
    data = court_coordinates.copy()

    # Load homography matrix
    matrix_file = f'asset/homography_matrix/homography_matrix_{video_num}_1.npy'
    homography_matrix = np.load(matrix_file)

    # Convert court coordinates to video coordinates
    for i in range(homography_matrix.shape[0]):
        frame_data = data[data['frame'] == i+1]
        frame_points = np.array([frame_data['x_court'].values, frame_data['y_court'].values, np.ones(frame_data.shape[0])])


        if homography_matrix[i, 2, 2] != 0:
            homography_matrix[i] = np.linalg.inv(homography_matrix[i])
        converted_points = np.dot(homography_matrix[i], frame_points).T
        converted_points = converted_points / converted_points[:, 2].reshape(-1, 1)

        data.loc[data['frame'] == i+1, 'x_video'] = converted_points[:, 0]
        data.loc[data['frame'] == i+1, 'y_video'] = converted_points[:, 1]

    # Fill NaN values
    data['x_video'] = data['x_video'].fillna(0)
    data['y_video'] = data['y_video'].fillna(0)

    # Get player's coordinates
    data.loc[data['x_court'] != 0, 'bb_width'] = 100
    data.loc[data['x_court'] != 0, 'bb_height'] = 100
    data.loc[data['x_court'] == 0, 'bb_height'] = 0
    data.loc[data['x_court'] == 0, 'bb_width'] = 0

    data['bb_left'] = data['x_video'] - data['bb_width'] / 2
    data['bb_top'] = data['y_video'] - data['bb_height'] / 2

    # Drop unnecessary columns
    data = data.drop(columns=['x_court', 'y_court', 'x_video', 'y_video'])

    return data


def plot_on_video(data, video_num):
    # Load the video
    video_file = f'asset/video/{video_num}/1.mov'
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f'Error opening video file {video_file}')
        return

    # Get the video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    output_path = f'result/video/{video_num}/bbox.mp4'
    if os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

    frame_id = 1
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id in range(1, int(data['frame'].max()) + 1):
                for index, row in data[data['frame'] == frame_id].iterrows():
                    color = generate_unique_color(int(row['id']))
                    cv2.rectangle(frame, (int(row['bb_left']), int(row['bb_top'])), (int(row['bb_left']) + int(row['bb_width']), int(row['bb_top']) + int(row['bb_height'])), color, 2)
                    cv2.putText(frame, str(int(row['id'])), (int(row['bb_left']), int(row['bb_top']) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            out.write(frame)
            frame_id += 1
            pbar.update(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def generate_unique_color(id):
    """
    Generates a unique color for a given ID using a deterministic random process.

    Args:
        id (int): The object ID for which to generate a color.

    Returns:
        tuple: A tuple representing the color in BGR format.
    """
    random.seed(id)
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def main():
    for video_num in range(1, 7):
        # Convert tracking data to court coordinates
        court_coordinates = tracking_to_court_coordinates(video_num)

        if court_coordinates is None:
            continue

        video_coordinates = court_coordinates_to_video_coordinates(court_coordinates, video_num)

        # Save the data
        # court_coordinates_path = f'result/coordinates/{video_num}'
        # os.makedirs(court_coordinates_path, exist_ok=True)
        # court_coordinates.to_csv(f'{court_coordinates_path}/court_coordinates.txt', index=False)
        video_coordinates_path = f'result/coordinates/{video_num}'
        os.makedirs(video_coordinates_path, exist_ok=True)
        video_coordinates.to_csv(f'{video_coordinates_path}/video_coordinates.txt', index=False)

        # Set filed
        field_image = create_field()

        # Plot the tracking data on the field
        plot_on_field(court_coordinates, field_image, video_num)
        # # Save the video
        # plot_on_video(video_coordinates, video_num)


if __name__ == '__main__':
    main()