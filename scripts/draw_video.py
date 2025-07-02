from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import pandas as pd
import typer

app = typer.Typer()


def create_trajectory_map(coords: np.ndarray, map_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Create a trajectory map from coordinates.

    Args:
        coords: Array of (x, y) coordinates
        map_size: Tuple of (width, height) for the map

    Returns:
        Tuple of (map image as numpy array with trajectory drawn, pixel coordinates)
    """
    map_width, map_height = map_size

    # Create white background
    map_img = np.ones((map_height, map_width, 3), dtype=np.uint8) * 255

    # Normalize coordinates to fit in map - SWAP X AND Y for proper visualization
    x_coords, y_coords = coords[:, 1], coords[:, 0]  # Swapped x and y

    # Add margin
    margin = 50
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # Scale coordinates to map size with margin
    x_range = x_max - x_min
    y_range = y_max - y_min

    if x_range == 0:
        x_range = 1
    if y_range == 0:
        y_range = 1

    scale_x = (map_width - 2 * margin) / x_range
    scale_y = (map_height - 2 * margin) / y_range
    scale = min(scale_x, scale_y)  # Use same scale for both axes to maintain aspect ratio

    # Convert to pixel coordinates
    x_pixels = ((x_coords - x_min) * scale + margin).astype(int)
    y_pixels = ((y_coords - y_min) * scale + margin).astype(int)

    # Draw trajectory line
    points = np.column_stack((x_pixels, y_pixels))
    for i in range(len(points) - 1):
        cv2.line(map_img, tuple(points[i]), tuple(points[i + 1]), (100, 100, 100), 2)

    return map_img, points


def draw_current_position(map_img: np.ndarray, position: tuple[int, int],
                         timestamp_text: str) -> np.ndarray:
    """Draw current position (X-mark) and timestamp on the trajectory map.

    Args:
        map_img: Map image to draw on
        position: Current position in pixel coordinates (x, y)
        timestamp_text: Timestamp text to display

    Returns:
        Map image with current position marked
    """
    map_copy = map_img.copy()
    x, y = position

    # Draw X-mark (red cross)
    size = 10
    thickness = 3
    color = (0, 0, 255)  # Red in BGR

    # Draw X
    cv2.line(map_copy, (x - size, y - size), (x + size, y + size), color, thickness)
    cv2.line(map_copy, (x - size, y + size), (x + size, y - size), color, thickness)

    # Draw circle around X
    cv2.circle(map_copy, (x, y), size + 5, color, 2)

    # Add timestamp text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_color = (0, 0, 0)  # Black
    thickness = 2

    # Get text size for positioning
    (text_width, text_height), _ = cv2.getTextSize(timestamp_text, font, font_scale, thickness)

    # Position text at top of map
    text_x = (map_copy.shape[1] - text_width) // 2
    text_y = 30

    # Add white background for text
    cv2.rectangle(map_copy, (text_x - 5, text_y - text_height - 5),
                  (text_x + text_width + 5, text_y + 5), (255, 255, 255), -1)
    cv2.putText(map_copy, timestamp_text, (text_x, text_y), font, font_scale, text_color, thickness)

    return map_copy


def resize_camera_image(img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize camera image to target size while maintaining aspect ratio.

    Args:
        img: Input image
        target_size: Target (width, height)

    Returns:
        Resized image
    """
    if img is None:
        # Return black image if original is None
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    target_width, target_height = target_size
    h, w = img.shape[:2]

    # Calculate scale to fit within target size
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize image
    resized = cv2.resize(img, (new_w, new_h))

    # Create target size image and center the resized image
    result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    start_x = (target_width - new_w) // 2
    start_y = (target_height - new_h) // 2
    result[start_y:start_y + new_h, start_x:start_x + new_w] = resized

    return result


def create_combined_frame(front_img: np.ndarray, back_img: np.ndarray,
                         map_img: np.ndarray, camera_width: int, camera_height: int,
                         map_width: int, map_height: int) -> np.ndarray:
    """Combine front camera, back camera, and trajectory map into single frame.

    Args:
        front_img: Front camera image
        back_img: Back camera image
        map_img: Trajectory map image with current position
        camera_width: Width of camera panels
        camera_height: Height of camera panels
        map_width: Width of map panel
        map_height: Height of map panel

    Returns:
        Combined frame image
    """
    total_width = camera_width + map_width
    total_height = map_height

    # Create output frame
    combined_frame = np.zeros((total_height, total_width, 3), dtype=np.uint8)

    # Resize camera images
    front_resized = resize_camera_image(front_img, (camera_width, camera_height))
    back_resized = resize_camera_image(back_img, (camera_width, camera_height))

    # Resize map to fit right side
    map_resized = cv2.resize(map_img, (map_width, map_height))

    # Place images in frame
    # Front camera: top-left
    combined_frame[0:camera_height, 0:camera_width] = front_resized

    # Back camera: bottom-left
    combined_frame[camera_height:map_height, 0:camera_width] = back_resized

    # Map: right side
    combined_frame[0:map_height, camera_width:total_width] = map_resized

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color = (255, 255, 255)  # White
    thickness = 2

    cv2.putText(combined_frame, "Front Camera", (10, 30), font, font_scale, color, thickness)
    cv2.putText(combined_frame, "Back Camera", (10, camera_height + 30), font, font_scale, color, thickness)
    cv2.putText(combined_frame, "Trajectory Map", (camera_width + 10, 30), font, font_scale, color, thickness)

    return combined_frame


@app.command()
def main(
    input_dir: Annotated[Path, typer.Argument(help="Path to the input directory containing track.csv and camera folders")],
    output_dir: Annotated[Path | None, typer.Option(help="Output directory for the video")] = None,
    fps: Annotated[int, typer.Option(help="Frames per second for the output video")] = 5,
) -> None:
    """Create a visualization video from trajectory data and camera images.

    This tool processes a dataset containing:
    \n\t- track.csv with trajectory coordinates and timestamps
    \n\t- front_cam/ folder with front camera images
    \n\t- back_cam/ folder with back camera images

    \nThe output video shows front camera, back camera, and trajectory map with current position.
    """
    # Set default output directory to current working directory if not provided
    if output_dir is None:
        output_dir = Path.cwd()
    # Frame layout settings
    camera_width = 640
    camera_height = 480
    map_width = 800
    map_height = 960  # Height for both camera views combined
    total_width = camera_width + map_width
    total_height = map_height

    # Validate input directory
    if not input_dir.exists():
        typer.echo(f"Error: Input directory {input_dir} does not exist", err=True)
        raise typer.Exit(1)

    track_csv = input_dir / "track.csv"
    if not track_csv.exists():
        typer.echo(f"Error: track.csv not found in {input_dir}", err=True)
        raise typer.Exit(1)

    front_cam_dir = input_dir / "front_cam"
    back_cam_dir = input_dir / "back_cam"

    if not front_cam_dir.exists():
        typer.echo(f"Error: front_cam directory not found in {input_dir}", err=True)
        raise typer.Exit(1)

    if not back_cam_dir.exists():
        typer.echo(f"Error: back_cam directory not found in {input_dir}", err=True)
        raise typer.Exit(1)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate video filename from input directory name
    track_name = input_dir.name
    video_output_path = output_dir / f"{track_name}_visualization.mp4"

    typer.echo("Loading trajectory data...")
    track_df = pd.read_csv(track_csv)

    # Extract data arrays
    timestamps = track_df["timestamp"].values
    front_cam_timestamps = track_df["front_cam_ts"].values
    back_cam_timestamps = track_df["back_cam_ts"].values
    coords = track_df[["tx", "ty"]].values

    typer.echo(f"Loaded {len(track_df)} trajectory points")

    # Create trajectory map from all coordinates
    typer.echo("Creating trajectory map...")
    trajectory_map, pixel_coords = create_trajectory_map(coords, (map_width, map_height))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(video_output_path), fourcc, fps, (total_width, total_height))

    typer.echo(f"Creating video with {len(track_df)} frames at {fps} fps...")

    # Process all frames
    for i, (_timestamp, front_ts, back_ts, (tx, ty)) in enumerate(
        zip(timestamps,
            front_cam_timestamps,
            back_cam_timestamps,
            coords,
            strict=True)
    ):
        if (i + 1) % 50 == 0 or i == 0:  # Progress update every 50 frames
            typer.echo(f"Processing frame {i + 1}/{len(track_df)}")

        # Load camera images
        front_img_path = front_cam_dir / f"{front_ts}.jpg"
        back_img_path = back_cam_dir / f"{back_ts}.jpg"

        front_img = cv2.imread(str(front_img_path))
        back_img = cv2.imread(str(back_img_path))

        if front_img is None:
            typer.echo(f"Warning: Front image not found: {front_img_path}", err=True)
        if back_img is None:
            typer.echo(f"Warning: Back image not found: {back_img_path}", err=True)

        # Create timestamp text - coordinates are now swapped back for display
        timestamp_text = f"Frame {i + 1} - Position: ({ty:.2f}, {tx:.2f})"

        # Draw current position on trajectory map
        current_pixel_pos = tuple(pixel_coords[i])
        map_with_position = draw_current_position(trajectory_map, current_pixel_pos, timestamp_text)

        # Create combined frame
        combined_frame = create_combined_frame(
            front_img, back_img, map_with_position,
            camera_width, camera_height, map_width, map_height
        )

        # Write frame to video
        video_writer.write(combined_frame)

    # Clean up
    video_writer.release()

    typer.echo(f"Video saved to: {video_output_path}")
    typer.echo("Video creation completed!")


if __name__ == "__main__":
    app()
