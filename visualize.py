import cv2
import numpy as np

# Function to draw a stylized border around the bounding boxes
def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=50, line_length_y=50):
    """
    Draws a stylized border around a bounding box.

    Args:
        img (ndarray): The image on which to draw.
        top_left (tuple): The (x, y) coordinates for the top-left corner of the bounding box.
        bottom_right (tuple): The (x, y) coordinates for the bottom-right corner of the bounding box.
        color (tuple): The color of the border in (B, G, R) format.
        thickness (int): The thickness of the border lines.
        line_length_x (int): The length of the horizontal lines for the stylized border.
        line_length_y (int): The length of the vertical lines for the stylized border.

    Returns:
        img (ndarray): Image with the drawn border.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Top-left corner
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    # Bottom-left corner
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    # Top-right corner
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    # Bottom-right corner
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

# Real-time visualization function for vehicle detection and license plate recognition
def visualize(results, video_source):
    """
    Visualize results in real-time for the given frame.

    Args:
        results (dict): A dictionary containing detection and tracking results for the current frame, in the format:
                        {car_id: {'car': {'bbox': [x1, y1, x2, y2]},
                                  'license_plate': {'bbox': [x1, y1, x2, y2], 'text': 'ABC123'}}}
        video_source (ndarray): The frame (image) to visualize results on.

    Returns:
        None
    """
    print("-------------------------------------------------------------")
    # Check if the video frame is valid
    if video_source is None:
        print("No video source provided for visualization.")
        return

    # Iterate through each detected car and its corresponding license plate information
    for car_id, car_data in results.items():
        # Draw car bounding box
        if 'car' in car_data:
            car_bbox = car_data['car']['bbox']
            xcar1, ycar1, xcar2, ycar2 = map(int, car_bbox)
            draw_border(video_source, (xcar1, ycar1), (xcar2, ycar2), (0, 255, 0), 10, line_length_x=50, line_length_y=50)

            # Add car ID label
            cv2.putText(video_source, f"Car ID: {car_id}", (xcar1, ycar1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw license plate bounding box
        if 'license_plate' in car_data:
            license_bbox = car_data['license_plate']['bbox']
            xlp1, ylp1, xlp2, ylp2 = map(int, license_bbox)
            cv2.rectangle(video_source, (xlp1, ylp1), (xlp2, ylp2), (0, 0, 255), 3)

            # Display license plate text below the bounding box
            license_text = car_data['license_plate'].get('text', 'Unknown')
            cv2.putText(video_source, f"Plate: {license_text}", (xlp1, ylp2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Resize frame for display purposes (optional)
    resized_frame = cv2.resize(video_source, (1280, 720))
    # return resized_frame
    # Show frame with real-time visualizations
    cv2.imshow("Vehicle and License Plate Detection", resized_frame)
    
    # Handle key events for quitting visualization
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        exit()  # Exit the script when 'q' is pressed







