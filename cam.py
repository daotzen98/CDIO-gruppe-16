import cv2
import numpy as np
import math

def calculate_distance(point1, point2):
    # Calculate the Euclidean distance between two points
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_angle(robot_center, ball_center):
    # Calculate the angle between the robot's heading and the ball position
    delta_x = ball_center[0] - robot_center[0]
    delta_y = ball_center[1] - robot_center[1]
    return math.atan2(delta_y, delta_x) * 180 / math.pi

def detect_table_tennis_balls_and_robots():
    # Open the camera
    cap = cv2.VideoCapture(0)

    # Define the dimensions of the rectangle for ball detection
    rect_bottom_left = (20, 20)
    rect_top_right = (600, 450)  # Initial values (adjust as needed)

    # Calculate the width and height of the rectangle
    rect_width = rect_top_right[0] - rect_bottom_left[0]
    rect_height = rect_top_right[1] - rect_bottom_left[1]

    # Define the lower and upper boundaries for the blue color range
    lower_blue = np.array([90, 50, 50])  # Adjust lower threshold for darker blue
    upper_blue = np.array([130, 255, 255])

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 30, 100)

        # Find contours in the edge image for ball detection
        ball_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find blue regions in the frame for robot detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_white = cv2.inRange(frame, (200, 200, 200), (255, 255, 255))

        # Combine the blue and white masks
        mask = cv2.bitwise_or(mask_blue, mask_white)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        robot_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter ball contours based on area and circularity
        min_ball_area = 30
        max_ball_area = 1600
        min_ball_circularity = 0.7

        detected_balls = []

        for contour in ball_contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)

                if area > min_ball_area and area < max_ball_area and circularity > min_ball_circularity:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    radius = int(radius)

                    if rect_bottom_left[0] < x < rect_top_right[0] and rect_bottom_left[1] < y < rect_top_right[1]:
                        relative_x = x - rect_bottom_left[0]
                        relative_y = rect_top_right[1] - y
                        detected_balls.append((relative_x, relative_y, radius))

        # Filter robot contours based on size
        min_robot_area = 2000  # Adjust as needed

        detected_robots = []

        for contour in robot_contours:
            area = cv2.contourArea(contour)

            if area > min_robot_area:
                rect = cv2.boundingRect(contour)
                top_left = (rect[0], rect[1])
                bottom_right = (rect[0] + rect[2], rect[1] + rect[3])
                detected_robots.append((top_left, bottom_right))

        # Draw detected circles for balls
        for (relative_x, relative_y, radius) in detected_balls:
            center = (int(relative_x + rect_bottom_left[0]), int(rect_top_right[1] - relative_y))
            cv2.circle(frame, center, radius, (0, 255, 0), 2)

        # Draw rectangles for robot detection
        for (top_left, bottom_right) in detected_robots:
            cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)

        # Draw the rectangle for ball detection
        cv2.rectangle(frame, rect_bottom_left, rect_top_right, (0, 0, 255), 2)

        # Display the frame with detected balls and robots
        cv2.imshow("Table Tennis Ball and Blue Robot Detection", frame)

        # Calculate the distance and angle between the robot and the nearest ball
        if detected_balls and detected_robots:
            robot_center = ((detected_robots[0][0][0] + detected_robots[0][1][0]) // 2,
                            (detected_robots[0][0][1] + detected_robots[0][1][1]) // 2)
            ball_centers = [(int(relative_x + rect_bottom_left[0]), int(rect_top_right[1] - relative_y))
                            for relative_x, relative_y, _ in detected_balls]
            distances = [calculate_distance(robot_center, ball_center) for ball_center in ball_centers]
            min_distance = min(distances)
            nearest_ball_index = distances.index(min_distance)
            nearest_ball_center = ball_centers[nearest_ball_index]
            angle = calculate_angle(robot_center, nearest_ball_center)
            print(f"Distance to nearest ball: {min_distance:.2f} pixels")
            print(f"Angle to face nearest ball: {angle:.2f} degrees")

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and destroy windows
    cap.release()
    cv2.destroyAllWindows()


# Call the function to detect table tennis balls and blue robots from the camera
detect_table_tennis_balls_and_robots()
