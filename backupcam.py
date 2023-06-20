import cv2
import numpy as np
import math
import paramiko
import time


ip = "192.168.186.149"
host = "ev3dev"
port = "22"
username = "robot"
password = "maker"

command = "df"

client = paramiko.client.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, port, username, password, look_for_keys=False)


def drive(seconds="5", speed="30", backward=""):
    ssh_command = ".\drive" + seconds + " " + speed + " " + backward
    _stdin, _stdout, _stderr = client.exec_command(ssh_command)
    print(_stdout.read().decode())
    _stdin, _stdout, _stderr = client.exec_command("./drive 5 2")
    print(_stdout.read().decode())


def lift(seconds="4.5", speed="66", backward=""):
    ssh_command = ".\drive" + seconds + " " + speed + " " + backward
    _stdin, _stdout, _stderr = client.exec_command(ssh_command)
    print(_stdout.read().decode())
    _stdin, _stdout, _stderr = client.exec_command("./lift 5, 20")
    print(_stdout.read().decode())


def turn_right(seconds="5", speed="66", backward=""):
    ssh_command = ".\drive" + seconds + " " + speed + " " + backward
    _stdin, _stdout, _stderr = client.exec_command(ssh_command)
    print(_stdout.read().decode())
    _stdin, _stdout, _stderr = client.exec_command("./turn 10")
    print(_stdout.read().decode())


#   print(ssh_command)

def turn_left(seconds="5", speed="66", backward=""):
    ssh_command = ".\drive" + seconds + " " + speed + " " + backward
    _stdin, _stdout, _stderr = client.exec_command(ssh_command)
    print(_stdout.read().decode())
    _stdin, _stdout, _stderr = client.exec_command("./turn 0, 10")
    print(_stdout.read().decode())


def calculate_distance(point1, point2):
    # Calculate the Euclidean distance between two points
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


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

def detect_table_tennis_balls(frame, rect_bottom_left, rect_top_right):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 30, 100)

    # Find contours in the edge image for ball detection
    ball_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter ball contours based on area and circularity
    min_ball_area = 30
    max_ball_area = 200
    min_ball_circularity = 0.6

    detected_balls = []

    for contour in ball_contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)

            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            if area > min_ball_area and area < max_ball_area and circularity > min_ball_circularity:
                if rect_bottom_left[0] < x < rect_top_right[0] and rect_bottom_left[1] < y < rect_top_right[1]:
                    detected_balls.append((x, y, radius))

    return detected_balls


def detect_blue_and_green_robots(frame, rect_bottom_left, rect_top_right, min_robot_area):
    # Define the lower and upper boundaries for the green and blue color ranges
    lower_green = np.array([20, 100, 100]) #np.array([35, 50, 50])
    upper_green = np.array([40, 255, 255]) #np.array([80, 255, 255])
    lower_blue = np.array([0, 0, 0]) #np.array([100, 50, 50])
    upper_blue = np.array([255, 255, 30]) #np.array([130, 255, 255])

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for green and blue regions
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours for green and blue regions
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter green and blue contours based on position and size within the specified rectangle
    green_robot = None
    blue_robot = None

    for contour in green_contours:
        x, y, w, h = cv2.boundingRect(contour)
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        area = w * h

        if rect_bottom_left[0] < x < rect_top_right[0] and rect_bottom_left[1] < y < rect_top_right[
            1] and area > min_robot_area:
            green_robot = (top_left, bottom_right)

    for contour in blue_contours:
        x, y, w, h = cv2.boundingRect(contour)
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        area = w * h

        if rect_bottom_left[0] < x < rect_top_right[0] and rect_bottom_left[1] < y < rect_top_right[
            1] and area > min_robot_area:
            if green_robot is None or not (top_left[0] > green_robot[1][0] and bottom_right[0] < green_robot[0][0]):
                blue_robot = (top_left, bottom_right)

    return green_robot, blue_robot



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
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    min_area = 500

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            break

        # Detect table tennis balls
        detected_balls = detect_table_tennis_balls(frame, rect_bottom_left, rect_top_right)

        # Detect blue robots
        green_robot, blue_robot = detect_blue_and_green_robots(frame, rect_bottom_left, rect_top_right, min_area)

        # Draw detected circles for balls
        for (x, y, radius) in detected_balls:
            center = (int(x), int(y))
            cv2.circle(frame, center, radius, (0, 255, 0), 2)

        # Draw rectangles for robots
        if green_robot is not None:
            top_left, bottom_right = green_robot
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        if blue_robot is not None:
            top_left, bottom_right = blue_robot
            cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)

        # Display the frame with detected balls and robots
        cv2.imshow("Table Tennis Ball and Blue Robot Detection", frame)

        # Print the coordinates of the detected balls
        for i, (x, y, radius) in enumerate(detected_balls):
            print(f"Ball {i + 1}: Coordinate ({x}, {y})")

        # Print the coordinates of the detected robots
        if green_robot is not None:
            print("Green Robot: Top Left =", green_robot[0], "Bottom Right =", green_robot[1])

        if blue_robot is not None:
            print("Blue Robot: Top Left =", blue_robot[0], "Bottom Right =", blue_robot[1])


        # Calculate the distance and angle between the robot and the nearest ball
        if detected_balls and blue_robot:
            robot_center = ((blue_robot[0][0] + blue_robot[1][0]) // 2,
                            (blue_robot[0][1] + blue_robot[1][1]) // 2)
            ball_centers = [(int(relative_x + rect_bottom_left[0]), int(rect_top_right[1] - relative_y))
                           for relative_x, relative_y, _ in detected_balls]
            distances = [calculate_distance(robot_center, ball_center) for ball_center in ball_centers]
            min_distance = min(distances)
            nearest_ball_index = distances.index(min_distance)
            nearest_ball_center = ball_centers[nearest_ball_index]
            angle = calculate_angle(robot_center, nearest_ball_center)
            print(f"Distance to nearest ball: {min_distance:.2f} pixels")
            print(f"Angle to face nearest ball: {angle:.2f} degrees")
            if min_distance < 30:
                drive()
            else:
                print("BEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEPPPPP")

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and destroy windows
    cap.release()
    cv2.destroyAllWindows()


# Call the function to detect table tennis balls and blue robots from the camera
detect_table_tennis_balls_and_robots()
