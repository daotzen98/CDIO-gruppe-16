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


def drive(seconds="3", speed="50", backward=""):
    ssh_command = "./drive " + seconds + " " + speed + " " + backward
    _stdin, _stdout, _stderr = client.exec_command(ssh_command)
    #print(_stdout.read().decode())
    #_stdin, _stdout, _stderr = client.exec_command("./drive 3 60")
    #print(_stdout.read().decode())


def lift(seconds="4.5", speed="66", backward=""):
    client.exec_command("./lift")
    reverse()
    time.sleep(4)
    client.exec_command("./down")

def ralese(seconds="4.5", speed="66", backward=""):
    client.exec_command("./open")
    reverse()
    time.sleep(4)


def turn_right(seconds="5", turns="340", backward=""):
    command = f'./lasthope {turns}'
    _stdin, _stdout, _stderr = client.exec_command(command=command)
    print(command)
    #_stdin, _stdout, _stderr = client.exec_command("./turn 20000")
    #print(_stdout.read().decode())


#   print(ssh_command)

def turn_left(seconds="5", speed="40", backward=""):
    ssh_command = ".\drive" + seconds + " " + speed + " " + backward
    _stdin, _stdout, _stderr = client.exec_command(ssh_command)
    print(_stdout.read().decode())
    _stdin, _stdout, _stderr = client.exec_command("./turn 20000, 10")
    print(_stdout.read().decode())

def reverse(seconds="5", speed="30", backward=""):
    ssh_command = "./drive" + seconds + " " + speed + " " + backward
    _stdin, _stdout, _stderr = client.exec_command(ssh_command)
    print(_stdout.read().decode())
    _stdin, _stdout, _stderr = client.exec_command("./drive 2 20 dsada")
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


def detect_table_tennis_balls(frame, rect_bottom_left, rect_top_right):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 0, 100)
    cv2.imshow("rest", edges)

    # Find contours in the edge image for ball detection
    ball_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter ball contours based on area and circularity
    min_ball_area = 30
    max_ball_area = 200
    min_ball_circularity = 0.5

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


def detect_black_and_yellow_robots(frame, rect_bottom_left, rect_top_right, min_robot_area):
    # Define the lower and upper boundaries for the yellow and black color ranges
    lower_yellow = np.array([20, 50, 200])  # np.array([35, 50, 50])
    upper_yellow = np.array([40, 255, 255])  # np.array([80, 255, 255])
    lower_black = np.array([0, 0, 0])  # np.array([100, 50, 50])
    upper_black = np.array([255, 255, 50])  # np.array([130, 255, 255])

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for yellow and black regions
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    # Find contours for yellow and black regions
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    black_contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter yellow and black contours based on position and size within the specified rectangle
    yellow_robot = None
    black_robot = None

    for contour in yellow_contours:
        x, y, w, h = cv2.boundingRect(contour)
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        area = w * h

        if rect_bottom_left[0] < x < rect_top_right[0] and rect_bottom_left[1] < y < rect_top_right[
            1] and area > min_robot_area:
            yellow_robot = (top_left, bottom_right)

    for contour in black_contours:
        x, y, w, h = cv2.boundingRect(contour)
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        area = w * h

        if rect_bottom_left[0] < x < rect_top_right[0] and rect_bottom_left[1] < y < rect_top_right[
            1] and area > min_robot_area:
            if yellow_robot is None or not (top_left[0] > yellow_robot[1][0] and bottom_right[0] < yellow_robot[0][0]):
                black_robot = (top_left, bottom_right)

    return yellow_robot, black_robot


def detect_red_lines(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper boundaries for the red color range
    lower_red = np.array([0, 50, 20])
    upper_red = np.array([255, 255, 255])

    # Apply color thresholding to detect red regions
    red_mask = cv2.inRange(hsv, lower_red, upper_red)

    # Perform edge detection on the red regions
    edges = cv2.Canny(red_mask, 50, 150)

    # Perform line detection using Hough Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

    # Filter and return the detected lines
    if lines is not None:
        lines = lines.reshape(-1, 4)
    return lines


def find_intersection_points(lines):
    intersection_points = []

    # Calculate the angles between the lines and find the intersection points
    for i in range(len(lines) - 1):
        x1, y1, x2, y2 = lines[i]
        angle1 = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

        for j in range(i + 1, len(lines)):
            x3, y3, x4, y4 = lines[j]
            angle2 = np.arctan2(y4 - y3, x4 - x3) * 180 / np.pi

            # Check if the angle difference is within the desired range
            angle_diff = np.abs(angle1 - angle2)
            if angle_diff >= 80 and angle_diff <= 100:
                # Calculate the intersection point
                det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if det != 0:
                    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
                    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det
                    intersection_point = (int(px), int(py))

                    # Check if the intersection point is at least 50 pixels away from other points
                    if all(np.linalg.norm(np.array(intersection_point) - np.array(p)) >= 50 for p in
                           intersection_points):
                        intersection_points.append(intersection_point)

    return intersection_points[:4]  # Limit to the closest four intersection points


def detect_table_tennis_balls_and_robots():
    lastangle = 0
    lifts = 0
    count = 1000
    # Open the camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Define the dimensions of the rectangle for ball detection
    rect_bottom_left = (20, 20)
    rect_top_right = (600, 450)  # Initial values (adjust as needed)

    # Calculate the width and height of the rectangle
    rect_width = rect_top_right[0] - rect_bottom_left[0]
    rect_height = rect_top_right[1] - rect_bottom_left[1]

    min_area = 500

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            break



        # Detect table tennis balls
        detected_balls = detect_table_tennis_balls(frame, rect_bottom_left, rect_top_right)

        # Detect black robots
        yellow_robot, black_robot = detect_black_and_yellow_robots(frame, rect_bottom_left, rect_top_right, min_area)

        # Detect red lines and get the detected lines
        # detected_lines = detect_red_lines(frame)

        # Find intersection points of the lines
        # intersection_points = find_intersection_points(detected_lines)

        # Draw the detected lines
        #if detected_lines is not None:
        #    for line in detected_lines:
        #        x1, y1, x2, y2 = line
        #        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

        # Draw the intersection points
        #if intersection_points:
        #    for point in intersection_points:
        #        cv2.circle(frame, point, 5, (255, 255, 255), -1)

        # Draw detected circles for balls
        for (x, y, radius) in detected_balls:
            center = (int(x), int(y))
            cv2.circle(frame, center, radius, (0, 255, 0), 2)

        goal = (600, 240)

        cv2.circle(frame, goal, 2, (255,255,0), 2)

        # Draw rectangles for robots
        if yellow_robot is not None:
            top_left, bottom_right = yellow_robot
            cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)

        if black_robot is not None:
            top_left, bottom_right = black_robot
            cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)

        # cv2.rectangle(frame, rect_top_right, rect_bottom_left, (255, 0, 0), 2)

        # Display the frame with detected balls and robots
        cv2.imshow("Table Tennis Ball and Black Robot Detection", frame)

        # Print the coordinates of the detected balls
        # for i, (x, y, radius) in enumerate(detected_balls):
        #    print(f"Ball {i + 1}: Coordinate ({x}, {y})")

        # Print the coordinates of the detected robots
        # if yellow_robot is not None:
        #    print("Yellow Robot: Top Left =", yellow_robot[0], "Bottom Right =", yellow_robot[1])

        # if black_robot is not None:
        #    print("Black Robot: Top Left =", black_robot[0], "Bottom Right =", black_robot[1])

        # Calculate the distance and angle between the robot and the nearest ball
        if detected_balls and black_robot and yellow_robot:
            black_robot_center = ((black_robot[0][0] + black_robot[1][0]) // 2,
                                  (black_robot[0][1] + black_robot[1][1]) // 2)

            yellow_robot_center = ((yellow_robot[0][0] + yellow_robot[1][0]) // 2,
                                   (yellow_robot[0][1] + yellow_robot[1][1]) // 2)

            ball_centers = [(int(x), int(y)) for (x, y, radius) in detected_balls]

            distances = [calculate_distance(black_robot_center, ball_center) for ball_center in ball_centers]
            min_distance = min(distances)

            # Find the indices of balls that are at least 10 pixels away from the robots
            valid_ball_indices = [i for i, ball_center in enumerate(ball_centers) if
                                  calculate_distance(black_robot_center, ball_center) >= 30 and calculate_distance(
                                      yellow_robot_center, ball_center) >= 40]

            # Filter the ball centers and distances based on the valid indices
            filtered_ball_centers = [ball_centers[i] for i in valid_ball_indices]
            filtered_distances = [distances[i] for i in valid_ball_indices]

            if filtered_distances:
                nearest_ball_index = filtered_distances.index(min(filtered_distances))
                nearest_ball_center = filtered_ball_centers[nearest_ball_index]
            else:
                # Handle the case when there are no valid balls
                nearest_ball_index = None
                nearest_ball_center = None

            #min_distance = min(filtered_distances)
            if(lifts >= 3):
                nearest_ball_center = goal

            if nearest_ball_center is not None:
                angle = calculate_angle(yellow_robot_center, nearest_ball_center)
            else:
                angle = 0

            direction = calculate_angle(yellow_robot_center, black_robot_center)

            # print(f"the robot is pointing in the direction: {direction:.2f}")
            # print(f"Distance to nearest ball: {min_distance:.2f} pixels")
            # print(f"Angle to face nearest ball: {angle:.2f} degrees")
            # print(yellow_robot_center)
            # print(black_robot_center)
            # print(nearest_ball_center)
            # print(count)



            angledif = int(angle) - int(direction)
            lest = int(340*(angledif/90))

            test = str(int(lest))

            if(count > 300):
                count = 0

                if (lifts >= 3 ):
                    print("help")
                    print(direction - angle)
                    count=250
                    if((direction - angle) < 188 and (direction -angle) > 172):
                        drive("4", "20", "basd")
                        time.sleep(4)
                        ralese()
                        lifts = 0
                    elif((direction - angle) > 195 and (direction -angle) < 145):
                        turn_right("0","140")
                    else:
                        turn_right("0","40")
                        print(angledif)
                        lastangle =(direction)
                        print(test)
                        # print(min_distance)

                elif((lastangle < (direction +5) and lastangle > (direction -5)) and lifts < 10):
                    drive("1", "90")
                    time.sleep(1)
                    turn_right("0, 100")
                    time.sleep(1)
                    drive("1", "20", "dsda")

                elif(angledif < 6 and angledif > -6):
                    drive("2","45")
                    time.sleep(2)
                    lift()
                    time.sleep(2)
                    reverse()
                    lifts = lifts +1
                elif(angledif > 0):
                    turn_right(0, test)
                    count = 160

                else:
                    drive("1", "20", "dsad")
                    count = 160
                    turn_right(0, test)




        count= count+1
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and destroy windows
    cap.release()
    cv2.destroyAllWindows()


# Call the function to detect table tennis balls and black robots from the camera
drive("3","100")
time.sleep(3)
drive("3", "100", "dsada")
time.sleep(3)
detect_table_tennis_balls_and_robots()
