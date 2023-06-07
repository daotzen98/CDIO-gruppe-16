#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor
from pybricks.parameters import Port
from pybricks.tools import wait
from pybricks.parameters import Stop
from pybricks.robotics import DriveBase
import math

# Initialize the EV3 Brick.
ev3 = EV3Brick()

# Initialize the motors.
left_motor = Motor(Port.B)
right_motor = Motor(Port.C)
# shit_motor = Motor(Port.D)
# peter_motor = Motor(Port.A)

# Initialize the drive base.
wheel_diameter = 68.8
axle_track = 150
robot = DriveBase(left_motor, right_motor, wheel_diameter, axle_track)

# Read coordinates from file.
with open('nav_data.txt', 'r') as file:
    data = file.read().split(',')
    print(data)

# Initialize robot's current position
current_x = 0
current_y = 0
orientation = 0

# Iterate over each coordinate in the data array
for i in range(0, len(data), 2):
    # Extract target coordinates
    target_x = int(data[i])
    target_y = int(data[i + 1])

    print('target x:' + str(target_x))
    print('target y:' + str(target_y))
    
    # Calculate the differences in x and y coordinates
    delta_x = target_x - current_x
    delta_y = target_y - current_y
    
    # Update the robot's orientation
    if delta_x < 0:
        orientation = 180
    else:
        orientation = 0
    
    # Turn the robot to the new orientation if needed
    if orientation != 0:
        robot.turn(orientation)
    
    # Drive the robot straight to the target x coordinate
    robot.straight(abs(delta_x))

    print('delta x:' + str(delta_x))
    
    # Update the current position to the target coordinates
    current_x = target_x
    current_y = target_y
    
    print('current x and y:' + str(current_x) + ';' + str(current_y))
    # Turn the robot to face the target y coordinate
    if delta_y < 0:
        orientation = 270
    else:
        orientation = 90
    
    # Turn the robot to the new orientation if needed
    if orientation != 0:
        robot.turn(orientation)
    
    # Drive the robot straight to the target y coordinate
    print('delta y:' + str(delta_y))
    robot.straight(abs(delta_y))

