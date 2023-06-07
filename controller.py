# Example code for continuously updating navigation data
import time

while True:
    # Calculate or obtain the latest navigation data
    target_x = input("x: ")
    target_y = input("y: ")

    # Write navigation data to a text file
    with open('nav_data.txt', 'w') as file:
        file.write(f'{target_x},{target_y}')

    # Wait for a short duration before updating again
    time.sleep(1)
