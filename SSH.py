# Update the next three lines with your
# server's information
import paramiko
import time

ip = "192.168.38.149"
host = "ev3dev"
port = "22"
username = "robot"
password = "maker"

command = "df"

client = paramiko.client.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, port, username, password, look_for_keys=False)


# _stdin, _stdout,_stderr = client.exec_command("./drive 5")
# print(_stdout.read().decode())

def drive(seconds="5", speed="66", backward=""):
    ssh_command = ".\drive" + seconds + " " + speed + " " + backward
    _stdin, _stdout, _stderr = client.exec_command(ssh_command)
    print(_stdout.read().decode())
    print(ssh_command)

drive()
# client.exec_command("./drive 5 66 gfdst")
# time.sleep(3)
# client.exec_command("./turn 3 r")
client.close()