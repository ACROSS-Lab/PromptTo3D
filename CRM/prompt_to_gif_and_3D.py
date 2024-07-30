import subprocess


def start_xvfb():
    cmd = "Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &"
    subprocess.Popen(cmd, shell=True)

def path_to_gif(path):
    

