import subprocess, re
import cv2

class ImageRotater(object):
    def __init__(self, videoPath):
        self.videoPath = videoPath
        self.toBeRotated = False

    def detectRotation(self):
        cmd = 'ffmpeg -i %s' % self.videoPath
        p = subprocess.Popen(
            cmd.split(" "),
            stderr=subprocess.PIPE,
            close_fds=True
        )
        stdout, stderr = p.communicate()

        reo_rotation = re.compile('rotate\s+:\s(?P<rotation>.*)')
        stderr = stderr.decode('ISO-8859-1')
        match_rotation = reo_rotation.search(stderr)
        try:
            rotation = match_rotation.groups()[0]
            if rotation == '90':
                self.toBeRotated = True
            return rotation
        except:
            print("No Rotation Found")
            return "0"

    def rotateFrame(self, frame):
        if self.toBeRotated:
            frame =  cv2.rotate(frame, rotateCode=cv2.ROTATE_90_CLOCKWISE)
        return frame
