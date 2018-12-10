# video utils 
import numpy as np 
from PIL import Image 
import tensorflow as tf 

video_dir_name = "~/Downloads/project/"

def read_image(filename):
    return np.asarray(Image.open(filename))

def get_filepath(scene_number, video_number, frame_number):
    """Get filepath to frame corresponding to given scene, video, and frame.

    Example filepath: ""~/Google Drive/Google Photos/project/scene3/scene3_vid3_frames/scene3_vid3_007.jpg"

    """
    filepath = video_dir_name
    scene = "/scene" + str(scene_number)
    video = scene + "_vid" + str(video_number) + "_frames"
    frame = scene + "_vid" + str(video_number) + "_" + str(frame_number).zfill(3) + ".jpg"  # note: assumption 
    #  here is that all videos have fewer than 100 frames
    filepath += scene + video + frame 
    return filepath 



def read_image(scene_number, video_number, frame_number):
    """Read frame given the scene type, the video index within that scene, and frame number."""
    return read_image(get_filepath(scene_number, video_number, frame_number))


if __name__=="__main__":

    # test 
    print(get_filepath(3, 2, 12))

    
	
