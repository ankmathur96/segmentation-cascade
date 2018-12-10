# video utils 
import numpy as np 
from PIL import Image 
import tensorflow as tf 
import os

video_dir_name = "~/Downloads/project"

scenes = [1, 3, 4, 5]
scene_to_label = {1:0, 3:1, 4:2, 5:3}

def read_image(filename):
    return np.asarray(Image.open(filename))

def get_filepath(scene_number, video_number, frame_number):
    """Get filepath to frame corresponding to given scene, video, and frame.

    Example filepath: ""~/Downloads/project/scene3/scene3_vid3_frames/scene3_vid3_007.jpg"

    """
    filepath = video_dir_name
    scene = "/scene" + str(scene_number)
    video = scene + "_vid" + str(video_number) + "_frames"
    frame = scene + "_vid" + str(video_number) + "_" + str(frame_number).zfill(3) + ".jpg"  # note: assumption 
    #  here is that all videos have fewer than 100 frames
    filepath += scene + video + frame 
    return filepath 

def read_image_from_ids(scene_number, video_number, frame_number):
    """Read frame given the scene type, the video index within that scene, and frame number."""
    return read_image(get_filepath(scene_number, video_number, frame_number))

def get_all_images_for_scene(scene_number, root=video_dir_name, test=False):
    numpy_frames = []
    scene_dir = root + "/scene" + str(scene_number)
    for video in os.listdir(scene_dir):
        video_path = os.path.join(scene_dir, video)
        for frame in os.listdir(video_path):
            frame_path = os.path.join(video_path, frame)
            if test: 
                print(frame_path)
            else:
                numpy_frames.append(read_image(frame_path))
    return numpy_frames 

if __name__=="__main__":

    # test 
    get_all_images_for_scene(3, test=True)

    
	
