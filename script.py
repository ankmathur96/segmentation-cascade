from PIL import Image
import numpy as np
from video_utils import get_all_images_for_scene
scenes = [1,3,4,5]
for scene_idx in scenes:
	scene_vids = get_all_images_for_scene(scene_idx, root='/Users/ankitmathur/datasets/project/')
	for i, video in enumerate(scene_vids):
		print(i, len(scene_vids))
		img = Image.open(video)
		new_img = img.resize((200,200), Image.ANTIALIAS)
		np_array = np.array(new_img)
		np.save(np_array)
		new_img.save(video.split('.')[0] + '_small.jpg')