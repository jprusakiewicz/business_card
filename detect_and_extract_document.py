import os
from cv2_process_image import find_best_ratio


DIR = '../all_photos/'
file ='IMG_5316.JPG'

file_path = os.path.join(DIR, file)
processed_image = find_best_ratio(file_path, ratio_parameter=450)