import os
from cv2_process_image import process_photo,binary


DIR = '../all_photos/'
file ='IMG_5314.JPG'

file_path = os.path.join(DIR, file)
processed_image = process_photo(file_path, ratio_parameter=450, blur=5)