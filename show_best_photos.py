import cv2_process_image

DIR = '../all_photos/'
#from cv2_process_image import process_photo
import os
import json
with open('experiment_best_performance3.json') as json_file:
    data = json.load(json_file)

print(data)
for photo in data.items():
    file_path = os.path.join(DIR, photo[0])
    print(file_path)
    cv2_process_image.show_segment(file_path, ratio_parameter=photo[1])
    # print(photo[1])
#results = experiment(file_path = os.path.join(DIR, file))