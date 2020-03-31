import cv2_process_image
from cv2_process_image import find_best_ratio
from os.path import isfile, join
import pandas as pd
import os
from PIL import Image
import io
import time

times_list = []


def main(file_path):
    d = get_all_parameters_for_single_file(file_path)
    best_ratio = get_best_parameter_for_single_file(d)
    scanned = cv2_process_image.show_segment(file_path, ratio_parameter=best_ratio)
    img = Image.fromarray(scanned, "L")
    output = io.BytesIO()
    img.save(output, format='JPEG')
    return output.getvalue()



def get_best_parameter_for_single_file(d):
    df = pd.DataFrame(d)
    try:
        for row in df.values:
            score = [float(a) if a != 'None' else 0 for a in row[1:]]
    except Exception as e:
        print(e, 'data frame may be broken')
    m = pd.Series(score)
    idx = m.idxmax()
    best_ratio = (idx + 1) * 50
    return best_ratio


def single_parameter_experiment(file, ratio_parameter):
    try:
        contour_area = find_best_ratio(file, ratio_parameter=ratio_parameter)
        _results = {'file': 'file', ratio_parameter: str(contour_area)}
        return _results
    except Exception:
        _results = {'file': 'file', ratio_parameter: 'failed'}
        return _results


def get_all_parameters_for_single_file(file):
    d = []
    ratio_parameter = 50
    results = single_parameter_experiment(file=file, ratio_parameter=ratio_parameter)
    ratio_parameter += 50
    while ratio_parameter < 550:
        b = single_parameter_experiment(file=file, ratio_parameter=ratio_parameter)
        results[ratio_parameter] = b[ratio_parameter]
        ratio_parameter += 50
    d.append(results)
    return d

DIR = '../all_photos/'

files_names = [f for f in os.listdir(DIR) if isfile(join(DIR, f))]
files_names.sort()

for file in files_names[:1]:
 file = 'IMG_5435.jpg'
 # file = 'IMG_5313.jpg' #testing on single file
 file_path = os.path.join(DIR, file)
 with open(file_path, "rb") as image:
     f = image.read()
     b = bytearray(f)
     #print(b)

 main(b)

# time_df = pd.DataFrame({'file': files_names, "time": times_list})
#
# timing_save_path = os.path.join('../output/', "times_patent.csv")
#
# time_df.to_csv(timing_save_path, index=False)

