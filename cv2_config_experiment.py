from cv2_process_image import find_best_ratio
import os
from os.path import isfile, join
import requests
import fitz
import pandas as pd


DIR = '../all_photos/'
# DIR = '../vertical_cards/'
#ratio_parameter = 200
#blur = 5
#canny_thresh_l = 50
#canny_thresh_h = 230
files_names = [f for f in os.listdir(DIR) if isfile(join(DIR, f))]
files_names.sort()

print(len(files_names))

def second_experiment(file_path, ratio_parameter,blur):
    try:
        contour_area = find_best_ratio(file_path, ratio_parameter=ratio_parameter)

        _results = {'file': file, ratio_parameter: str(contour_area)}
        return _results
    except Exception:
        # print(e)
        # print(file, 'failed |', e)
        # _results = {'file': file, 'len': 'failed', 'ratio_parameter': ratio_parameter, 'blur': blur,
        #             'canny_thresh_l': canny_thresh_l, 'canny_thresh_h': canny_thresh_h}
        _results = {'file': file, ratio_parameter: 'failed'}
        return _results

# def experiment(file_path, ratio_parameter, blur):
#     try:
#         processed_image = process_photo(file_path, ratio_parameter=ratio_parameter)
#         req = requests.post('http://localhost:88/ocr', files=processed_image)
#         with open('temporary_experiment_file' + '.pdf', 'wb') as fw:
#             fw.write(req.content)
#         doc = fitz.open('temporary_experiment_file.pdf')
#         page = doc.loadPage(0)
#         textpage = page.getTextPage()
#         text = textpage.extractText()
#         # print(file, str(len(text)))
#         # _results = {'file': file, 'len': str(len(text)), 'ratio_parameter': ratio_parameter, 'blur': blur,
#         #         #             'canny_thresh_l': canny_thresh_l, 'canny_thresh_h': canny_thresh_h}
#         _results = {'file': file, ratio_parameter: str(len(text))}
#         return _results
#     except Exception:
#         # print(e)
#         # print(file, 'failed |', e)
#         # _results = {'file': file, 'len': 'failed', 'ratio_parameter': ratio_parameter, 'blur': blur,
#         #             'canny_thresh_l': canny_thresh_l, 'canny_thresh_h': canny_thresh_h}
#         _results = {'file': file, ratio_parameter: 'failed'}
#         return _results

d = []
for file in files_names[:]:
    file_path = os.path.join(DIR, file)
    ratio_parameter = 50
    results = second_experiment(file_path=file_path, ratio_parameter=ratio_parameter, blur=5)
    ratio_parameter += 50
    while ratio_parameter < 550:
        b = second_experiment(file_path=file_path, ratio_parameter=ratio_parameter, blur=5)
        results[ratio_parameter] = b[ratio_parameter]
        ratio_parameter += 50
    print(results)
    d.append(results)


df = pd.DataFrame(d)
print(df)
df.to_csv('experiment_out.csv', index=False)
    # http://localhost:5000/ocr
