import os, os.path
from os.path import isfile, join
import fitz
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pdf2image import convert_from_path
import pandas as pd

#DIR = 'tests/test6/'
#file = 'test6.2.JPG'
DIR = '../regular_cards/'
file = 'IMG_5335.JPG'


def find_features(directory, file_name):
    file_path = os.path.join(DIR, file_name)
    doc = fitz.open(file_path + '.pdf')
    page = doc.loadPage(0)
    textpage = page.getTextPage()
    text = textpage.extractText()
    #regular expression to find emails
    emails = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", text, re.IGNORECASE)

    #regular expression to find phone numbers
    numbers = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text, re.IGNORECASE)

    names = re.findall(r'[A-Z][a-z]*[\s-][A-Z][a-z]*', text)
    #print(names)

    features_found = []
    features_found.extend(emails)
    features_found.extend(numbers)
    features_found.extend(names)

    #print(features_found)
    #print(len(features_found))

    bounding_boxes = []
    for text in features_found:
        bounding_boxes += page.searchFor(text)

    #print(bounding_boxes)
    #print(len(bounding_boxes))

    # Create a zipped list of tuples from above lists
    zippedList = list(zip([file_name], names, numbers, emails))
    df = pd.DataFrame(zippedList, columns=['Filename', 'Name', 'number', 'email'])
    #dic = {'name': names, 'number': numbers, 'email': emails}
    #df = pd.DataFrame(dic)
    csv = df.to_csv(index=False)

    def draw():
        page_dict = page.getText('dict')
        #page_blocks = page.getText('blocks') unused yet
        page_width = page_dict['width']
        page_height = page_dict['height']

        image = convert_from_path(file_path+'.pdf', size=(page_width, page_height))

        im = np.array(image[0], dtype=np.uint8)
        fig, ax = plt.subplots(figsize=(30, 20))

        # Display the image
        for rectangle in bounding_boxes:
            rect = patches.Rectangle((rectangle.x0, rectangle.y0), (rectangle.x1 - rectangle.x0), (rectangle.y1 - rectangle.y0),
                                     linewidth=1, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

        ax.imshow(im)
        plt.savefig('foo.JPG')
    return csv
    #draw()

a = find_features(directory=DIR, file_name=file)
print(a)
#plt.show()



