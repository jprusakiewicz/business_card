import os, os.path
from os.path import isfile, join
import fitz
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pdf2image import convert_from_path

#DIR = 'tests/test6/'
#file = 'test6.2.JPG'
DIR = '../regular_cards/'
file = 'IMG_5334.JPG'
file_path = os.path.join(DIR, file)
doc = fitz.open(file_path+'.pdf')
page = doc.loadPage(0)
textpage = page.getTextPage()
textb = textpage.extractText()
#regular expression to find emails
emails = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", textb, re.IGNORECASE)

#regular expression to find phone numbers
numbers = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', textb, re.IGNORECASE)

texts_to_draw = []
texts_to_draw.extend(emails)
texts_to_draw.extend(numbers)
print(texts_to_draw)

boxes = []
for text in texts_to_draw:
    boxes += page.searchFor(text)

print(boxes)

page_dict = page.getText('dict')
#page_blocks = page.getText('blocks') unused yet
page_width = page_dict['width']
page_height = page_dict['height']
###
image = convert_from_path(file_path+'.pdf', size=(page_width, page_height))

im = np.array(image[0], dtype=np.uint8)
fig, ax = plt.subplots(figsize=(30, 20))

# Display the image
for rectangle in boxes:
    rect = patches.Rectangle((rectangle.x0, rectangle.y0), (rectangle.x1 - rectangle.x0), (rectangle.y1 - rectangle.y0),
                             linewidth=1, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

ax.imshow(im)
plt.savefig('foo.png')
#plt.show()



