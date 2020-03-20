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
DIR = '../regular_cards/pdfs'
file = 'IMG_5335.JPG'

files_names_list = [f for f in os.listdir(DIR) if isfile(join(DIR, f))]
files_names_list.sort()
print(len(files_names_list))
def find_features(directory, file_name):
    file_path = os.path.join(DIR, file_name)
    #doc = fitz.open(file_path + '.pdf')
    doc = fitz.open(file_path)

    page = doc.loadPage(0)
    textpage = page.getTextPage()
    text = textpage.extractText()
    #regular expression to find emails
    emails = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", text, re.IGNORECASE)
    numbers = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text, re.IGNORECASE)
    names = re.findall(r'[A-Z][a-z]{3,}[\s-][A-Z][a-z]{3,}', text)
    www = re.findall(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})', text)
    #print(www)
    #print(len(names))
    if len(names) > 1:
        longest = ' '
        for features in names[1:]:
            if len(features) > len(longest):
                #print(features)
                longest = features
        #print(longest)
        new_names = names[0] + " " + longest
        #print(type(new_names))
        new_names = new_names.replace('\n', " ")
        names = [new_names]
        #print(new_names)
    elif len(names) == 1:
        names[0] = names[0].replace('\n', " ")
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

    if not names:
        names = ['']
    if not numbers:
        numbers = ['']
    if not emails:
        emails = ['']
    if not www:
        www = ['']
    # Create a zipped list of tuples from above lists
    zippedList = list(zip([file_name], names, numbers, emails, www))
    #list(zip([file_name][0], names[0], numbers[0], emails, www))
    print(zippedList)
    df = pd.DataFrame(zippedList, columns=['filename', 'name', 'number', 'email', 'www'])
    #dic = {'name': names, 'number': numbers, 'email': emails}
    #df = pd.DataFrame(dic)


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
    #return csv
    return df
    #draw()


def find_inpdf(doc):

    page = doc.loadPage(0)
    textpage = page.getTextPage()
    text = textpage.extractText()

    #regular expression to find emails
    emails = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", text, re.IGNORECASE)
    numbers = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text, re.IGNORECASE)
    names = re.findall(r'[A-Z][a-z]{3,}[\s-][A-Z][a-z]{3,}', text)
    www = re.findall(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})', text)
    #print(www)
    #print(len(names))
    if len(names) > 1:
        longest = ' '
        for features in names[1:]:
            if len(features) > len(longest):
                #print(features)
                longest = features
        #print(longest)
        new_names = names[0] + " " + longest
        #print(type(new_names))
        new_names = new_names.replace('\n', " ")
        names = [new_names]
        #print(new_names)
    elif len(names) == 1:
        names[0] = names[0].replace('\n', " ")
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

    if not names:
        names = ['']
    if not numbers:
        numbers = ['']
    if not emails:
        emails = ['']
    if not www:
        www = ['']
    # Create a zipped list of tuples from above lists
    zippedList = list(zip([file_name], names, numbers, emails, www))
    #list(zip([file_name][0], names[0], numbers[0], emails, www))
    print(zippedList)
    df = pd.DataFrame(zippedList, columns=['filename', 'name', 'number', 'email', 'www'])
    #dic = {'name': names, 'number': numbers, 'email': emails}
    #df = pd.DataFrame(dic)


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
    #return csv
    return df


dfs = find_features(directory=DIR, file_name=files_names_list[0])
for file in files_names_list[1:]:
    final =find_features(DIR, file)
    #print(final)
    dfs = dfs.append(final)
#print(dfs)
dfs.to_csv('out.csv', index=False)
#plt.show()




