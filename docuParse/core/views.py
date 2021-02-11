from django.shortcuts import render
import os, io
from google.cloud import vision_v1 as vision
from google.cloud.vision_v1 import types
from google.cloud import translate_v2 as translate
import pandas as pd
import cv2
import numpy as np
import json
import re
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from enum import Enum
from termcolor import colored
from itertools import groupby
import statistics
import csv
import uuid
from django.core.files.storage import FileSystemStorage
from django.conf import settings

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'Demz_vision_API_token.json' # google API credentials

def parser(image):

    # Google API client
    client = vision.ImageAnnotatorClient()

    # Open and Read contents on the documents
    with io.open(os.path.join(image), 'rb') as image_file1:
            content = image_file1.read()
    content_image = types.Image(content=content) # Reading the Image Content
    rc1_response = client.text_detection(image=content_image) # Text Detection 
    rc1_texts = rc1_response.text_annotations # Text Response

    # face detection
    response_face = client.face_detection(image=content_image)
    faceAnnotations = response_face.face_annotations


    def arrange_the_texts(rc1_response):

        # rearrange the Response using the bbox values
        items = []
        lines = {}

        for text in rc1_response.text_annotations[1:]:
            left_x_axis = text.bounding_poly.vertices[0].x # top left
            left_y_axis = text.bounding_poly.vertices[0].y # top left
            top_x_axis = text.bounding_poly.vertices[1].x # top
            top_y_axis = text.bounding_poly.vertices[1].y # top
            right_x_axis = text.bounding_poly.vertices[2].x # right
            right_y_axis = text.bounding_poly.vertices[2].y # right
            bottom_x_axis = text.bounding_poly.vertices[3].x # bottom
            bottom_y_axis = text.bounding_poly.vertices[3].y # bottom

            if left_y_axis not in lines:
                lines[left_y_axis] = [(left_y_axis, bottom_y_axis), []]
            for s_top_y_axis, s_item in lines.items():
                if left_y_axis < s_item[0][1]:
                    lines[s_top_y_axis][1].append(([left_x_axis, bottom_y_axis, top_x_axis], text.description))
                    break
        for _, item in lines.items():
            if item[1]:
                words = sorted(item[1], key=lambda t: t[0])
                items.append((item[0], ' '.join([word for _, word in words]), words))

        # Find the most common space
        def find_common_space(items):
            spaces = []
            for i,k in enumerate(items):
                try:
                    calculated_space = items[i][2][0][0][1] - items[i -1][2][0][0][1]
                    spaces.append(calculated_space)
                except:
                    ...

            common_space = abs(statistics.median(spaces))

            return common_space

        # print(find_common_space(items))

        # Split the content based on horizontal space
        new_content = []

        most_common_space = find_common_space(items)
        # print(most_common_space)

        len_content = len(items) / 5.5

        for i,k in enumerate(items):
            try:
                if abs(items[i][2][0][0][1] - items[i -1][2][0][0][1]) >= most_common_space -len_content:
                    new_content.append([])
                    new_content.append(items[i][2])
                else:
                    new_content.append(items[i][2])
            except:
                new_content.append(items[i][2])


        # group the contents based on horizontal space
        new_content = [list(l) for i, l in groupby(new_content, bool) if i]

        # join list that the length is greater than one
        for i in range(len(new_content)):
            if len(new_content[i]) > 1:
                new_content[i] = sum(new_content[i], [])
                new_content[i] = [new_content[i]]
            else:
                new_content[i] = new_content[i]

        # find common width of white space on the document
        def find_common_diff(content):
            diffs = []
            for i,j in enumerate(new_content):
                for u,t in enumerate(new_content[i][0]):
                    try:
                        difference = abs(new_content[i][0][u][0][2] - new_content[i][0][u-1][0][0])
                        diffs.append(difference)
                    except:
                        ...
            common_diffs = abs(statistics.median(diffs))
            return common_diffs

        # print(find_common_diff(new_content))

        # find common width of white space on a line
        def find_common_diff_in_line(content):
            diffs = []
            for i,j in enumerate(new_content):
                temp_diffs = []
                for u,t in enumerate(new_content[i][0]):
                    try:
                        difference = abs(new_content[i][0][u][0][0] - new_content[i][0][u-1][0][2])
                        temp_diffs.append(difference)
                    except:
                        ...
                diffs.append(temp_diffs)
            return diffs

        # print(find_common_diff_in_line(new_content))

        def check_for_max_width(new_content):
            '''Check for max with of a text in a line'''
            widths = []
            for i,j in enumerate(new_content):
                temp_diffs = []
                for u,t in enumerate(new_content[i][0]):
                    try:
                        text_width = abs(new_content[i][0][u][0][0] - new_content[i][0][u][0][2])
                        temp_diffs.append(text_width)
                    except:
                        ...
                widths.append(temp_diffs)
            max_width = max(widths)
            return widths
        # print(check_for_max_width(new_content))


        # split line content based on vertical spacing
        contents = []
        max_text_width = 600
        most_common_diffs = find_common_diff(new_content)

        for i,j in enumerate(new_content):
            temp_content = []
            most_common_diffs_in_a_line = abs(statistics.median(find_common_diff_in_line(new_content)[i]))
            text_width = check_for_max_width(new_content)[i]
            max_width = max(text_width)
        #     print('')
            for u,t in enumerate(new_content[i][0]):
                try:
                    difference = abs(new_content[i][0][u][0][0] - new_content[i][0][u-1][0][2])
                    if most_common_diffs >= most_common_diffs_in_a_line and max_width < max_text_width:
                        if difference >= most_common_diffs_in_a_line+20: # getting the min width of white space in that line
                            temp_content.append([])
                            temp_content.append(new_content[i][0][u])

                        else:
                            temp_content.append(new_content[i][0][u]) 
                    elif most_common_diffs >= most_common_diffs_in_a_line and max_width > max_text_width:
                        if difference > most_common_diffs_in_a_line-15: # getting the min width of white space in that line
                            temp_content.append([])
                            temp_content.append(new_content[i][0][u])

                        else:
                            temp_content.append(new_content[i][0][u])
                    else:
                        if difference > most_common_diffs-65: # getting the most common width of white space in the doc
                            temp_content.append([])
                            temp_content.append(new_content[i][0][u])

                        else:
                            temp_content.append(new_content[i][0][u])
                except:
                    ...
        #     print(temp_content)
            new_temp_content = [list(l) for i, l in groupby(temp_content, bool) if i]
            contents.append(new_temp_content)    
        #     contents.append(temp_content)
        return contents

    def extract_keys_and_values(contents):
        """ This functionality extracts they keys and values as in the document structure"""
        def check_for_non_english(new_content):
            """this function Counts the number of non-english and english texts in a line"""
            max_pos = 800 # this is the max starting position that an english word does exceed
            translate_client = translate.Client()
            target = 'en'
            detected_words = []
            for i,j in enumerate(new_content):
                temp_words = []
                for u,t in enumerate(new_content[i]):
                    counts = {'en':0, 'af':0}
                    left_bbox = contents[i][u][0][0][0]
                    for s,r in enumerate(new_content[i][u]):
                        try:
                            text = new_content[i][u][s][1]
                            output = translate_client.translate(text, target_language=target)
                            if output['detectedSourceLanguage'] == 'en' or left_bbox < max_pos:
                                counts['en'] += 1
                            else:
                                counts['af'] += 1
                        except:
                            ...
                    temp_words.append(counts)
                detected_words.append(temp_words)
            return detected_words

        word_detection = check_for_non_english(contents)

        new_contents = []
        max_position = 1200
        list_of_special_keywords = ['RC1', 'RNC', 'RTS', 'NRW'] # special key words to add to the extracted values

        for i,j in enumerate(contents): # 
            temp_content = []
            word_detected = word_detection[i]
            content = contents[i]
            for u,t in enumerate(contents[i]):
                left_bbox = contents[i][u][0][0][0]
                if word_detected[u]['en'] > word_detected[u]['af'] and left_bbox < max_position or contents[i][u][0][1] in list_of_special_keywords:
                    temp_content.append(content[u])
            new_contents.append(temp_content)

        # get the left bbox of the keys
        left_bbox_lists = []
        for i in range(len(new_contents)):
            try:
                left_bbox = new_contents[i][0][0][0][0]
                left_bbox_lists.append(left_bbox)
            except:
                left_bbox = 0
                left_bbox_lists.append(left_bbox)
        # print(left_bbox_lists)
        max_left_bbox = max(left_bbox_lists)


        # getting the keys and values 
        # getting the keys and values 
        rc1_contents = []
        max_starting_pos = 250

        for i in range(len(new_contents)):
            keys = ''
            values = ''
            for u in range(len(new_contents[i])):
                left_bbox = new_contents[i][u][0][0][0]
                if left_bbox <= max_left_bbox and left_bbox <=max_starting_pos:
                    try:
                        for j in range(len(new_contents[i][u])):
                            text = new_contents[i][u][j][1]
                            keys += ' '+text
                    except:
                        ...
                else:
                    try:
                        for j in range(len(new_contents[i][u])):
                            text = new_contents[i][u][j][1]
                            values += ' '+text
                    except:
                        ...
            # if the content of the current line does not have a key but has a value. 
            # if so join the value with the value of the previous item in the rc1_contents.
            # Also check if the current key started with uppercase and the next key started wih lowercase
            # if so merge the current key with the next key
            last_key =  ''
            last_value = ''
            try:
                last_key = [i for i,j in rc1_contents[-1].items()][0] # previous item key
                last_value = [j for i,j in rc1_contents[-1].items()][0] # previous item value
            except:
                ...

            if keys == '' and values != '' and last_key and last_value:   # check for condition
                k = '' 
                val = ''
                for key, value in rc1_contents[-1].items():
                    if key and value:
                        val = value 
                        val += ' '+values
                        k = key
        #                 print(key, value)
                    else:
                        k = key
                        val = value
                rc1_contents.remove(rc1_contents[-1])
                rc1_contents.append({k:val})
            elif last_key != '' and keys != '':
                if last_key.strip()[0].isupper() and (keys.strip()[0].islower() and not values):
                    k = '' 
                    val = ''
                    for key, value in rc1_contents[-1].items():
                        if key and value:
                            val = value 
                            k = key
                            k += ' '+keys
            #                 print(key, value)
                        else:
                            k = key
                            val = value
                    rc1_contents.remove(rc1_contents[-1])
                    rc1_contents.append({k:val})
                else:
                    rc1_contents.append({keys:values})

            else:
                rc1_contents.append({keys:values})
        return rc1_contents

    contents = arrange_the_texts(rc1_response)
    contents = extract_keys_and_values(contents)

    keys_to_pull_out_RNC = ['Registering authority', 'Traffic register number', 'Name', 'Postal address', 'Street address', 'Address where notices must be served', 'Control number', 'Issue number', 'Date of issue']
    keys_not_to_pull_out_RC1 = [ "4024", "at Registering which registered authority", "RET(7)(2005/02)", "Republic of South Africa" ]
    keys_not_to_pull_out_NRW = ["NRW(2)(2003/10)", "NOTICE", "PARTICULARS (National Road Traffic OF Act, VEHICLE"]

    keys_to_pull_out_RNC = [i.upper().strip() for i in keys_to_pull_out_RNC]
    keys_not_to_pull_out_RC1 = [i.upper().strip() for i in keys_not_to_pull_out_RC1]
    keys_not_to_pull_out_NRW = [i.upper().strip() for i in keys_not_to_pull_out_NRW]

    extracted_values = []

    # check if the special keys are in the document, if so pull the right keys
    for e,f in contents[0].items():
        if 'RNC' in f:
            for i in range(len(contents)):
                for key, value in contents[i].items():
                    key = ' '.join(key.split())
                    if key.upper() in keys_to_pull_out_RNC:
                        extracted_values.append(contents[i])
        elif 'RC1' in f:
            for i in range(len(contents)):
                for key, value in contents[i].items():
                    key = ' '.join(key.split())
                    if key.upper() in keys_not_to_pull_out_RC1:
                        ...
                    else:
                        if key != '':
                            extracted_values.append(contents[i])
        elif 'NRW' in f:
            for i in range(len(contents)):
                for key, value in contents[i].items():
                    key = ' '.join(key.split())
                    if key.upper() in keys_not_to_pull_out_NRW:
                        ...
                    else:
                        if key != '':
                            extracted_values.append(contents[i])

    
    return extracted_values

def home(request):
	# print("MY_MEDIA_ROOT:  ",settings.MEDIA_ROOT)
	if request.method == 'POST' and request.FILES['filename']:
		doc_image = request.FILES['filename']
		fs = FileSystemStorage()
		DOCUMENT_IMAGE = fs.save(doc_image.name, doc_image)
		DOCUMENT_IMAGE = fs.path(DOCUMENT_IMAGE)

		# Call the parser
		doc_parser = parser(DOCUMENT_IMAGE)

		filename = doc_image.name
		filename = filename.split('.')[0]

		unique_id = uuid.uuid4().hex
		selected_filename = f'{filename}_{unique_id}.txt'
		
		# with open(selected_filename, 'w') as csv_file:  
		#     writer = csv.writer(csv_file)
		#     for item in doc_parser:
		# 	    for key, value in item.items():
		# 	       writer.writerow([key, value])
		
		media_path = os.path.join(settings.MEDIA_URL,selected_filename)
		# print('media_path:  ',media_path)

		with io.open(os.path.join(settings.MEDIA_ROOT,selected_filename), 'a', encoding="utf-8") as file:
		    for item in doc_parser:
		        for key, value in item.items():
		            line = f'{key}: {value}\n'
		            file.write(line)

		csv_path = fs.path(selected_filename)
		# print('csv_path:  ', csv_path)

		context = {
			'doc_parser': doc_parser,
			'media_path': media_path,
			'selected_filename':selected_filename,
		}
		return render(request, 'processed.html', context)
	return render(request, 'index.html')

def processed(request):
	return render(request, 'processed.html')