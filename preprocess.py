import csv
import json
import sys
import pandas as pd
from pycocotools import mask
import pycococreatortools as pycoco
import numpy as np

#from PIL import Image
from sklearn.model_selection import train_test_split
from collections import OrderedDict

def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    rle = {
            'counts': [],
            'size': [shape[0], shape[1]]
        }
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for i, (lo, hi) in enumerate(zip(starts, ends)):
        img[lo:hi] = 1
        rle['counts'].append(starts[i] - ends[i-1])
        rle['counts'].append(ends[i] - starts[i])
        
    rle['counts'][0] = starts[0]
    rle['counts'].append(shape[0]*shape[1] - ends[-1])
    img_rs = img.reshape(shape, order='F')
    return (rle, img_rs)

def create_coco_style(input_path):
    maxInt = sys.maxsize

    while True:
        try:
            csv.field_size_limit(maxInt)
            break
        except:
            maxInt = int(maxInt/10)
            
    data_path = 'train.csv'
    json_des = 'label_descriptions.json'

    with open(json_des, 'r') as f:
        des = json.load(f)
        
    info = des['info']
    categories = des['categories']
    attributes = des['attributes']

    #f = open(data_path, 'r')
    #
    X = pd.read_csv(data_path)

    X_train, X_test = train_test_split(X,test_size=0.2)
    # X_train = X

    X_dtrain = X_train.to_dict('records', into=OrderedDict)
    X_dtest = X_test.to_dict('records', into=OrderedDict)

    #reader = csv.DictReader(f)#, fieldnames=('imageid', 'height', 'width', 'encodedpixels', 'classid'))

    rows_train = []
    rows_test = []

    myorder = ['ImageId', 'Height', 'Width', 'EncodedPixels', 'ClassId']

    image_id = 1
    segmentation_id = 1

    coco_output = {
            "info": info,
            "licenses": "",
            "categories": categories,
            "images": [],
            "annotations": []
            }

    coco_output_test = coco_output

    with open('{}train.txt'.format(input_path), 'w') as output_text_file:
        for row in X_dtrain:
            # Write training text
            output_text_file.write('{} '.format(row['ImageId']))

            # Ordered Dict
            ordered = OrderedDict((k, row[k]) for k in myorder)
        #    ordered['EncodedPixels'] = list(map(int, ordered['EncodedPixels'].split(' ')))
            if len(ordered['ClassId']) > 2:
                classes = [list(map(int, ordered['ClassId'].split('_')))]
                ordered['ClassId'] = classes[0][0]
            else:
                ordered['ClassId'] = int(ordered['ClassId'])
                
            # COCO
            image_info = pycoco.create_image_info(image_id, input_path + row['ImageId'], (row['Width'], row['Height']))
            coco_output["images"].append(image_info)
            
            rle, binary_mask = rle_decode(row['EncodedPixels'], (row['Height'], row['Width']))
            fortran_binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))

            binary_mask_encoded = mask.encode(fortran_binary_mask)
        #    rle2 = pycoco.binary_mask_to_rle(fortran_binary_mask)
            
            area = mask.area(binary_mask_encoded)
            bounding_box = mask.toBbox(binary_mask_encoded)
            
            annotation_info = {
                "id": segmentation_id,
                "image_id": image_id,
                "category_id": ordered['ClassId'],
                "iscrowd": 1,
                "area": area.tolist(),
                "bbox": bounding_box.tolist(),
                "segmentation": rle,
                "width": row['Width'],
                "height": row['Height'],
                }
            coco_output["annotations"].append(annotation_info)    
            segmentation_id += 1
            image_id += 1
            
            rows_train.append(ordered)

    with open('{}train.json'.format(input_path), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)

    with open('{}test.txt'.format(input_path), 'w') as output_text_file:
        for row in X_dtest:
            # Write test text
            output_text_file.write('{} '.format(row['ImageId']))
            
            # Ordered Dict
            ordered = OrderedDict((k, row[k]) for k in myorder)
        #    ordered['EncodedPixels'] = list(map(int, ordered['EncodedPixels'].split(' ')))
            if len(ordered['ClassId']) > 2:
                classes = [list(map(int, ordered['ClassId'].split('_')))]
                ordered['ClassId'] = classes[0][0]
            else:
                ordered['ClassId'] = int(ordered['ClassId'])
                
            # COCO
            image_info = pycoco.create_image_info(image_id, input_path + row['ImageId'], (row['Width'], row['Height']))
            coco_output_test["images"].append(image_info)
            
            rle, binary_mask = rle_decode(row['EncodedPixels'], (row['Height'], row['Width']))
            fortran_binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))

            binary_mask_encoded = mask.encode(fortran_binary_mask)
        #    rle2 = pycoco.binary_mask_to_rle(fortran_binary_mask)
            
            area = mask.area(binary_mask_encoded)
            bounding_box = mask.toBbox(binary_mask_encoded)
            
            annotation_info = {
                "id": segmentation_id,
                "image_id": image_id,
                "category_id": ordered['ClassId'],
                "iscrowd": 1,
                "area": area.tolist(),
                "bbox": bounding_box.tolist(),
                "segmentation": rle,
                "width": row['Width'],
                "height": row['Height'],
                }
            coco_output_test["annotations"].append(annotation_info)    
            segmentation_id += 1
            image_id += 1
            
            rows_train.append(ordered)

        with open('{}test.json'.format(input_path), 'w') as output_json_file:
            json.dump(coco_output_test, output_json_file)


#out_train = json.dumps(rows_train)
#out_test = json.dumps(rows_test)
##
#print('JSON parsed!')
#
#f2 = open('train2.json','w')
#f2.write(out_train)
#
#f2 = open('test2.json','w')
#f2.write(out_test)
#print('JSON saved')
