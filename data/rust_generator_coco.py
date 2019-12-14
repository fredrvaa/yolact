import cv2
import numpy as np
import random
import os
import json
from imantics import Polygons, Mask
from itertools import groupby


class Image(object):
    def __init__(self, id):
        self.filename = '{}.jpg'.format(id)
        self.id = id
        self.rust = None
        self.bg = None
        self.masks = list()
        self.mask = None
        self.image = None
        self.image_shape = None

    def load_image(self, rust_path, bg_path):
        self.rust = cv2.imread(rust_path)
        self.rust = cv2.resize(self.rust, (1300,866))
        self.image_shape = self.rust.shape

        self.bg = cv2.imread(bg_path)
        self.bg = cv2.resize(self.bg, (self.image_shape[1], self.image_shape[0]))
        
        self.mask = np.zeros(self.image_shape[:2], dtype=np.uint8)
    
    def get_random_loc(self, radius):
        locx = random.randrange(radius, self.image_shape[0] - radius)
        locy = random.randrange(radius, self.image_shape[1] - radius)
        return locx, locy

    def get_random_point(self, locx, locy, sigma_x, sigma_y, threshold):
        ptx = np.random.normal(locx, sigma_x)
        pty = np.random.normal(locy, sigma_y)
        if ptx <= 0: ptx = 0
        if ptx >= self.image_shape[1]: ptx = self.image_shape[1]
        if pty <= 0: pty = 0
        if pty >= self.image_shape[0]: pty = self.image_shape[0]
        return ptx, pty

    def get_point_list(self, sigma_x, sigma_y, radius, num_points, kernel, locx, locy):
        ptsx = list()
        ptsy = list()
        for j in range(num_points):
            ptx, pty = self.get_random_point(locx, locy, sigma_x, sigma_y, radius)
            ptsx.append(ptx)
            ptsy.append(pty)
        return ptsx, ptsy

    def draw_dots(self, ptsx, ptsy, radius, mask):
        for ptx, pty in zip(ptsx, ptsy):
            cv2.circle(mask, (int(ptx), int(pty)), radius, (255,255,255), -1)
        return mask

    def morph_dots(self, kernel, mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
        return mask
        

    def create_image(self, rust_path, bg_path, sigmas_x = (10, 20, 40, 60, 100), sigmas_y = (10, 20, 40, 60, 100), 
                     radius = 2, num_locs = 5, num_points = 50, 
                     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 30))):
        #Loading image
        self.load_image(rust_path, bg_path)
        #Randomly altering mask to show rust spots
        for i in range(num_locs):
            sigma_x = random.choice(sigmas_x)
            sigma_y = sigma_x
            locx, locy = self.get_random_loc(radius)
            ptsx1, ptsy1 = self.get_point_list(sigma_x, sigma_y, radius, num_points, kernel, locx, locy)

            #Creating main rust area
            mask = np.zeros(self.image_shape[:2], dtype=np.uint8)
            mask = self.draw_dots(ptsx1, ptsy1, radius, mask)
            #mask = self.morph_dots(kernel, mask)

            #Adding some noise
            ptsx2, ptsy2 = self.get_point_list(sigma_x, sigma_y, radius, num_points*2, kernel, locx, locy)
            mask = self.draw_dots(ptsx2, ptsy2, radius - 1, mask)

            ptsx3, ptsy3 = self.get_point_list(sigma_x, sigma_y, radius, num_points*10, kernel, locx, locy)
            mask = self.draw_dots(ptsx3, ptsy3, radius - 2, mask)

            mask[0][0] = 0
            
            self.masks.append(mask)

        #Adding images
        for mask in self.masks:
            self.mask = cv2.add(self.mask, mask)
        masked_rust = cv2.bitwise_and(self.rust, self.rust, mask=self.mask)
        inv_mask = cv2.bitwise_not(self.mask)
        masked_bg = cv2.bitwise_and(self.bg, self.bg, mask=inv_mask)

        self.image = cv2.add(masked_rust, masked_bg)

def get_counts(mask):
    binary_mask = mask
    binary_mask[binary_mask == 255] = 1

    counts = []
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='C'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return counts

def get_rle_area(mask):
    return int(np.sum(mask == 255))

def get_poly_area(x, y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def get_bbox(mask):
    a = np.where(mask != 0)
    y1,x1,y2,x2 = np.min(a[0]), np.min(a[1]), np.max(a[0]), np.max(a[1])
    bbox = [x1, y1, x2 - x1, y2 - y1]
    return list(map(int,bbox))

def image_data(filename, height, width, id):
    image = {
        'filename': filename,
        'height': height,
        'width': width,
        'id': id
    }
    return image

def rle_annotation_data(filename, counts, size, area, bbox, image_id):
    annotation = {
        'segmentation' : {
            'counts': counts,
            'size': size
        },
        'area': area,
        'bbox': bbox,
        'iscrowd': 0,
        'image_id': image_id,
        'category_id': 1,
        'id': image_id
    }
    return annotation

def poly_annotation_data(filename, segmentation, area, bbox, image_id):
    annotation = {
        'segmentation': segmentation,
        'area': area,
        'bbox': bbox,
        'iscrowd': 0,
        'image_id': image_id,
        'category_id': 1,
        'id': image_id
    }
    return annotation

def generate_rle_images(num_images, IMAGE_DIR, ANNOTATION_DIR, RUST_DIR, BG_DIR):

    data = {'images': [], 'annotations': [], 'categories': []}

    for x in range(num_images):
        rust = random.choice(os.listdir(RUST_DIR))
        bg = random.choice(os.listdir(BG_DIR))

        num_locs = random.choice((1,3,5))
        image = Image(x)
        image.create_image('{}/{}'.format(RUST_DIR, rust), '{}/{}'.format(BG_DIR, bg),
                        sigmas_x = (5, 40, 80), sigmas_y = (5, 40, 80), radius = 3, num_locs=num_locs, num_points = 500, 
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 30)))

        im_data = image_data(image.filename, image.image_shape[0], image.image_shape[1], image.id)
        data['images'].append(im_data)

        size = [image.image_shape[1], image.image_shape[0]]
        for mask in image.masks:
            counts = get_counts(mask)
            area = get_rle_area(mask)
            bbox = get_bbox(mask)

            an_data = rle_annotation_data(image.filename, counts, size, area, bbox, image.id)
            data['annotations'].append(an_data)

        cv2.imwrite('{}/{}'.format(IMAGE_DIR, image.filename), image.image)
        print('SAVED {} IN {}'.format(image.filename, IMAGE_DIR))

    with open('{}/{}'.format(ANNOTATION_DIR, 'annotations.json'), 'w') as f:
        json.dump(data, f)

def generate_polygon_images(num_images, IMAGE_DIR, ANNOTATION_DIR, RUST_DIR, BG_DIR):
    data = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'rust'},]}

    for x in range(num_images):
        rust = random.choice(os.listdir(RUST_DIR))
        bg = random.choice(os.listdir(BG_DIR))

        num_locs = random.choice((1,3,5))
        image = Image(x)
        image.create_image('{}/{}'.format(RUST_DIR, rust), '{}/{}'.format(BG_DIR, bg),
                        sigmas_x = (5, 40, 80), sigmas_y = (5, 40, 80), radius = 3, num_locs=num_locs, num_points = 500, 
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 30)))

        im_data = image_data(image.filename, image.image_shape[0], image.image_shape[1], image.id)
        data['images'].append(im_data)

        for mask in image.masks:
            segmentations = Mask(mask).polygons().segmentation

            filtered_segmentations = []
            for segmentation in segmentations:
                if len(segmentation) > 4: filtered_segmentations.append(segmentation)

            area = 0
            for segmentation in filtered_segmentations:
                x = segmentation[::2]
                y = segmentation[1::2]
                area = area + get_poly_area(x, y)

            bbox = get_bbox(mask)

            if len(filtered_segmentations):
                an_data = poly_annotation_data(image.filename, filtered_segmentations, area, bbox, image.id)
                data['annotations'].append(an_data)

        cv2.imwrite('{}/{}'.format(IMAGE_DIR, image.filename), image.image)
        print('SAVED {} IN {}'.format(image.filename, IMAGE_DIR))

    with open('{}/{}'.format(ANNOTATION_DIR, 'annotations.json'), 'w') as f:
        json.dump(data, f)

def test(RUST_DIR, BG_DIR):
    rust = random.choice(os.listdir(RUST_DIR))
    bg = random.choice(os.listdir(BG_DIR))

    num_locs = random.choice((1,3,5))
    image = Image(1)
    image.create_image('{}/{}'.format(RUST_DIR, rust), '{}/{}'.format(BG_DIR, bg),
                    sigmas_x = (5, 40, 80), sigmas_y = (5, 40, 80), radius = 3, num_locs=num_locs, num_points = 500, 
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 30)))

    cv2.imshow('image', image.image)
    cv2.waitKey(-1)
                    
if __name__ == "__main__":
    seg_mode = 'test' #seg_mode = 'polygon' / 'rle'     //'test'
    subset = 'val' #subset = 'train' / 'val'

    IMAGE_DIR = 'C:/Users/Fredrik/Desktop/yolact/data/generated_dataset/{}/{}/images'.format(seg_mode, subset)
    ANNOTATION_DIR = 'C:/Users/Fredrik/Desktop/yolact/data/generated_dataset/{}/{}/annotations'.format(seg_mode, subset)

    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    if not os.path.exists(ANNOTATION_DIR):
        os.makedirs(ANNOTATION_DIR)
        
    RUST_DIR = 'C:/Users/Fredrik/Desktop/Mask_RCNN/datasets/generated_dataset/rust_images'
    BG_DIR = 'C:/Users/Fredrik/Desktop/Mask_RCNN/datasets/generated_dataset/bg_images'

    num_images = 200

    if seg_mode == 'polygon':
        generate_polygon_images(num_images, IMAGE_DIR, ANNOTATION_DIR, RUST_DIR, BG_DIR)
    elif seg_mode == 'rle':
        generate_rle_images(num_images, IMAGE_DIR, ANNOTATION_DIR, RUST_DIR, BG_DIR)
    elif seg_mode == 'test':
        test(RUST_DIR, BG_DIR)
    
