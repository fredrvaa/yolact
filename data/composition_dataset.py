import numpy as np
import cv2
import os
import json
from imantics import Mask, Polygons

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def image_comp(background, image, startx, starty, h, w):
    comp = background.copy()

    roi = comp[starty:starty+h, startx:startx+w]

    result = np.zeros((h,w,3), np.uint8)
    alpha = image[:, :, 3] / 255.0
    result[:, :, 0] = (1. - alpha) * roi[:, :, 0] + alpha * image[:, :, 0]
    result[:, :, 1] = (1. - alpha) * roi[:, :, 1] + alpha * image[:, :, 1]
    result[:, :, 2] = (1. - alpha) * roi[:, :, 2] + alpha * image[:, :, 2]

    comp[starty:starty+h, startx:startx+w] = result

    return comp

def get_bbox(mask):
    a = np.where(mask != 0)
    y1,x1,y2,x2 = np.min(a[0]), np.min(a[1]), np.max(a[0]), np.max(a[1])
    bbox = [x1, y1, x2 - x1, y2 - y1]
    return list(map(int,bbox))

def get_poly_area(x, y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def image_data(filename, height, width, id):
    image = {
        'filename': filename,
        'height': height,
        'width': width,
        'id': id
    }
    return image

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

class Image(object):
    def __init__(self, BG_DIR, RUST_DIR):
        self.BG_DIR = BG_DIR
        self.RUST_DIR = RUST_DIR
        self.background = None
        self.masks = []

    def load_bg(self):
        self.background = cv2.imread('{}/{}'.format(self.BG_DIR, np.random.choice(os.listdir(self.BG_DIR))), -1)

        #Resizing
        self.background = cv2.resize(self.background, (1310,866))

        #Flipping
        flip = np.random.choice((-1, 0, 1, 2))
        if flip != 2:
            self.background = cv2.flip(self.background, flip)

    def create_image(self, num_spots):
        #Loading random background
        self.load_bg()

        n = 0
        #Adding rust spots to background image
        while n < num_spots:
            #Reading random rust image
            rust = cv2.imread('{}/{}'.format(self.RUST_DIR,np.random.choice(os.listdir(self.RUST_DIR))), -1)

            #Random scale and rotation
            scale = np.random.choice((1, 0.8, 0.5))
            angle = np.random.randint(0,350)
            rust = cv2.resize(rust, (0,0), fx=scale, fy=scale)
            rust = rotate_image(rust, angle)

            if rust.shape[0]<self.background.shape[0] and rust.shape[1]<self.background.shape[1]:
                #Rust shape
                h, w, c = rust.shape

                #Getting composite image of background and rust
                startx, starty = np.random.randint(0, self.background.shape[1]-w), np.random.randint(0, self.background.shape[0]-h)
                self.background = image_comp(self.background, rust, startx, starty, h, w)

                #Finding largest rust contour
                alpha = rust[:, :, 3]
                contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                #Creating binary mask for largest rust contour
                rust_mask = np.zeros(alpha.shape)
                if len(contours) != 0:
                    c = max(contours, key=cv2.contourArea)
                    rust_mask = cv2.drawContours(rust_mask, [c], -1, 255, -1)

                #Creating the full mask for image
                comp_mask = np.zeros(self.background.shape[:2])
                comp_mask[starty:starty+h, startx:startx+w] = rust_mask

                #Adding new comp mask to mask
                self.masks.append(comp_mask)

                n = n + 1
            else:
                print('Rust shape bigger than background')

def generate_poly_images(num_images, BG_DIR, RUST_DIR, IMAGE_DIR, ANNOTATION_DIR):
    #Storing data in coco format
    data = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'rust'},],}

    for n in range(num_images):
        filename = '{}.jpg'.format(n)
        image_id = n
        #Initialzing image object
        image = Image(BG_DIR, RUST_DIR)
        
        #Creating images with random number of rust spots
        num_spots = np.random.choice((1,2,3))
        image.create_image(num_spots)

        for mask in image.masks:
            segmentations = Mask(mask).polygons().segmentation

            area = 0
            for segmentation in segmentations:
                x = segmentation[::2]
                y = segmentation[1::2]
                area = area + get_poly_area(x, y)

            bbox = get_bbox(mask)

            im_data = image_data(filename, image.background.shape[0], image.background.shape[1], image_id)
            an_data = poly_annotation_data(filename, segmentations, area, bbox, image_id)

            data['images'].append(im_data)
            data['annotations'].append(an_data)

        cv2.imwrite('{}/{}'.format(IMAGE_DIR, filename), image.background)
        print('{} saved in {}'.format(filename, IMAGE_DIR))

        if n % 1000 == 0:
            print('---Saving Annotation Data---')
            with open('{}/{}'.format(ANNOTATION_DIR, 'annotations.json'), 'w') as f:
                json.dump(data, f)
    with open('{}/{}'.format(ANNOTATION_DIR, 'annotations.json'), 'w') as f:
                json.dump(data, f)

if __name__ == '__main__':
    SUBSET = 'val' #change to 'train'/'val'  
    BG_DIR ='bg_images'
    RUST_DIR = 'rust_images'
    IMAGE_DIR = 'composition_dataset_2/{}/images'.format(SUBSET)
    ANNOTATION_DIR = 'composition_dataset_2/{}/annotations'.format(SUBSET)  

    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    if not os.path.exists(ANNOTATION_DIR):
        os.makedirs(ANNOTATION_DIR)

    generate_poly_images(1000, BG_DIR, RUST_DIR, IMAGE_DIR, ANNOTATION_DIR)


