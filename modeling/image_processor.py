import os
import cv2
import numpy as np

class Preprocessor:
    def __init__(self, image_path, save_path):
        self.image = cv2.imread(image_path)
        self.image_name = ".".join(image_path.split("\\")[-1].split(".")[:-1])
        self.save_path = save_path
        self.hh, self.ww = self.image.shape[:2]
        self.lower = np.array([235, 235, 235])
        self.upper = np.array([255, 255, 255])

    def create_mask(self):
        return cv2.inRange(self.image, self.lower, self.upper)

    def apply_morphy(self, thresh):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
        return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    def add_threshold(self, image, val=0):
        _, thresh = cv2.threshold(image, val, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return cv2.bitwise_and(image, image, mask=thresh)
    
    def edge_detection(self, image, val1=15, val2=200):
        return cv2.Canny(image, val1, val2)
    
    def contour_detection(self, morph, draw=True):
        contours, hierarchy = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_copy1, img_copy2 = self.image.copy(), self.image.copy()
        boxes = []
        for cnt in contours:    
            a = cv2.boundingRect(cnt)
            x, y, w, h = a
            if w > 100 and h > 100:
                boxes.append(a)
                if draw:
                    cv2.rectangle(img_copy1, (x, y), (x + w, y + h), (0, 255, 0), 5)
        if draw:
            cv2.drawContours(img_copy2, contours, -1, (255, 0, 255), 5)
        return boxes, contours, hierarchy, img_copy1, img_copy2
    
    def clip_image(self, boxes):

        save_file = os.path.join(self.save_path, self.image_name + ".jpg")
        if len(boxes) == 0:
            return 0, save_file
        
        obj = boxes[0]
        area = 0
        for b in boxes:
            if b[2]*b[3] > area and self.ww > b[2]:
                area = b[2]*b[3]
                obj = b
        
        x, y, w, h = obj
        clip_image = self.image[y:y + h, x:x + w]
        save_file = os.path.join(self.save_path, self.image_name + ".jpg")
        cv2.imwrite(save_file, clip_image)

        return 1, save_file
    
    def process(self):
        th = self.create_mask()
        morph_image = self.apply_morphy(th)
        boxes, _, _, _, _ = self.contour_detection(morph_image)
        res, file = self.clip_image(boxes)

        return res, file