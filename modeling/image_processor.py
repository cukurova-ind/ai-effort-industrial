import os
import cv2
import time

class Preprocessor:
    def __init__(self, image_path, save_path):
        self.image = cv2.imread(image_path)
        self.image_name = ".".join(image_path.split("/")[-1].split(".")[:-1])
        self.save_path = save_path

    def convert_gray(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def add_blur(self, image, val=5):
        return cv2.medianBlur(image, val)
    
    def add_threshold(self, image, val=0):
        return cv2.threshold(image, val, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    def edge_detection(self, image, val1=15, val2=200):
        return cv2.Canny(image, val1, val2)
    
    def contour_detection(self, image, draw=True):
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_copy1, img_copy2 = self.image.copy(), self.image.copy()
        boxes = []
        for cnt in contours:    
            a = cv2.boundingRect(cnt)
            x, y, w, h = a
            if w > 100 and h > 300:
                boxes.append(a)
                #print(w, h)
                if draw:
                    cv2.rectangle(img_copy1, (x, y), (x + w, y + h), (0, 255, 0), 5)
        if draw:
            cv2.drawContours(img_copy2, contours, -1, (255, 0, 255), 5)
        return boxes, contours, hierarchy, img_copy1, img_copy2
    
    def clip_image(self, boxes):
        obj = boxes[0]
        area = 0
        for b in boxes:
            
            if b[2]*b[3] > area:
                area = b[2]*b[3]
                obj = b

        x, y, w, h = obj
        clip_image = self.image[y:y + h, x:x + w]
        save_file = os.path.join(self.save_path, self.image_name + ".jpg")
        cv2.imwrite(save_file, clip_image)
    
    def process(self):
        gray = self.convert_gray(self.image)
        blur = self.add_blur(gray)
        _, th = self.add_threshold(blur)
        edged = self.edge_detection(th)
        boxes, _, _, _, _ = self.contour_detection(edged, draw=False)
        self.clip_image(boxes)