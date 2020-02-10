import pickle
from django.conf import settings
import cv2
import numpy as np
import os
from PIL import Image, ImageFont, ImageDraw


class Test:
    def __init__(self, classifier):
        if not os.path.isdir(settings.MEDIA_ROOT+'/pre-processed'):
            os.mkdir(settings.MEDIA_ROOT+'/pre-processed')
        if not os.path.isdir(settings.MEDIA_ROOT+'/segmented'):
            os.mkdir(settings.MEDIA_ROOT+'/segmented')
        if not os.path.isdir(settings.MEDIA_ROOT+'/classified'):
            os.mkdir(settings.MEDIA_ROOT+'/classified')
        self.class_list = ['Normal', 'Stage_1', 'Stage_2', 'Stage_3']
        self.classifiers_path = settings.BASE_DIR+'/diabetic_retinopathy/classifiers/'+classifier+'.pkl'
        self.classifiers_name = classifier

    @staticmethod
    def segment_image(i_n):
        fundus = cv2.imread(i_n)
        b, green_fundus, r = cv2.split(fundus)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced_green_fundus = clahe.apply(green_fundus)
        r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        r_1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        r2 = cv2.morphologyEx(r_1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
        r_2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                               iterations=1)
        r3 = cv2.morphologyEx(r_2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)), iterations=1)
        r_3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)),
                               iterations=1)
        f4 = cv2.subtract(r_3, contrast_enhanced_green_fundus)
        f5 = clahe.apply(f4)
        seg1 = r_2.copy()
        ret, f6 = cv2.threshold(f5, 15, 255, cv2.THRESH_BINARY)
        mask = np.ones(f5.shape[:2], dtype="uint8") * 255
        contours, _ = cv2.findContours(f6.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) <= 200:
                cv2.drawContours(mask, [cnt], -1, 0, -1)
        im = cv2.bitwise_and(f5, f5, mask=mask)
        ret, fin = cv2.threshold(im, 15, 255, cv2.THRESH_BINARY_INV)
        new_fin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        fundus_eroded = cv2.bitwise_not(new_fin)
        xmask = np.ones(fundus.shape[:2], dtype="uint8") * 255
        xcontours, _ = cv2.findContours(fundus_eroded.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in xcontours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)
            if len(approx) > 4 and 3000 >= cv2.contourArea(cnt) >= 100:
                shape = "circle"
            else:
                shape = "veins"
            if shape == "circle":
                cv2.drawContours(xmask, [cnt], -1, 0, -1)
        segmented_img = cv2.bitwise_and(fundus_eroded, fundus_eroded, mask=xmask)
        blood_vessels = cv2.bitwise_not(segmented_img)
        blood_vessels = cv2.cvtColor(blood_vessels, cv2.COLOR_GRAY2BGR)

        blood_vessels[np.where((blood_vessels == [0, 0, 0]).all(axis=2))] = [0, 255, 0]
        blood_vessels[np.where((blood_vessels == [255, 255, 255]).all(axis=2))] = [0, 0, 0]

        return seg1, blood_vessels

    @staticmethod
    def segment(i_n):
        pp, seg = Test.segment_image(settings.MEDIA_ROOT+'/query/'+i_n)
        cv2.imwrite(settings.MEDIA_ROOT+'/pre-processed/'+i_n, pp)
        cv2.imwrite(settings.MEDIA_ROOT+'/segmented/'+i_n, seg)

    def test(self, i_n):
        from keras.models import load_model
        from keras.preprocessing import image
        from keras.applications.inception_v3 import preprocess_input
        model = load_model(settings.BASE_DIR+'/diabetic_retinopathy/retina_fe.model')
        self.segment(i_n)
        img = image.load_img(settings.MEDIA_ROOT+'/segmented/'+i_n, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = model.predict(x)
        flat = feature.flatten()
        flat = np.expand_dims(flat, axis=0)
        classifier = pickle.load(open(self.classifiers_path, 'rb'))
        classifiers_res = classifier.predict(flat)[0]
        txt = self.classifiers_name + ' :: ' + self.class_list[classifiers_res]
        f_p = settings.BASE_DIR+'/static/fonts/RobotoSlab-Light.ttf'
        print(f_p)
        orig = Image.open(settings.MEDIA_ROOT+'/query/'+i_n)
        fontsize = self.font_size_cal(orig, txt, f_p, fraction=0.5)
        font = ImageFont.truetype(f_p, fontsize)
        draw = ImageDraw.Draw(orig)
        draw.text((5, 5), txt, font=font, align="left")
        orig.save(settings.MEDIA_ROOT+'/classified/'+i_n)

        orig = Image.open(settings.MEDIA_ROOT+'/query/'+i_n)
        orig = orig.resize((320, 280), Image.ANTIALIAS)
        fontsize = self.font_size_cal(orig, txt, f_p, fraction=0.9)
        font = ImageFont.truetype(f_p, fontsize)
        draw = ImageDraw.Draw(orig)
        draw.text((5, 5), txt, font=font, align="left")
        orig.save(settings.MEDIA_ROOT + '/classified/' + i_n[:-3]+'320.jpg')

    @staticmethod
    def font_size_cal(img, text, font_path, fraction):
        fontsize = 1
        font_ = ImageFont.truetype(font_path, fontsize)
        while font_.getsize(text)[0] < fraction * img.size[0]:
            fontsize += 1
            font_ = ImageFont.truetype(font_path, fontsize)
        fontsize -= 1
        return fontsize
