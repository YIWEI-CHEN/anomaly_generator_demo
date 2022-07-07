import imgaug.augmenters as iaa
import glob
import os
import numpy as np
import random
import cv2
from functools import partial

from contour_detection import object_detection
from perlin import rand_perlin_2d_np


class AnomalyGenerator(object):
    def __init__(self, img_path, resize_shape=None):
        anomaly_source_path = os.path.join('static', 'anomaly_source')
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path + "/*.jpg"))
        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]
        self.resize_shape = resize_shape
        self.retry = 20
        self.img_path = img_path
        self.img = self.load_image(self.img_path, cv2.IMREAD_COLOR, False, floating=False)
        x, y, w, h = object_detection(img=self.img)
        self.start_x, self.start_y = x, y
        self.end_x, self.end_y = x + w, y + h

    def rotate(self, angle=None):
        if angle is None:
            return iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        else:
            return iaa.Sequential([iaa.Affine(rotate=angle)])

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def do_rot_aug(self):
        return False

    def load_image(self, image_path, flags=None, do_aug_orig=False, floating=True):
        image = cv2.imread(image_path, flags)
        if self.resize_shape is not None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        if do_aug_orig:
            image = self.rotate()(image=image)
        if floating:
            image = (image / 255.0).astype(np.float32)
        return image

    def perlin_noise(self, perlin_scale=6, beta=0.8):
        start_x, start_y, end_x, end_y = self.start_x, self.start_y, self.end_x, self.end_y
        image = (self.img / 255.0).astype(np.float32)

        anomaly_source_idx = random.randint(0, len(self.anomaly_source_paths) - 1)
        anomaly_source_path = self.anomaly_source_paths[anomaly_source_idx]
        anomaly_source_img = self.load_image(anomaly_source_path, cv2.IMREAD_COLOR, floating=False)
        anomaly_img_augmented = self.randAugmenter()(image=anomaly_source_img)

        min_perlin_scale = 0
        perlin_scalex = 2 ** (random.randint(min_perlin_scale, perlin_scale))
        perlin_scaley = 2 ** (random.randint(min_perlin_scale, perlin_scale))
        threshold = 0.5
        trial = 0
        while trial < self.retry:
            perlin_noise = rand_perlin_2d_np((self.resize_shape[1], self.resize_shape[0]),
                                             (perlin_scaley, perlin_scalex))
            perlin_noise = self.rotate()(image=perlin_noise)
            perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
            mask = np.zeros_like(perlin_thr)
            mask[start_y:end_y, start_x:end_x] = 1.0
            perlin_thr *= mask
            if np.sum(perlin_thr) != 0:
                break
            trial += 1
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = (anomaly_img_augmented * perlin_thr / 255.0).astype(np.float32)
        beta = random.random() * beta
        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (perlin_thr)

        augmented_image = augmented_image.astype(np.float32)
        msk = perlin_thr.astype(np.float32)
        augmented_image = msk * augmented_image + (1 - msk) * image
        has_anomaly = 1.0
        if np.sum(msk) == 0:
            has_anomaly = 0.0

        if self.do_rot_aug():
            angle = random.randint(-90, 90)
            augmented_image = self.rotate(angle=angle)(image=augmented_image)
            image = self.rotate(angle=angle)(image=image)
            msk = self.rotate(angle=angle)(image=msk)

        augmented_image = (augmented_image * 255.0).astype(np.uint8)
        image = (image * 255.0).astype(np.uint8)
        return image, augmented_image

    def create_mask(self, img, draw_func):
        mask = np.zeros((img.shape[1], img.shape[0]), dtype=img.dtype)
        draw_func(mask, color=(255, 255, 255))
        return mask

    def rand2int_retry(self, low, high):
        i = 0
        v1, v2 = np.random.randint(low, high, size=2)
        while v1 == v2:
            i += 1
            if i > self.retry:
                break
            v1, v2 = np.random.randint(low, high, size=2)
        return v1, v2

    def scar(self, cutpaste=False):
        start_x, start_y, end_x, end_y = self.start_x, self.start_y, self.end_x, self.end_y
        scar_max = self.resize_shape[0] // 5
        thickness_max = self.resize_shape[0] // 16
        image = self.img
        aug_img = image.copy()
        thickness = random.randint(1, thickness_max)

        org_w, org_h = end_x, end_y
        dx, dy = random.randint(0, scar_max), random.randint(0, scar_max)
        max_x = org_w - dx - (thickness + thickness % 2)
        max_y = org_h - dy - (thickness + thickness % 2)
        if start_x >= max_x:
            x1, paste_x1 = random.randint(0, max_x - 2), max_x - 1
        else:
            x1, paste_x1 = self.rand2int_retry(start_x, max_x)
        if start_y >= max_y:
            y1, paste_y1 = random.randint(0, max_y - 2), max_y - 1
        else:
            y1, paste_y1 = self.rand2int_retry(start_y, max_y)
        x2, y2 = x1 + dx, y1 + dy

        draw_func = partial(cv2.line, pt1=(x1, y1), pt2=(x2, y2), thickness=thickness)
        cut_mask = self.create_mask(img=image, draw_func=draw_func)

        cut_index = np.where(cut_mask == 255)
        paste_index = cut_index[0] + (paste_y1 - y1), cut_index[1] + (paste_x1 - x1)

        if cutpaste:
            scar = image[cut_index]
        else:
            color = np.random.randint(0, 255, size=(3,)).astype(image.dtype).tolist()
            scar = np.array(color, dtype=image.dtype)

        aug_img[paste_index] = scar
        has_anomaly = np.array([1.0], dtype=np.float32)

        # create mask
        draw_func.keywords["pt1"] = (paste_x1, paste_y1)
        draw_func.keywords["pt2"] = (paste_x1 + dx, paste_y1 + dy)
        mask = self.create_mask(img=image, draw_func=draw_func)

        if cutpaste and x1 == paste_x1 and y1 == paste_y1:
            mask = np.zeros((image.shape[1], image.shape[0]), dtype=image.dtype)
            has_anomaly = np.array([0.0], dtype=np.float32)

        if self.do_rot_aug():
            angle = random.randint(-90, 90)
            aug_img = self.rotate(angle=angle)(image=aug_img)
            image = self.rotate(angle=angle)(image=image)
            mask = self.rotate(angle=angle)(image=mask)

        return image, aug_img

    def cutout(self, area_ratio=(0.007, 0.07), aspect_ratio=(0.3, 3.3), cutpaste=False):
        start_x, start_y, end_x, end_y = self.start_x, self.start_y, self.end_x, self.end_y
        image = self.img
        aug_img = image.copy()

        org_w, org_h = end_x, end_y
        img_area = org_w * org_h

        patch_area = random.uniform(*area_ratio) * img_area
        patch_aspect = random.uniform(*aspect_ratio)
        patch_w = int(np.sqrt(patch_area * patch_aspect))
        patch_h = int(np.sqrt(patch_area / patch_aspect))

        max_x, max_y = org_w - patch_w, org_h - patch_h
        if start_x >= max_x:
            x1, paste_x1 = random.randint(0, max_x - 2), max_x - 1
        else:
            x1, paste_x1 = self.rand2int_retry(start_x, max_x)
        if start_y >= max_y:
            y1, paste_y1 = random.randint(0, max_y - 2), max_y - 1
        else:
            y1, paste_y1 = self.rand2int_retry(start_y, max_y)

        x2, y2 = x1 + patch_w, y1 + patch_h
        paste_x2, paste_y2 = paste_x1 + patch_w, paste_y1 + patch_h

        if cutpaste:
            patch = image[y1:y2, x1:x2, ...]
        else:
            color = np.random.randint(0, 255, size=(3,)).astype(image.dtype).tolist()
            patch = np.array(color, dtype=image.dtype)

        aug_img[paste_y1:paste_y2, paste_x1:paste_x2, ...] = patch
        has_anomaly = np.array([1.0], dtype=np.float32)
        mask = self.create_mask(img=image, draw_func=partial(cv2.rectangle,
                                                             pt1=(paste_x1, paste_y1),
                                                             pt2=(paste_x2, paste_y2),
                                                             thickness=-1))
        if cutpaste and x1 == paste_x1 and y1 == paste_y1:
            mask = np.zeros((image.shape[1], image.shape[0]), dtype=image.dtype)
            has_anomaly = np.array([0.0], dtype=np.float32)

        if self.do_rot_aug():
            angle = random.randint(-90, 90)
            aug_img = self.rotate(angle=angle)(image=aug_img)
            image = self.rotate(angle=angle)(image=image)
            mask = self.rotate(angle=angle)(image=mask)

        return image, aug_img

    def cutpaste(self):
        return self.cutout(cutpaste=True)

    def cutpaste_scar(self):
        return self.scar(cutpaste=True)


if __name__ == '__main__':
    img_size = 256
    generator = AnomalyGenerator(resize_shape=(img_size, img_size))
    img_path = os.path.join('static', 'uploads', '000.png')

    x = np.random.randint(0, img_size, size=2)
    # start_x, end_x = min(x), max(x)
    start_x, end_x = 0, img_size
    y = np.random.randint(0, img_size, size=2)
    # start_y, end_y = min(y), max(y)
    start_y, end_y = 0, img_size

    # img, aug_img = generator.perlin_noise(img_path)
    # img, aug_img = generator.cutout(img_path)
    img, aug_img = generator.scar(img_path)
    cv2.imwrite("aug_img.png", aug_img)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    # cv2.imshow("aug_img", aug_img)
    # cv2.waitKey(0)
    #
    # cv2.destroyAllWindows()
