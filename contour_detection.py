import glob
import os

import cv2


def get_bg_grayscale(img):
    height, weight = img.shape[1], img.shape[0]
    kernel_x = max(weight // 100, 5)
    kernel_y = max(height // 100, 5)
    return (img[0:kernel_y, 0:kernel_x].mean() + img[-kernel_y:, 0:kernel_x].mean() + img[0:kernel_y, -kernel_x:].mean() + \
    img[-kernel_y:, -kernel_x:].mean()) / 4.0


def union(rect1, rect2):
    # x, y, w, h
    x1 = min(rect1[0], rect2[0])
    y1 = min(rect1[1], rect2[1])
    x2 = max(rect1[0] + rect1[2], rect2[0] + rect2[2])
    y2 = max(rect1[1] + rect1[3], rect2[1] + rect2[3])
    return x1, y1, x2 - x1, y2 - y1


def object_detection(img, black_threshold=50, min_area_ratio=0.001):
    min_area = img.shape[0] * img.shape[1] * min_area_ratio

    # converting image into grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # setting threshold of gray image
    bg_grayscale = get_bg_grayscale(gray)
    if bg_grayscale < black_threshold:
        _, threshold = cv2.threshold(gray, bg_grayscale + 20, 255, cv2.THRESH_BINARY)
    else:
        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    contours = []
    for threshold in [threshold]:
        sub_contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(sub_contours)

    rects = [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) > min_area]
    merged_rect = rects[0]  # x, y, w, h
    for rect in rects[1:]:
        merged_rect = union(merged_rect, rect)

    return merged_rect


if __name__ == '__main__':
    # reading image
    img_paths = glob.glob(os.path.join('static', 'uploads',
                                       # '018.png'
                                       '*.png'
                                       ))
    # img_path = os.path.join('static', 'uploads', '000.png')

    for img_path in img_paths:
        img = cv2.imread(img_path)
        img_name = os.path.basename(img_path).split('.')[0]

        # converting image into grayscale image
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray', gray)
        # cv2.waitKey(0)

        # print(f"{img_path} background grayscale: {get_bg_grayscale(gray)}")

        # erode/dilate
        # kernel = np.ones((5, 5), np.uint8)
        # gray = cv2.erode(gray, kernel, iterations=1)
        # gray = cv2.dilate(gray, kernel, iterations=1)

        # histogram
        # hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        # binary_flag = cv2.THRESH_BINARY_INV if hist[:128].sum() > hist[128:].sum() else cv2.THRESH_BINARY

        # _, threshold_global_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        # cv2.imshow('threshold_global', threshold_global)
        # cv2.waitKey(0)
        # threshold2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
        # gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # _, threshold_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Max Bounding Rectangle
        # max_contour = max(contours, key=lambda c: cv2.contourArea(c))
        # x, y, w, h = cv2.boundingRect(max_contour)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # for i, contour in enumerate(contours):
        #     # remove small area
        #     if cv2.contourArea(contour) < min_area:
        #         continue
        #     # using drawContours() function
        #     # cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)
        #     x, y, w, h = cv2.boundingRect(contour)
        #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        x, y, w, h = object_detection(img)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite(os.path.join('static', 'uploads', 'bounding_box', f"{img_name}_bb.png"), img)

        # displaying the image after drawing contoursour)
        # cv2.imshow('shapes', img)
        # cv2.waitKey(0)

    # cv2.destroyAllWindows()
