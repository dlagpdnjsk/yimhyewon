# -*- coding: cp949 -*-
# -*- coding: utf-8 -*- # 한글 주석쓰려면 이거 해야함
import cv2  # opencv 사용
import numpy as np


def grayscale(img):  # 흑백이미지로 변환
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):  # Canny 알고리즘
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):  # 가우시안 필터
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 10)


def region_of_interest(img, vertices, color3=(255, 255, 255), color1=255):  # ROI 셋팅

    mask = np.zeros_like(img)  # mask = img와 같은 크기의 빈 이미지

    if len(img.shape) > 2:  # Color 이미지(3채널)라면 :
        color = color3
    else:  # 흑백 이미지(1채널)라면 :
        color = color1

    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움
    cv2.fillPoly(mask, vertices, color)

    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image


def draw_lines(img, lines, color=[135, 0, 128], thickness=8):  # 선 그리기
    if lines is None :
        return 0
    for line in lines:
        for x1, y1, x2, y2 in line:
            tmp = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if np.abs(tmp) > 20:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len=40, max_line_gap=4):  # 허프 변환
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_len,
                            max_line_gap)

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    return line_img


def weighted_img(img, initial_img, α=1, β=1., λ=0.):  # 두 이미지 operlap 하기
    return cv2.addWeighted(initial_img, α, img, β, λ)


cap = cv2.VideoCapture('test.mp4')  # 동영상 불러오기

while (cap.isOpened()):
    ret, image = cap.read()

    height, width = image.shape[:2]  # 이미지 높이, 너비
    ### 색 검출 ###

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([14, 90, 165])
    upper_yellow = np.array([25, 126, 195])

    lower_white = np.array([0, 7, 160])
    upper_white = np.array([50, 70, 255])

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    yellow_result = cv2.bitwise_and(image, image, mask=mask_yellow)
    white_result = cv2.bitwise_and(image, image, mask=mask_white)
    tmp_img = weighted_img(yellow_result, white_result)

    ##########################
    gray_img = grayscale(tmp_img)  # 흑백이미지로 변환

    blur_img = gaussian_blur(gray_img, 3)  # Blur 효과


    canny_img = canny(blur_img, 30, 200)  # Canny edge 알고리즘
    #canny_img1 = canny(blur_img, 150, 180)  # Canny edge 알고리즘
    #canny_img2 = canny(blur_img, 50, 200)  # Canny edge 알고리즘



#    vertices = np.array(
 #       [[(50, height), (width / 2 - 45, height / 2 + 60), (width / 2 + 45, height / 2 + 60), (width - 50, height)]],
  #      dtype=np.int32)

    vertices = np.array(
        [[(0, height), (0, height *2.3/5), (width, height *2.3/5), (width, height)]],
        dtype=np.int32)


    ROI_img = region_of_interest(canny_img, vertices)  # ROI 설정

    #hough_img = hough_lines(ROI_img, 4, 1* np.pi / 180, 100, 80, 5)  # 허프 변환
    hough_img = hough_lines(ROI_img, 4, 1 * np.pi / 180, 0, 80, 30)  # 허프 변환
    result = weighted_img(hough_img, image)  # 원본 이미지에 검출된 선 overlap

    #cv2.imshow('canny1', canny_img1)
    #cv2.imshow('canny2', canny_img2)

    #cv2.imshow('result', result)  # 결과 이미지 출력

    cv2.imshow('canny2', ROI_img)
    cv2.imshow('canny1', result)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=500, detectShadows=0)


while(1):
    ret, frame = cap.read()

    width = frame.shape[1]
    height = frame.shape[0]
    frame = cv2.resize(frame, (int(width), int(height)))

    fgmask = fgbg.apply(frame)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)

    for index, centroid in enumerate(centroids):
        if stats[index][0] == 0 and stats[index][1] == 0:
            continue
        if np.any(np.isnan(centroid)):
            continue

        x, y, width, height, area = stats[index]
        centerX, centerY = int(centroid[0]), int(centroid[1])

        if area > 10:
            cv2.circle(frame, (centerX, centerY), 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255))

    cv2.imshow('frame',frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release
cap.release()
cv2.destroyAllWindows()
