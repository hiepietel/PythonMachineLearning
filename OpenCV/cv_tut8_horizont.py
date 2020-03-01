#https://www.youtube.com/watch?v=eLTLtUVuuy4
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
rate = 0.25
def save_plt(image, title):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.savefig(title+".png", bbox_inches='tight')
    plt.show();

def region(image):
    height = image.shape[0]
    width = image.shape[1]

    rects = np.array([[(0, int(height * rate)),
                     (0, int(height * (1-rate))),
                     (width, int(height * (1 - rate))),
                     (width, int(height * rate))]])

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, rects, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_line(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            angle = math.atan2(y2- y1, x2 - x1) * 180 / np.pi
            if abs(int(angle)) < 5:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
    return line_image

def return_img(imagepath):
    #image = cv2.imread('../db/img/desert_sand_man_s.jpg')
    image = cv2.imread(str(imagepath))
    lane_image = np.copy(image)

    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    blur = image

    canny = cv2.Canny(blur, 50, 150)
    print("region we interest")

    cropped = region(canny)

    lines = cv2.HoughLinesP(cropped, 2, np.pi / 180, 3, np.array([]), minLineLength=40, maxLineGap=20)
    line_image = display_line(lane_image, lines)

    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

    save_plt(line_image, "line_image")
    save_plt(canny, "canny")
    save_plt(cropped, "cropped")
    save_plt(combo_image, "combo_image")


    return combo_image

name = 'horizont'
path = Path(".")
path = path.glob('../db/img/'+name+'/*.jpg')


for imagepath in path:
    combo_image = return_img(imagepath)
    plt.imshow(cv2.cvtColor(combo_image, cv2.COLOR_BGR2RGB))
    plt.title("blur")
    plt.show()


    break
# plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
# plt.title("line_image")
# plt.show()
# plt.imshow(cv2.cvtColor(canny, cv2.COLOR_BGR2RGB))
# plt.title("title")
# plt.show()
# plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
# plt.title("cropped")
# plt.show()
# plt.imshow(cv2.cvtColor(combo_image, cv2.COLOR_BGR2RGB))
# plt.title("combo_image")
# plt.show()
# cv2.imshow("result", line_image)
# cv2.waitKey(0)
