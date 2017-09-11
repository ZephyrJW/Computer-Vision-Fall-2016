import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

# detect possible radius range from given image
def getRadius(img):
    minR = 100
    maxR = 0
    for i in xrange(len(img)):
        distance = 0
        s = 0
        for j in xrange(len(img[0])):
            if img[i][j] == 255:
                if s == 0:
                    s = 1
                else:
                    s = 0
                    if distance < minR:
                        minR = distance
                    if distance > maxR:
                        maxR = distance
                    distance = 0
            if s == 1:
                distance += 1
    maxR = maxR/2+1
    minR = minR/2+1

    return (minR, maxR)


def getAccumulator(img, rad):
    accumulateArray = np.zeros((rad[1]+1, len(img), len(img[0])))
    edges = np.where(img == 255)

    for m in xrange(len(edges[0])):
        i = edges[0][m]
        j = edges[1][m]

        for r in xrange(rad[0], rad[1]+1):
            x = j-r
            y = j+r
            done0 = 0
            for o in xrange(x,y+1):
                if o<0:
                    continue
                if o>=len(img[0]):
                    break

                axisy = np.int(np.sqrt(np.square(r) - np.square(j-o))) #used np.int to convert, not work, nan problem
        
                if axisy == 0 and done0 == 0:
                    done0 = 1
                    if j-r >=0:
                        accumulateArray[r][i][j-r] += 1
                    if j+r < len(img[0]):
                        accumulateArray[r][i][j+r] += 1
                if axisy != 0:
                    if i - axisy >=0 :
                        accumulateArray[r][i-axisy][o] += 1
                    if i + axisy < len(img) :
                        accumulateArray[r][i+axisy][o] += 1

    return accumulateArray

# drawing function for circles
def drawCircle(img,a,b,r):
    theta = 0
    for theta in xrange(181):
        x = a + int(r*np.cos(theta))
        y = b + int(r*np.sin(theta))
        y1 = b - int(r*np.sin(theta))

        if x >= 0 and x < len(img):
            if y < len(img[0]):
                img[x][y] = (0,255,0)
            if y1>=0 :
                img[x][y1] = (0,255,0)

# check well supported circles from accumulate array
def getCircles1(output, accumulateArray):
    drawnCircles = []
    tmp = np.copy(accumulateArray)
    while(tmp.max() >= 60):
        print tmp.max(), len(drawnCircles)
        center = np.where(tmp == tmp.max())
        for i in xrange(len(center[0])):
            r = center[0][i]
            a = center[1][i]
            b = center[2][i]
            tmp[r][a][b] = 0

            should_draw = 1
            if len(drawnCircles) == 0:
                cv2.circle(output, (b,a), r, (0,255,0), 2)
                drawnCircles.append((a,b,r))
            else:
                cntr = 1
                flag = 1
                for item in drawnCircles:
                    if np.abs(item[0]-a) <= r and np.abs(item[1]-b) <= r :
                        should_draw = 0
                        break
            if should_draw:
                drawCircle(output, a, b, r)
                drawnCircles.append((a,b,r))

# main
start = time.time()
# img is to process for edge detection and accmulation
img = cv2.imread('/Users/jwzhao/Desktop/HoughCircles.jpg', 0)
# output is to draw circles upon
output = cv2.imread('/Users/jwzhao/Desktop/HoughCircles.jpg', 1)

blur = cv2.GaussianBlur(img, (3,3), 0)
ret,thre = cv2.threshold(blur,128,255,cv2.THRESH_BINARY)
edgeImg = cv2.Canny(thre,50,100)
r = getRadius(edgeImg)

t1 = time.time() - start

accumulateArray = getAccumulator(edgeImg,r)
t2 = time.time() - t1

t3 = time.time() - t2
getCircles1(output ,accumulateArray)


plt.subplot(131),plt.imshow(edgeImg, cmap = 'gray')
plt.title('Edge Detection Result'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(accumulateArray[len(accumulateArray)/2], cmap = 'gray')
plt.title('Accumulator Result(partial)'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(output, cmap = 'gray')
plt.title('Final Result'), plt.xticks([]), plt.yticks([])
plt.show()