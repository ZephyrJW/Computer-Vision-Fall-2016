import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

def computeDistance(a,b):
    return np.sqrt(np.square(a[0]-b[0]) + np.square(a[1]-b[1]))

def drawCircle(img,a,b,r):
    theta = 0
    for theta in xrange(181):
        x = a + int(r*np.cos(theta))
        y = b + int(r*np.sin(theta))
        y1 = b - int(r*np.sin(theta))

        if x >= 0 and x < len(img):
            if y < len(img[0]) and y>=0:
                img[x][y] = (0,255,0)
            if y1>=0 and y1 < len(img[0]):
                img[x][y1] = (0,255,0)

img = cv2.imread('/Users/jwzhao/Desktop/HoughCircles.jpg', 0)
output = cv2.imread('/Users/jwzhao/Desktop/HoughCircles.jpg', 1)


blur = cv2.GaussianBlur(img, (3,3), 0)
ret,thre = cv2.threshold(blur,128,255,cv2.THRESH_BINARY)
edgeImg = cv2.Canny(thre,70,100)

v = []
edges = np.where(edgeImg == 255)
for i in range(len(edges[0])):
    v.append((edges[0][i], edges[1][i]))

#set thresholds
tf = 30000
tmin = 300
ta = 20
td = 2
tr = 0.6

f = 0
vcount1 = 0
while(f < tf and len(v) >= tmin):
    agent = []
    x = len(v)
    for i in range(4):
        tmp = random.randint(0,len(v)-1)
        agent.append(v[tmp])
        v.remove(v[tmp])
        vcount1 = len(v)
    circle1 = [agent[0], agent[1], agent[2], agent[3]]
    circle2 = [agent[0], agent[1], agent[3], agent[2]]
    circle3 = [agent[0], agent[2], agent[3], agent[1]]
    circle4 = [agent[1], agent[2], agent[3], agent[0]]
    possibleCircles = [circle1, circle2, circle3, circle4]
    verifyCircles = []
    for item in possibleCircles:
        a = item[0]
        b = item[1]
        c = item[2]
        d = item[3]
        x1 = a[0]
        y1 = a[1]
        x2 = b[0]
        y2 = b[1]
        x3 = c[0]
        y3 = c[1]
        if computeDistance(a,b) > ta and computeDistance(a,c) > ta and computeDistance(b,c) > ta:
            if (x2-x1)*(y3-y1)-(x3-x1)*(y2-y1) != 0:
                centerx = ((np.square(x2)+np.square(y2)-np.square(x1)-np.square(y1))*2*(y3-y1) - (np.square(x3)+np.square(y3) - np.square(x1)-np.square(y1))*2*(y2-y1)) / (4*((x2-x1)*(y3-y1)-(x3-x1)*(y2-y1)))
                centery = (2*(x2-x1)*(np.square(x3)+np.square(y3)-np.square(x1)-np.square(y1)) - 2*(x3-x1)*(np.square(x2)+np.square(y2)-np.square(x1)-np.square(y1))) / (4*((x2-x1)*(y3-y1)-(x3-x1)*(y2-y1)))
                radii = np.sqrt(np.square(x1-centerx)+np.square(y1-centery))
                if np.abs(computeDistance(d, (centerx, centery))-radii) <= td:
                    verifyCircles.append((centerx, centery, radii))


    if len(verifyCircles)>0:
        #verification
        flag = 0
        for circle in verifyCircles:
            counter = 0
            removedPoint = []
            for point in v:
                if np.abs(computeDistance(point, (circle[0], circle[1])) - circle[2]) <= td:
                    counter += 1
                    removedPoint.append(point)
                    v.remove(point)

            if counter >= 2*np.pi*circle[2]*tr:
                cv2.circle(output, (int(circle[1]), int(circle[0])), int(circle[2]), (0,255,0),2)
                f = 0
                break
            else:
                f += 1
                for item in removedPoint:
                    v.append(item)
                if flag == 0:
                    for i in range(len(agent)):
                        v.append(agent[i])

                    flag = 1


    else:
        for i in range(len(agent)):
            v.append(agent[i])
        f += 1


plt.subplot(121),plt.imshow(edgeImg, cmap = 'gray')
plt.title('Edge Detection Result'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(output, cmap = 'gray')
plt.title('Final Result'), plt.xticks([]), plt.yticks([])
plt.show()
