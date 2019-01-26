import cv2
import numpy as np
import math
import pyautogui
#black
lowerBound = np.array([0, 0, 28])
upperBound = np.array([180, 255, 40])

#skin
#lowerBound = np.array([3, 30, 60])
#upperBound = np.array([10, 150, 255])

cam = cv2.VideoCapture(0)
kernelOpen = np.ones((4, 4))
kernelClose = np.ones((5, 5))
cX=0
cY=0
st = []
while True:
    ret, img2 = cam.read()
    img2 = cv2.flip(img2, 1)
    img2 = cv2.bilateralFilter(img2, 10, 50, 400)
    #img2=img1[10:240,320:630]
    imgHSV = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHSV, lowerBound, upperBound)

#    fgmask = fgbg.apply(img,4)

    erosion=cv2.erode(mask,kernelOpen,2)
    dilate=cv2.dilate(erosion,kernelClose,1)
    maskOpen = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernelOpen)
#    maskOpen = cv2.morphologyEx(maskOpen, cv2.MORPH_OPEN, kernelOpen)
    maskClose=maskOpen

    edges = cv2.Canny(maskOpen, 40, 80)
#    img = cv2.cvtColor(maskClose, cv2.COLOR_HSV2BGR)

    nes = cv2.bitwise_and(img2, img2, mask=maskClose)

    _, contours, heirarchy = cv2.findContours(maskClose, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    length = len(contours)
    maxArea = -1
    if length > 0:
        for i in range(length):
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i
        print(maxArea)
        res = contours[ci]
        hull1 = cv2.convexHull(res, True)
        cv2.drawContours(img2, [res], 0, (0, 255, 0), 1)
        cv2.drawContours(img2, [hull1], 0, (0, 0, 255), 3)
        hull = cv2.convexHull(res, returnPoints=False)
        if len(hull) > 3:
            defects = cv2.convexityDefects(res, hull)
            if type(defects) != type(None):  # avoid crashing.   (BUG not found)

                cnt = 1
                for i in range(defects.shape[0]):  # calculate the angle
                    s, e, f, d = defects[i][0]
                    start = tuple(res[s][0])

                    end = tuple(res[e][0])
                    far = tuple(res[f][0])
                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                    tArea= 0.5 * math.fabs( (start[0]-far[0])*(end[1]-far[1])-(end[0]-far[0])*(start[1]-far[1]) )


                    if angle <= math.pi / 2 and tArea>1000 and start[1]<cY and end[1]<cY and d>30:  # angle less than 90 degree, treat as fingers

                        cnt += 1
                        #st.append(start)
                        #if len(st)==5:
                        #    st=[]
                #if len(st)==4 and cnt==5:
                #    cv2.circle(img2, st[2], 6, [255, 0, 255], -1)
                #    cv2.circle(img2, st[3], 6, [0, 255, 255], -1)
                if cnt<3 :
                    pyautogui.press('up')
                cv2.putText(img2, str(cnt), (20, 420), cv2.FONT_HERSHEY_SIMPLEX,6, (255, 255, 255), 2)

        M = cv2.moments(res)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        cv2.circle(img2, (cX, cY), 2, (255, 0, 0), -1)
        #pyautogui.moveTo(cX*2*1366/(640)-1, cY*2*768/(480)-1)


    #cv2.imshow("mask", mask)
    cv2.imshow("maskClose", maskClose)
    cv2.imshow("img2", img2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()