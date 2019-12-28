import numpy as np
import cv2

#image = cv2.imread("Material/marker.png", cv2.IMREAD_GRAYSCALE)
image = cv2.imread("Material/marker.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#edges = cv2.Canny(gray, 200, 250)

contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


############################### Detect inner rectangles #################################
prev_center_of_mass = [0, 0]
inner_rectangles = []
for i in range(39):
    contour_points = np.squeeze(contours[i])
    center_of_mass = np.average(contour_points, axis=0)

    difference = center_of_mass - prev_center_of_mass
    distance = np.linalg.norm(difference)
    if distance < 10:       #If their centers are near, they are nested rectangles (for this example)
        inner_rectangles.append(i)

    prev_center_of_mass = center_of_mass

############################## Observe inner rectangles ##################################

#225 lenth of inner rectangles edge
rect_angle = []
for j in inner_rectangles:

    rect_points = np.squeeze(contours[j])
    center_of_mass = np.int32(np.average(contour_points, axis=0))

    if rect_points.shape[0] == 4:
        rect_angle.append(0)
    else:
        up_point_ind, left_point_ind = np.argmin(rect_points, axis=0)
        #print(up_point, left_point)
        horizontal_dist = rect_points[up_point_ind][1] - rect_points[left_point_ind][1]
        angle = np.arccos(horizontal_dist / 255)    #lenght of hypothenus is 255
        rect_angle.append(angle)
        #print(angle)


# I find the angles then i will separate those inner rectangles to 16 part and check each one of them to is there any circle in it



"""
for i in range(39):
    cv2.destroyAllWindows()

    kopya = np.copy(image)
    A = cv2.drawContours(kopya, contours, i, (255,255,0), 3)
    contour_points = np.squeeze(contours[i])
    print(contour_points)
    center_of_mass = np.average(contour_points, axis=0)
    print(center_of_mass)
    print("Contour i: ", contour_points.shape)

    cv2.imshow(str(i), kopya)
    cv2.waitKey()
"""
