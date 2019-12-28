import numpy as np
import cv2


def Hough_circle(image, quantization, error_treshold=1.0):

    img_edge_map = cv2.Canny(image, 200, 250)
    edge_points = np.where(img_edge_map == 255)
    edge_coords = np.vstack((edge_points[0], edge_points[1])).T   # n x 2 edge point set

    num_row, num_colm = image.shape
    max_r = np.int(np.floor(min(num_row, num_colm)/2))

    Acc_tensor = np.zeros((num_row, num_colm, max_r))
    for r in range(1+quantization, max_r+1, quantization):   #include max_r  // Exclude r=1
        print(r, "/", max_r)
        for a in range(r, num_row-r, quantization):   #do not search incomplete circles, This will speed-up a little
            for b in range(r, num_colm-r, quantization):   #do not search incomplete circles, This will speed-up a little

                center = np.array([a, b])
                r_error = np.abs(np.sqrt(np.sum((edge_coords - center)**2, axis=1)) - r)     #compare edge_points distace from center with r value
                Acc_tensor[a, b, r] = np.count_nonzero(r_error < error_treshold)/r    #store how many edge point on this equation(with parameter a', b', r') on Acc_tensor, to normalize divide with (2*pi)*r since 2*pi constant only r

    return Acc_tensor



image = cv2.imread("Material/marker.png", cv2.IMREAD_GRAYSCALE)

patches = []
num_part = 2
num_row, num_colm = image.shape
for i in range(num_part):
    for j in range(num_part):
        image_part_ij = image[int(i*num_row/num_part) : int((i+1)*num_row/num_part), int(j*num_colm/num_part) : int((j+1)*num_colm/num_part)]
        if i < 2 and j < 2:

            A = Hough_circle(image_part_ij, 2)
            max_number = np.max(A)
            parameters = np.where(A >= 0.8*max_number)
            #print(parameters)
            a = np.int32(parameters[0])
            b = np.int32(parameters[1])
            r = np.int32(parameters[2])

            edge1 = cv2.Canny(image_part_ij, 200, 250)

            for k in range(a.size):
                #print(A[a[k], b[k], r[k]])
                edge1[a[k]-1:a[k]+1, b[k]-1:b[k]+r[k]] = 255

            patches.append(edge1)
            #name = "image_part" + str(i) + str(j)
            #cv2.imshow("image_part_11", edge1)

for i, img in enumerate(patches):
    name = "image_part_2_" + str(i) + ".png"
    cv2.imshow(name, p)
    #cv2.waitKey()
    #cv2.imwrite(name, img)
