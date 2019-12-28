import numpy as np
import nibabel as nib
import cv2


def Dice(X_seg, X_gt):

    return 2*np.count_nonzero(np.logical_and(X_seg, X_gt)) / (np.count_nonzero(X_seg) + np.count_nonzero(X_gt))


def read_seeds(filename):
    seed_file = open(filename, newline='')
    seeds = []
    for i in seed_file:
        seeds.append(np.int32(i.split("\t")))
    seeds = np.array(seeds).astype(int)
    return seeds


def get_8_neighbors(points, max_row, max_colm):    #takes nx2 points return their 8_neighbors (8nx2)

    result = np.array([[0,0]])
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0: continue  #exclude current point (0, 0)
            direction = np.array([i, j])
            neighbors_in_direction = points + direction
            result = np.vstack((result, neighbors_in_direction))

    row_in_limit = np.logical_and((result[:,0] >= 0), (result[:,0] < max_row))      #check if index of neighbors are in limit
    colm_in_limit = np.logical_and((result[:,1] >= 0), (result[:,1] < max_colm))     #check if index of neighbors are in limit
    in_limit =  np.logical_and(colm_in_limit, row_in_limit)     #check neighbors are in image

    neighbors = result[in_limit]

    return neighbors[1:]     #eliminate the first dummy element


def get_4_neighbors(points, max_row, max_colm):    #takes nx2 points return their 8_neighbors (8nx2)

    result = np.array([[0,0]])
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (abs(i) + abs(j)) != 1: continue  ## only get [-1 0] ; [0 -1] ; [1 0] ; [0 1]
            direction = np.array([i, j])
            neighbors_in_direction = points + direction
            result = np.vstack((result, neighbors_in_direction))

    row_in_limit = np.logical_and((result[:,0] >= 0), (result[:,0] < max_row))      #check if index of neighbors are in limit
    colm_in_limit = np.logical_and((result[:,1] >= 0), (result[:,1] < max_colm))     #check if index of neighbors are in limit
    in_limit =  np.logical_and(colm_in_limit, row_in_limit)     #check neighbors are in image

    neighbors = result[in_limit]

    return neighbors[:-1]     #eliminate the first dummy element


def area_growing(image, seeds, neighborhood, treshold=0.57):

    num_row, num_colm, num_depth = image.shape
    segmented = np.zeros(image.shape)

    for voxel in seeds:
        seg_points = voxel[:2].reshape(1, 2)
        while len(seg_points) > 0:
            if neighborhood == "8-neighborhood":
                neighbors = get_8_neighbors(seg_points, num_row, num_colm)
            else:
                neighbors = get_4_neighbors(seg_points, num_row, num_colm)

            old_seg_layer = np.copy(segmented[:, :, voxel[2]])  #save it in order to keep track changes in this iteration
            segmented[neighbors[:,0], neighbors[:,1], voxel[2]] = [1 if x else -1 for x in (image[neighbors[:,0], neighbors[:,1], voxel[2]] > treshold)]    #label as 1(full) if > treshold otherwise -1 for indicate it is visited
            change = segmented[:, :, voxel[2]] - old_seg_layer # it will give us changes in this iteration
            seg_points = np.vstack((np.where(change == 1)[0], np.where(change == 1)[1])).T

    segmented[(segmented == -1)] = 0    #clear -1 entries in segmented
    return segmented


################################################################################

def get_26_neighbors(points, max_row, max_colm, max_depth):    #takes nx2 points return their 8_neighbors (8nx2)

    result = np.array([[0, 0, 0]])
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                if (i == 0 and j == 0) and k == 0: continue  #exclude current point (0, 0, 0)
                direction = np.array([i, j, k ])
                neighbors_in_direction = points + direction
                result = np.vstack((result, neighbors_in_direction))

    row_in_limit = np.logical_and((result[:,0] >= 0), (result[:,0] < max_row))      #check if index of neighbors are in limit
    colm_in_limit = np.logical_and((result[:,1] >= 0), (result[:,1] < max_colm))     #check if index of neighbors are in limit
    depth_in_limit = np.logical_and((result[:,2] >= 0), (result[:,2] < max_depth))     #check if index of neighbors are in limit

    in_limit =  np.logical_and(np.logical_and(colm_in_limit, row_in_limit), depth_in_limit)     #check neighbors are in image
    neighbors = result[in_limit]

    return neighbors[1:]     #eliminate the first dummy element


def get_6_neighbors(points, max_row, max_colm, max_depth):    #takes nx2 points return their 8_neighbors (8nx2)

    result = np.array([[0, 0, 0]])
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                if (abs(i) + abs(j) + abs(k)) != 1: continue  ## only get [-1 0 0] ; [1 0 0] ; [0 -1 0] ; [0 1 0] ; [0 0 -1] ; [0 0 1]
                direction = np.array([i, j, k ])
                neighbors_in_direction = points + direction
                result = np.vstack((result, neighbors_in_direction))

    row_in_limit = np.logical_and((result[:,0] >= 0), (result[:,0] < max_row))      #check if index of neighbors are in limit
    colm_in_limit = np.logical_and((result[:,1] >= 0), (result[:,1] < max_colm))     #check if index of neighbors are in limit
    depth_in_limit = np.logical_and((result[:,2] >= 0), (result[:,2] < max_depth))     #check if index of neighbors are in limit

    in_limit =  np.logical_and(np.logical_and(colm_in_limit, row_in_limit), depth_in_limit)     #check neighbors are in image
    neighbors = result[in_limit]

    return neighbors[1:]     #eliminate the first dummy element


def volume_growing(image, seeds, neighborhood, treshold=0.57):

    num_row, num_colm, num_depth = image.shape
    segmented = np.zeros(image.shape)

    for voxel in seeds:
        seg_points = voxel.reshape(1, 3)
        while len(seg_points) > 0:
            if neighborhood == "26-neighborhood":
                neighbors = get_26_neighbors(seg_points, num_row, num_colm, num_depth)
            else:
                neighbors = get_6_neighbors(seg_points, num_row, num_colm, num_depth)

            old_seg = np.copy(segmented)  #save it in order to keep track changes in this iteration
            segmented[neighbors[:,0], neighbors[:,1], neighbors[:,2]] = [1 if x else -1 for x in (image[neighbors[:,0], neighbors[:,1], neighbors[:,2]] > treshold)]    #label as 1(full) if > treshold otherwise -1 for indicate it is visited
            change = segmented - old_seg # it will give us changes in this iteration
            seg_points = np.vstack((np.where(change == 1)[0], np.where(change == 1)[1], np.where(change == 1)[2])).T

    segmented[(segmented == -1)] = 0    #clear -1 entries in segmented
    return segmented

################################################################################

file2d = "Material/2D_seeds.txt"
file3d = "Material/3D_seed.txt"

tuD_seeds = read_seeds(file2d)
triD_seeds = read_seeds(file3d)

nii_data = nib.load("Material/V.nii")
image = nii_data.get_fdata()

#Ground Truth:
gt_seg = nib.load("Material/V_seg_05.nii")
gt_img = gt_seg.get_fdata()

#Part3a
seg_8_neighbors = area_growing(image, tuD_seeds, "8-neighborhood")
print("performance of 8-neighbors: ", Dice(seg_8_neighbors, gt_img))

#Part3b
seg_4_neighbors = area_growing(image, tuD_seeds, "4-neighborhood")
print("performance of 4-neighbors: ", Dice(seg_4_neighbors, gt_img))

#Part3c
seg_26_neighbors = volume_growing(image, triD_seeds, "26-neighborhood")
print("performance of 26-neighbors: ", Dice(seg_26_neighbors, gt_img))

#Part3d
seg_6_neighbors = volume_growing(image, triD_seeds, "6-neighborhood")
print("performance of 6-neighbors: ", Dice(seg_6_neighbors, gt_img))
