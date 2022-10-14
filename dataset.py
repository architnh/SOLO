import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt



# Function to import the .npy or .npz data files
def load_data(file_name):
    return np.load(file_name, allow_pickle=True, encoding='latin1')

image_path = 'hw3_mycocodata_img_comp_zlib.h5'
label_path = 'hw3_mycocodata_labels_comp_zlib.npy'
bbox_path = 'hw3_mycocodata_bboxes_comp_zlib.npy'
mask_path = 'hw3_mycocodata_mask_comp_zlib.h5'

bboxes = load_data(bbox_path)
labels = load_data(label_path)
h5_file = h5py.File(image_path,'r')
images = h5_file[list(h5_file.keys())[0]]
# h5_file.close()

"""
DATASET SHAPE HELPFUL FOR DEBUGGING

# print(np.shape(bboxes)) # (3265,)
# print(np.shape(labels)) # (3265,)
# print(bboxes[0])        # [[169.57968  85.1625  263.1281  170.32031]]
# print(labels[0])        # [3]
# print(np.shape(images)) # (3265, 3, 300, 400)
# print(type(images))     # <HDF5 dataset "data": shape (3265, 3, 300, 400), type "<u4">

"""

def display():
    edit_image = images[2].copy()
    edit_image = edit_image
    # print(np.max(images[0]))
    # edited_imag = edit_image.squeeze().permute(1,2,0)
    # edited_imag = np.transpose(edit_image)
    edit_image =  edit_image.swapaxes(0,1)
    edit_imag = edit_image.swapaxes(1,2)
    # edit_imag = edit_imag/255
    # print(np.shape(edit_imag[0]))

    # DISPLAY IMAGE
    # plt.figure(figsize = (10,10))
    plt.imshow(edit_imag)
    plt.show()

# Vectorized
"""
# Normalize Pixel Values

print('Min: %.3f, Max: %.3f' % (np.min(images[300]), np.max(images[300])))
images[:] = images[:]/255
print('Min: %.3f, Max: %.3f' % (np.min(images[300]), np.max(images[300])))


# Normalize Each Channel with mean and standard deviation
mean = [0.485, 0.456, 0.406] 
std = [0.229, 0.224, 0.225]
images[:, 0, :, :] = images[:, 0, :, :] - mean[0]
images[:, 1, :, :] = images[:, 1, :, :] - mean[1]
images[:, 2, :, :] = images[:, 2, :, :] - mean[2]

images[:, 0, :, :] = images[:, 0, :, :]/std[0]
images[:, 1, :, :] = images[:, 1, :, :]/std[1]
images[:, 2, :, :] = images[:, 2, :, :]/std[2]
"""

# Non - Vectorized
mean = [0.485, 0.456, 0.406] 
std = [0.229, 0.224, 0.225]
for img in images:
    # print('Original Min: %.3f, Max: %.3f' % (np.min(img), np.max(img)))
    img = img/255
    # print('After normalizing Min: %.3f, Max: %.3f' % (np.min(img), np.max(img)))
    img[0] -= mean[0]
    img[0] /= std[0]
    # print('After -Mean, /STD: Channel 1 Min: %.3f, Max: %.3f' % (np.min(img[0]), np.max(img[0])))

    img[1] -= mean[1]
    img[1] /= std[1]
    # print('After -Mean, /STD: Channel 2 Min: %.3f, Max: %.3f' % (np.min(img[1]), np.max(img[1])))

    img[2] -= mean[2]
    img[2] /= std[2]
    # print('After -Mean, /STD: Channel 3 Min: %.3f, Max: %.3f' % (np.min(img[2]), np.max(img[2])))


