import logging
import math

import cv2
import exifread
import imageio
import numpy as np
from pylab import imshow, show # noqa F401

from .qr import find_qrcode

np.set_printoptions(precision=3)


def order_points(pts):
    """
    order such that [top-left,top-right,bottom-right,bottom-left]
    """
    # find which points are the topmost points
    by_top = np.argsort(pts[:,1].copy())
    by_left = np.argsort(pts[:,0].copy())
    top_left = [int(x) for x in by_top[:2] if x in by_left[:2]][0]
    bottom_left = [int(x) for x in by_left if x != top_left][0]
    top_right = [int(x) for x in by_top if x != top_left][0]
    bottom_right = [int(x) for x in [0,1, 2,3,] if x not in [top_left, bottom_left , top_right]][0]
    return pts[[top_left, top_right, bottom_right, bottom_left],:]


def rotmat(deg,w,h):
    r = math.radians(deg)
    return np.array([
        [ math.cos(r), -math.sin(r), h],
        [ math.sin(r), math.cos(r), w]
    ],dtype=np.float32)

# https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c

def get_rotation_transform(height, width, angle):
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

    return rotation_mat

def get_exif_rotation(im_file):
    rotation = None
    # Try with exif:
    tags = exifread.process_file(im_file)
    if isinstance(tags,dict) and 'Image Orientation' in tags:
        orientation = tags['Image Orientation'].values[0]
        if orientation == 6:
            rotation = -90
        elif orientation == 8:
            rotation = 90
        elif orientation == 3:
            rotation = 180
        else:
            rotation = 0
    im_file.seek(0)
    return im_file, rotation


def get_qr_rotation(image, detected_qrcode=None, qrprefix=''):
    rotation = None
    if not detected_qrcode:
        detected_qrcode = find_qrcode(image, qrprefix)
    if detected_qrcode:
        src_pts= np.array([(p.x,p.y) for p in detected_qrcode.polygon],np.float32)
        width = image.shape[1]
        height = image.shape[0]
        
        y_mean = np.mean(src_pts[:,1])
        flip_y = y_mean > height//2
        
        x_mean = np.mean(src_pts[:,0])
        flip_x = x_mean > width//2

        if flip_x and flip_y:
            rotation = 180

        if flip_y and not flip_x:
            rotation = 90

        if flip_x and not flip_y:
            rotation = -90
    return rotation, detected_qrcode


def rotate_image(image, rotation):
    logging.debug(f'Image is {rotation}')
    if rotation == -90:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    return image



def get_oriented_image(im_file, use_exif=True, use_qr=False, detected_qrcode=None, qrprefix=''):
    """
    Orient an image by it's exif data or by qr code that is expected to be on
    the image top-left side.
    :return: the rotated image, qrcode *in original cooordinates*, rotation in angles
    """

    # try exif
    im_file, rotation  = get_exif_rotation(im_file)

    image = imageio.imread(im_file)

    # if no oritenation in exif or don't use exif, maybe try qr code:
    detected_qrcode = None
    if rotation is None and use_qr or not use_exif:
        rotation, detected_qrcode = get_qr_rotation(image, detected_qrcode, qrprefix)

    image = rotate_image(image, rotation)

    # don't waste the qr code detection
    return image, detected_qrcode, rotation


def align_by_qrcode(image : np.ndarray, detected_qrcode, qrsize=100, boundery = 50.0, qrprefix=''):
    """
    Aling image by qrcode, normalize by qrcode size.
    Assumption - the image orietnation is already correct, if there is a qr code it's on the top
    left side of the image.
    WARNING: Aligining an image by a single qr code is pretty unstable. be carefull with this.
    """


    src_pts= np.array([(p.x,p.y) for p in detected_qrcode.polygon],np.float32)
    width = image.shape[1]
    height = image.shape[0]


    src_pts=order_points(src_pts).astype(np.float32)
    
    tgt_pts = np.array([[boundery,boundery],[qrsize+boundery,boundery],[qrsize+boundery,qrsize+boundery],[boundery,qrsize+boundery]],np.float32)
    shape_pts = np.array([[0,0],[width,0],[width,height],[0.0,height]],np.float32)
    M = cv2.getPerspectiveTransform(src_pts, tgt_pts)
    res = M @ np.concatenate([shape_pts,np.ones((4,1))],1).transpose()
    for r in range(4):
        res[:,r]/=res[-1,r]
    width = int(np.ceil(max(res[0,:]))) #+ int(np.floor(min(res[0,:])))
    height = int(np.ceil(max(res[1,:]))) #+ int(np.floor(min(res[1,:])))
    warped = cv2.warpPerspective(image, M,(width,height))
    return warped, M


def align_by_4_corners(image, corners, new_image_size=(1280, 768), margin_percent=10):
    """
    warp an image so the screen is aligned, and then crop the screen (with desired margin), and resize it to a desired size

    params:
    image - input screen image
    corners - screen coordinates on image 4 pairs of [x, y]
    new_image_size - tuple of (width, height) of the output patch
    margin_percent - margin percentage (from 0% to 40%) around the screen

    returns:
    warped_cropped - cropped patch after resize
    M - perspective transformation of the image
    """

    # check if margin percent is feasible
    if margin_percent > 40 or margin_percent < 0 :
        print('margin percent is illegal')
        warped = []
        M = []
        return warped, M

    width = image.shape[1]
    height = image.shape[0]

    # src_pts = screen corners
    src_pts = order_points(corners).astype(np.float32)  # [top_left, top_right, bottom_right, bottom_left]

    top = min(src_pts[0, 1], src_pts[1, 1])
    bottom = max(src_pts[2, 1], src_pts[3, 1])
    left = min(src_pts[0, 0], src_pts[3, 0])
    right = max(src_pts[1, 0], src_pts[2, 0])

    # tgt_pts = desired coordinates of the screen
    tgt_pts = np.float32([[left, top],
                          [right, top],
                          [right, bottom],
                          [left, bottom]])

    # find screen size after transformation
    screen_width = right - left
    screen_height = bottom - top

    # find cropped patch size after transformation (including margin)
    patch_width = screen_width / (1 - 2 * margin_percent / 100)
    patch_height =  screen_height / (1 - 2 * margin_percent / 100)

    # find margin in pixels from each size
    width_marg_pix = (patch_width - screen_width) / 2
    height_marg_pix = (patch_height - screen_height) / 2

    # shape_pts - size of image at the end
    shape_pts = np.array([[0, 0], [width, 0], [width, height], [0.0, height]], np.float32)

    # find transformation matrix which transforms screen corner to a rectangle
    M = cv2.getPerspectiveTransform(src_pts, tgt_pts)

     # find the transformation of image corner to get the size of the new image
    res = M @ np.concatenate([shape_pts, np.ones((4, 1))], 1).transpose()
    norm_res = res[:-1, :]/[res[-1, :], res[-1, :]]  # normalize coordinates by last row

    # warp image according to the transformation
    dest_width = int(np.ceil(max(norm_res[0, :])))
    dest_height = int(np.ceil(max(norm_res[1, :])))
    warped = cv2.warpPerspective(image, M, (dest_width, dest_height))

    # crop screen from image including margin
    crop_top = max(int(top - height_marg_pix), 0)
    crop_bottom = min(int(bottom + height_marg_pix + 1), dest_height)
    crop_left = max(int(left - width_marg_pix), 0)
    crop_right = min(int(right + width_marg_pix + 1), dest_width)
    warped_cropped = warped[crop_top : crop_bottom, crop_left : crop_right]

    # resize cropped image to an input size
    if new_image_size:
        warped_cropped = cv2.resize(warped_cropped, new_image_size)

    return warped_cropped, M

