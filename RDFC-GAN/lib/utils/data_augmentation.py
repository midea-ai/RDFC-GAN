import struct

import numpy as np
import cv2
import Imath
import OpenEXR
from PIL import Image
import random


def exr_loader(EXR_PATH, ndim=3):
    """Loads a .exr file as a numpy array
    Args:
        EXR_PATH: path to the exr file
        ndim: number of channels that should be in returned array. Valid values are 1 and 3.
                        if ndim=1, only the 'R' channel is taken from exr file
                        if ndim=3, the 'R', 'G' and 'B' channels are taken from exr file.
                            The exr file must have 3 channels in this case.
    Returns:
        numpy.ndarray (dtype=np.float32): If ndim=1, shape is (height x width)
                                          If ndim=3, shape is (3 x height x width)
    """

    exr_file = OpenEXR.InputFile(EXR_PATH)
    cm_dw = exr_file.header()['dataWindow']
    size = (cm_dw.max.x - cm_dw.min.x + 1, cm_dw.max.y - cm_dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    if ndim == 3:
        # read channels indivudally
        allchannels = []
        for c in ['R', 'G', 'B']:
            # transform data to numpy
            channel = np.frombuffer(exr_file.channel(c, pt), dtype=np.float32)
            channel.shape = (size[1], size[0])
            allchannels.append(channel)

        # create array and transpose dimensions to match tensor style
        exr_arr = np.array(allchannels).transpose((0, 1, 2))
        return exr_arr

    if ndim == 1:
        # transform data to numpy
        channel = np.frombuffer(exr_file.channel('R', pt), dtype=np.float32)
        channel.shape = (size[1], size[0])  # Numpy arrays are (row, col)
        exr_arr = np.array(channel)
        return exr_arr


def exr_saver(EXR_PATH, ndarr, ndim=3):
    '''Saves a numpy array as an EXR file with HALF precision (float16)
    Args:
        EXR_PATH (str): The path to which file will be saved
        ndarr (ndarray): A numpy array containing img data
        ndim (int): The num of dimensions in the saved exr image, either 3 or 1.
                        If ndim = 3, ndarr should be of shape (height, width) or (3 x height x width),
                        If ndim = 1, ndarr should be of shape (height, width)
    Returns:
        None
    '''
    if ndim == 3:
        # Check params
        if len(ndarr.shape) == 2:
            # If a depth image of shape (height x width) is passed, convert into shape (3 x height x width)
            ndarr = np.stack((ndarr, ndarr, ndarr), axis=0)

        if ndarr.shape[0] != 3 or len(ndarr.shape) != 3:
            raise ValueError(
                'The shape of the tensor should be (3 x height x width) for ndim = 3. Given shape is {}'.format(
                    ndarr.shape))

        # Convert each channel to strings
        Rs = ndarr[0, :, :].astype(np.float16).tostring()
        Gs = ndarr[1, :, :].astype(np.float16).tostring()
        Bs = ndarr[2, :, :].astype(np.float16).tostring()

        # Write the three color channels to the output file
        HEADER = OpenEXR.Header(ndarr.shape[2], ndarr.shape[1])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        HEADER['channels'] = dict([(c, half_chan) for c in "RGB"])

        out = OpenEXR.OutputFile(EXR_PATH, HEADER)
        out.writePixels({'R': Rs, 'G': Gs, 'B': Bs})
        out.close()
    elif ndim == 1:
        # Check params
        if len(ndarr.shape) != 2:
            raise ValueError(('The shape of the tensor should be (height x width) for ndim = 1. ' +
                              'Given shape is {}'.format(ndarr.shape)))

        # Convert each channel to strings
        Rs = ndarr[:, :].astype(np.float16).tostring()

        # Write the color channel to the output file
        HEADER = OpenEXR.Header(ndarr.shape[1], ndarr.shape[0])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        HEADER['channels'] = dict([(c, half_chan) for c in "R"])

        out = OpenEXR.OutputFile(EXR_PATH, HEADER)
        out.writePixels({'R': Rs})
        out.close()


def save_uint16_png(path, image):
    '''save weight file - scaled png representation of outlines estimation
        Args:
            path (str): path to save the file
            image (numpy.ndarray): 16-bit single channel image to be saved.
                                          Shape=(H, W), dtype=np.uint16
        '''
    assert image.dtype == np.uint16, ("data type of the array should be np.uint16." + "Got {}".format(image.dtype))
    assert len(image.shape) == 2, ("Shape of input image should be (H, W)" + "Got {}".format(len(image.shape)))

    array_buffer = image.tobytes()
    img = Image.new("I", image.T.shape)
    img.frombytes(array_buffer, 'raw', 'I;16')
    img.save(path)


def _normalize_depth_img(depth_img, dtype=np.uint8, min_depth=0.0, max_depth=1.0):
    '''Converts a floating point depth image to uint8 or uint16 image.
    The depth image is first scaled to (0.0, max_depth) and then scaled and converted to given datatype.
    Args:
        depth_img (numpy.float32): Depth image, value is depth in meters
        dtype (numpy.dtype, optional): Defaults to np.uint16. Output data type. Must be np.uint8 or np.uint16
        max_depth (float, optional): The max depth to be considered in the input depth image. The min depth is
            considered to be 0.0.
    Raises:
        ValueError: If wrong dtype is given
    Returns:
        numpy.ndarray: Depth image scaled to given dtype
    '''

    if dtype != np.uint16 and dtype != np.uint8:
        raise ValueError('Unsupported dtype {}. Must be one of ("np.uint8", "np.uint16")'.format(dtype))

    # Clip depth image to given range
    depth_img = np.ma.masked_array(depth_img, mask=(depth_img == 0.0))
    depth_img = np.ma.clip(depth_img, min_depth, max_depth)

    # Get min/max value of given datatype
    type_info = np.iinfo(dtype)
    min_val = type_info.min
    max_val = type_info.max

    # Scale the depth image to given datatype range
    depth_img = ((depth_img - min_depth) / (max_depth - min_depth)) * max_val
    depth_img = depth_img.astype(dtype)

    depth_img = np.ma.filled(depth_img, fill_value=0)  # Convert back to normal numpy array from masked numpy array

    return depth_img


def depth2rgb(depth_img, min_depth=0.0, max_depth=1.5, color_mode=cv2.COLORMAP_JET, reverse_scale=False,
              dynamic_scaling=False):
    '''Generates RGB representation of a depth image.
    To do so, the depth image has to be normalized by specifying a min and max depth to be considered.
    Holes in the depth image (0.0) appear black in color.
    Args:
        depth_img (numpy.ndarray): Depth image, values in meters. Shape=(H, W), dtype=np.float32
        min_depth (float): Min depth to be considered
        max_depth (float): Max depth to be considered
        color_mode (int): Integer or cv2 object representing Which coloring scheme to use.
                          Please consult https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html
                          Each mode is mapped to an int. Eg: cv2.COLORMAP_AUTUMN = 0.
                          This mapping changes from version to version.
        reverse_scale (bool): Whether to make the largest values the smallest to reverse the color mapping
        dynamic_scaling (bool): If true, the depth image will be colored according to the min/max depth value within the
                                image, rather that the passed arguments.
    Returns:
        numpy.ndarray: RGB representation of depth image. Shape=(H,W,3)
    '''
    # Map depth image to Color Map
    if dynamic_scaling:
        depth_img_scaled = _normalize_depth_img(depth_img, dtype=np.uint8,
                                                min_depth=max(depth_img[depth_img > 0].min(), min_depth),    # Add a small epsilon so that min depth does not show up as black (invalid pixels)
                                                max_depth=min(depth_img.max(), max_depth))
    else:
        depth_img_scaled = _normalize_depth_img(depth_img, dtype=np.uint8, min_depth=min_depth, max_depth=max_depth)

    if reverse_scale is True:
        depth_img_scaled = np.ma.masked_array(depth_img_scaled, mask=(depth_img_scaled == 0.0))
        depth_img_scaled = 255 - depth_img_scaled
        depth_img_scaled = np.ma.filled(depth_img_scaled, fill_value=0)

    depth_img_mapped = cv2.applyColorMap(depth_img_scaled, color_mode)
    depth_img_mapped = cv2.cvtColor(depth_img_mapped, cv2.COLOR_BGR2RGB)

    # Make holes in input depth black:
    depth_img_mapped[depth_img_scaled == 0, :] = 0

    return depth_img_mapped


def scale_depth(depth_image):
    '''Convert depth in meters (float32) to a scaled uint16 format as required by depth2depth module.
    Args:
        depth_image (numpy.ndarray, float32): Depth Image
    Returns:
        numpy.ndarray: scaled depth image. dtype=np.uint16
    '''

    assert depth_image.dtype == np.float32, "data type of the array should be float32. Got {}".format(depth_image.dtype)
    SCALING_FACTOR = 4000
    OUTPUT_DTYPE = np.uint16

    # Prevent Overflow of data by clipping depth values
    type_info = np.iinfo(OUTPUT_DTYPE)
    max_val = type_info.max
    depth_image = np.clip(depth_image, 0, np.floor(max_val / SCALING_FACTOR))

    return (depth_image * SCALING_FACTOR).astype(OUTPUT_DTYPE)


def unscale_depth(depth_image):
    '''Unscale the depth image from uint16 to denote the depth in meters (float32)
    Args:
        depth_image (numpy.ndarray, uint16): Depth Image
    Returns:
        numpy.ndarray: unscaled depth image. dtype=np.float32
    '''

    assert depth_image.dtype == np.uint16, "data type of the array should be uint16. Got {}".format(depth_image.dtype)
    SCALING_FACTOR = 4000

    return depth_image.astype(np.float32) / SCALING_FACTOR



def _get_point_cloud(color_image, depth_image, fx, fy, cx, cy):
    """Creates point cloud from rgb images and depth image
    Args:
        color image (numpy.ndarray): Shape=[H, W, C], dtype=np.uint8
        depth image (numpy.ndarray): Shape=[H, W], dtype=np.float32. Each pixel contains depth in meters.
        fx (int): The focal len along x-axis in pixels of camera used to capture image.
        fy (int): The focal len along y-axis in pixels of camera used to capture image.
        cx (int): The center of the image (along x-axis, pixels) as per camera used to capture image.
        cy (int): The center of the image (along y-axis, pixels) as per camera used to capture image.
    Returns:
        numpy.ndarray: camera_points - The XYZ location of each pixel. Shape: (num of pixels, 3)
        numpy.ndarray: color_points - The RGB color of each pixel. Shape: (num of pixels, 3)
    """
    # camera instrinsic parameters
    # camera_intrinsics  = [[fx 0  cx],
    #                       [0  fy cy],
    #                       [0  0  1]]
    camera_intrinsics = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    image_height = depth_image.shape[0]
    image_width = depth_image.shape[1]
    pixel_x, pixel_y = np.meshgrid(np.linspace(0, image_width - 1, image_width),
                                   np.linspace(0, image_height - 1, image_height))
    camera_points_x = np.multiply(pixel_x - camera_intrinsics[0, 2], (depth_image / camera_intrinsics[0, 0]))
    camera_points_y = np.multiply(pixel_y - camera_intrinsics[1, 2], (depth_image / camera_intrinsics[1, 1]))
    camera_points_z = depth_image
    camera_points = np.array([camera_points_x, camera_points_y, camera_points_z]).transpose(1, 2, 0).reshape(-1, 3)

    color_points = color_image.reshape(-1, 3)

    return camera_points, color_points


def write_point_cloud(filename, color_image, depth_image, fx, fy, cx, cy):
    """Creates and Writes a .ply point cloud file using RGB and Depth images.
    Args:
        filename (str): The path to the file which should be written. It should end with extension '.ply'
        color image (numpy.ndarray): Shape=[H, W, C], dtype=np.uint8
        depth image (numpy.ndarray): Shape=[H, W], dtype=np.float32. Each pixel contains depth in meters.
        fx (int): The focal len along x-axis in pixels of camera used to capture image.
        fy (int): The focal len along y-axis in pixels of camera used to capture image.
        cx (int): The center of the image (along x-axis, pixels) as per camera used to capture image.
        cy (int): The center of the image (along y-axis, pixels) as per camera used to capture image.
    """
    xyz_points, rgb_points = _get_point_cloud(color_image, depth_image, fx, fy, cx, cy)

    # Write header of .ply file
    with open(filename, 'wb') as fid:
        fid.write(bytes('ply\n', 'utf-8'))
        fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
        fid.write(bytes('element vertex %d\n' % xyz_points.shape[0], 'utf-8'))
        fid.write(bytes('property float x\n', 'utf-8'))
        fid.write(bytes('property float y\n', 'utf-8'))
        fid.write(bytes('property float z\n', 'utf-8'))
        fid.write(bytes('property uchar red\n', 'utf-8'))
        fid.write(bytes('property uchar green\n', 'utf-8'))
        fid.write(bytes('property uchar blue\n', 'utf-8'))
        fid.write(bytes('end_header\n', 'utf-8'))

        # Write 3D points to .ply file
        for i in range(xyz_points.shape[0]):
            fid.write(
                bytearray(
                    struct.pack("fffccc", xyz_points[i, 0], xyz_points[i, 1], xyz_points[i, 2],
                                rgb_points[i, 0].tostring(), rgb_points[i, 1].tostring(), rgb_points[i, 2].tostring())))




##################### some functions for cleargrasp dataset ##############################
def add_noise(image, level = 0.1):

    # random number
    r = np.random.rand(1)

    # gaussian noise
    if r < 0.9:
        row,col,ch= image.shape
        mean = 0
        noise_level = random.uniform(0, level)
        sigma = np.random.rand(1) * noise_level * 256
        gauss = sigma * np.random.randn(row,col) + mean
        gauss = np.repeat(gauss[:, :, np.newaxis], ch, axis=2)
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 255)
    else:
        # motion blur
        sizes = [3, 5, 7, 9, 11, 15]
        size = sizes[int(np.random.randint(len(sizes), size=1))]
        kernel_motion_blur = np.zeros((size, size))
        if np.random.rand(1) < 0.5:
            kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
        else:
            kernel_motion_blur[:, int((size-1)/2)] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        noisy = cv2.filter2D(image, -1, kernel_motion_blur)

    return noisy.astype('uint8')


def chromatic_transform(im, label=None, d_h=None, d_s=None, d_l=None):
    """
    Given an image array, add the hue, saturation and luminosity to the image
    """
    # Set random hue, luminosity and saturation which ranges from -0.1 to 0.1
    if d_h is None:
        d_h = (np.random.rand(1) - 0.5) * 0.1 * 180
    if d_l is None:
        d_l = (np.random.rand(1) - 0.5) * 0.2 * 256
    if d_s is None:
        d_s = (np.random.rand(1) - 0.5) * 0.2 * 256
    # Convert the BGR to HLS
    hls = cv2.cvtColor(im, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    # Add the values to the image H, L, S
    new_h = (h + d_h) % 180
    new_l = np.clip(l + d_l, 0, 255)
    new_s = np.clip(s + d_s, 0, 255)
    # Convert the HLS to BGR
    new_hls = cv2.merge((new_h, new_l, new_s)).astype('uint8')
    new_im = cv2.cvtColor(new_hls, cv2.COLOR_HLS2BGR)

    if label is not None:
        I = np.where(label > 0)
        new_im[I[0], I[1], :] = im[I[0], I[1], :]

    return new_im


def build_matrix_of_indices(height, width):
    """ Builds a [height, width, 2] numpy array containing coordinates.
        @return: 3d array B s.t. B[..., 0] contains y-coordinates, B[..., 1] contains x-coordinates
    """
    return np.indices((height, width), dtype=np.float32).transpose(1,2,0)


def compute_xyz(depth_img, camera_params):
    """ Compute ordered point cloud from depth image and camera parameters.
        If focal lengths fx,fy are stored in the camera_params dictionary, use that.
        Else, assume camera_params contains parameters used to generate synthetic data (e.g. fov, near, far, etc)
        @param depth_img: a [H x W] numpy array of depth values in meters
        @param camera_params: a dictionary with parameters of the camera used
    """

    # Compute focal length from camera parameters
    fx = camera_params['fx']
    fy = camera_params['fy']
    x_offset = camera_params['cx']
    y_offset = camera_params['cy']
    indices = build_matrix_of_indices(camera_params['yres'], camera_params['xres'])
    z_e = depth_img
    x_e = (indices[..., 1] - x_offset) * z_e / fx
    y_e = (indices[..., 0] - y_offset) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]

    return xyz_img


##### Depth Augmentations #####
def dropout_random_ellipses_4corruptmask(mask, noise_params):
    """ Randomly drop a few ellipses in the image for robustness.
        This is adapted from the DexNet 2.0 codebase.
        Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py
        @param depth_img: a [H x W] set of depth z values
    """
    dropout_mask = mask.copy()

    # Sample number of ellipses to dropout
    num_ellipses_to_dropout = np.random.poisson(noise_params['ellipse_dropout_mean'])

    # Sample ellipse centers
    zero_pixel_indices = np.array(np.where(dropout_mask == 0)).T # Shape: [#nonzero_pixels x 2]
    dropout_centers_indices = np.random.choice(zero_pixel_indices.shape[0], size=num_ellipses_to_dropout)
    dropout_centers = zero_pixel_indices[dropout_centers_indices, :] # Shape: [num_ellipses_to_dropout x 2]

    # Sample ellipse radii and angles
    x_radii = np.random.gamma(noise_params['ellipse_gamma_shape'], noise_params['ellipse_gamma_scale'], size=num_ellipses_to_dropout)
    y_radii = np.random.gamma(noise_params['ellipse_gamma_shape'], noise_params['ellipse_gamma_scale'], size=num_ellipses_to_dropout)
    angles = np.random.randint(0, 360, size=num_ellipses_to_dropout)

    # Dropout ellipses
    for i in range(num_ellipses_to_dropout):
        center = dropout_centers[i, :]
        x_radius = np.round(x_radii[i]).astype(int)
        y_radius = np.round(y_radii[i]).astype(int)
        angle = angles[i]

        # get ellipse mask
        tmp_mask = np.zeros_like(dropout_mask)
        tmp_mask = cv2.ellipse(tmp_mask, tuple(center[::-1]), (x_radius, y_radius), angle=angle, startAngle=0, endAngle=360, color=1, thickness=-1)
        # update depth and corrupt mask
        dropout_mask[tmp_mask == 1] = 1

    return dropout_mask

def dropout_random_ellipses_4mask(valid_mask, noise_params):
    """ Randomly drop a few ellipses in the image for robustness.
        This is adapted from the DexNet 2.0 codebase.
        Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py
        @param depth_img: a [H x W] set of depth z values
    """
    dropout_valid_mask = valid_mask.copy()

    # Sample number of ellipses to dropout
    num_ellipses_to_dropout = np.random.poisson(noise_params['ellipse_dropout_mean'])

    # Sample ellipse centers
    nonzero_pixel_indices = np.array(np.where(dropout_valid_mask > 0)).T # Shape: [#nonzero_pixels x 2]
    dropout_centers_indices = np.random.choice(nonzero_pixel_indices.shape[0], size=num_ellipses_to_dropout)
    dropout_centers = nonzero_pixel_indices[dropout_centers_indices, :] # Shape: [num_ellipses_to_dropout x 2]

    # Sample ellipse radii and angles
    x_radii = np.random.gamma(noise_params['ellipse_gamma_shape'], noise_params['ellipse_gamma_scale'], size=num_ellipses_to_dropout)
    y_radii = np.random.gamma(noise_params['ellipse_gamma_shape'], noise_params['ellipse_gamma_scale'], size=num_ellipses_to_dropout)
    angles = np.random.randint(0, 360, size=num_ellipses_to_dropout)

    # Dropout ellipses
    for i in range(num_ellipses_to_dropout):
        center = dropout_centers[i, :]
        x_radius = np.round(x_radii[i]).astype(int)
        y_radius = np.round(y_radii[i]).astype(int)
        angle = angles[i]

        # get ellipse mask
        mask = np.zeros_like(dropout_valid_mask)
        mask = cv2.ellipse(mask, tuple(center[::-1]), (x_radius, y_radius), angle=angle, startAngle=0, endAngle=360, color=1, thickness=-1)
        # update depth and corrupt mask
        dropout_valid_mask[mask == 1] = 0

    return dropout_valid_mask



def add_noise_to_depth(depth_img, noise_params):
    """ Distort depth image with multiplicative gamma noise.
        This is adapted from the DexNet 2.0 codebase.
        Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py
        @param depth_img: a [H x W] set of depth z values
    """
    depth_img = depth_img.copy()

    # Multiplicative noise: Gamma random variable
    multiplicative_noise = np.random.gamma(noise_params['gamma_shape'], noise_params['gamma_scale'])
    depth_img = multiplicative_noise * depth_img

    return depth_img

def add_noise_to_xyz(xyz_img, depth_img, noise_params):
    """ Add (approximate) Gaussian Process noise to ordered point cloud.
        This is adapted from the DexNet 2.0 codebase.
        @param xyz_img: a [H x W x 3] ordered point cloud
    """
    xyz_img = xyz_img.copy()

    H, W, C = xyz_img.shape

    # Additive noise: Gaussian process, approximated by zero-mean anisotropic Gaussian random variable,
    #                 which is rescaled with bicubic interpolation.
    small_H, small_W = (np.array([H, W]) / noise_params['gp_rescale_factor']).astype(int)
    additive_noise = np.random.normal(loc=0.0, scale=noise_params['gaussian_scale'], size=(small_H, small_W, C))
    additive_noise = cv2.resize(additive_noise, (W, H), interpolation=cv2.INTER_CUBIC)
    xyz_img[depth_img > 0, :] += additive_noise[depth_img > 0, :]

    return


def imread_indexed(filename):
    """ Load segmentation image (with palette) given filename."""
    im = Image.open(filename)
    annotation = np.array(im)
    return annotation
