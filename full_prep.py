import os
import numpy as np
from scipy.io import loadmat
import h5py
from scipy.ndimage.interpolation import zoom
from skimage import measure
import warnings
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
from multiprocessing import Pool
from functools import partial
import warnings
import matplotlib.pyplot as plt
from preprocessing.step12 import step1_python
import SimpleITK as sitk
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
import scipy
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.segmentation import clear_border
from skimage.measure import label,regionprops, perimeter
from skimage.filters import roberts, sobel
from scipy import ndimage as ndi
from glob import glob
from skimage.transform import resize


# return dilated masks
def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>2*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10) 
    return dilatedMask


# a kind of lung segmentation operate
def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg

# 3D resample
# new_spacing = [1,1,1]
def resample(imgs, spacing, new_spacing, order = 2):
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)

            height = 512
            width = 512
            new_imgs = np.ndarray([imgs.shape[0], height, width], dtype=np.float32)

            '''
            Solution to resample problem
            '''
            for fcount, img in enumerate(imgs):
                mean = np.mean(img)
                img = img - mean
                min = np.min(img)
                max = np.max(img)
                img = img / (max - min)
                new_img = resize(img, [512, 512])
                new_imgs[fcount] = new_img

        return new_imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')

# def savenpy(id):
id = 1
# use step1_python function to generate masks of lung
def savenpy(id,filelist,prep_folder,data_path,use_existing=True):
    warnings.filterwarnings("ignore")
    resolution = np.array([1,1,1])
    name = filelist[id]
    if use_existing:
        if os.path.exists(os.path.join(prep_folder,name+'_label.npy')) and os.path.exists(os.path.join(prep_folder,name+'_clean.npy')):
            print(name+' had been done')
            return
    try:
        im, m1, m2, spacing = step1_python(os.path.join(data_path,name))
        Mask = m1+m2
        
        newshape = np.round(np.array(Mask.shape)*spacing/resolution)
        xx,yy,zz= np.where(Mask)
        box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
        box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
        box = np.floor(box).astype('int')
        margin = 5
        extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
        extendbox = extendbox.astype('int')

        convex_mask = m1
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1+dm2
        Mask = m1+m2
        extramask = dilatedMask ^ Mask
        bone_thresh = 210
        pad_value = 170

        im[np.isnan(im)]=-2000
        sliceim = lumTrans(im)
        sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
        bones = sliceim*extramask>bone_thresh
        sliceim[bones] = pad_value
        sliceim1, true_spacing = resample(sliceim,spacing,resolution,order=1)
        print(sliceim1.shape)
        print(true_spacing)
        sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                    extendbox[1,0]:extendbox[1,1],
                    extendbox[2,0]:extendbox[2,1]]
        sliceim = sliceim2[np.newaxis,...]
        np.save(os.path.join(prep_folder,name+'_clean'),sliceim)
        np.save(os.path.join(prep_folder,name+'_label'),np.array([[0,0,0,0]]))
    except Exception:
        print('bug in '+name)
        print(str(Exception))
        raise
    print(name+' done')

# return a list of .mhd files
def read_filelist(path):
    l = [file for file in os.listdir(path) if '.mhd' in file if '.npy' not in file]
    l.sort()
    return l



# This function is quiet time consuming.
# data_path: file_list path
# return filelist and save segmented lung as .npy
def full_prep(data_path,prep_folder, num = 100, n_worker = None,use_existing=True):
    warnings.filterwarnings("ignore")
    if not os.path.exists(prep_folder):
        os.mkdir(prep_folder)
            
    print('starting preprocessing')
    pool = Pool(n_worker)
    # filelist = [f for f in os.listdir(data_path)]
    filelist = read_filelist(data_path)
    partial_savenpy = partial(savenpy,filelist=filelist,prep_folder=prep_folder,
                              data_path=data_path,use_existing=use_existing)

    # N = len(filelist)
    N = num
    _=pool.map(partial_savenpy,range(N))
    pool.close()
    pool.join()
    print('end preprocessing')
    return filelist

# show a bunch of scans at a time
def plot_ct_scan(scan):
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(25, 25))
    for i in range(0, scan.shape[0], 5):
        plots[int(i / 20), int((i % 20) / 5)].axis('off')
        plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i], cmap=plt.cm.bone)
    plt.show()

def read_original(case_path):
    itk_img = sitk.ReadImage(case_path)
    case_pixels = sitk.GetArrayFromImage(itk_img)
    return case_pixels


def get_pixels_hu(slices):
    image = np.stack([s for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

# show a CT scan in 3D coordinate
def plot_3d(image, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)
    # p = image
    # threshold = 604
    p = p[:, :, ::-1]
    # print(p.max(), p.min())

    verts, faces = measure.marching_cubes_classic(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


def read_from_csv(path):
    df = pd.read_csv(path, nrows=1186)
    # print(df.values[0][4])
    return df

'''
This funciton reads a '.mhd' file using SimpleITK and return the image array, 
origin and spacing of the image.
'''
def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    slope = np.array(itkimage)
    print(itkimage)

    return ct_scan, origin, spacing


'''
This function is used to convert the world coordinates to voxel coordinates using 
the origin and spacing of the ct_scan
'''
def world_2_voxel(world_coordinates, origin, spacing):
    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing
    return voxel_coordinates


'''
This function is used to convert the voxel coordinates to world coordinates using 
the origin and spacing of the ct_scan.
'''
def voxel_2_world(voxel_coordinates, origin, spacing):
    stretched_voxel_coordinates = voxel_coordinates * spacing
    world_coordinates = stretched_voxel_coordinates + origin
    return world_coordinates


def seq(start, stop, step=1):
    n = int(round((stop - start) / float(step)))
    if n > 1:
        return ([start + step * i for i in range(n + 1)])
    else:
        return ([])


'''
This function is used to create spherical regions in binary masks
at the given locations and radius.
'''
def draw_circles(image, cands, origin, spacing):
    # make empty matrix, which will be filled with the mask
    RESIZE_SPACING = [1, 1, 1]

    image_mask = np.zeros(image.shape)

    # run over all the nodules in the lungs
    for ca in cands.values:
        # get middel x-,y-, and z-worldcoordinate of the nodule
        radius = np.ceil(ca[4]) / 2
        coord_x = ca[1]
        coord_y = ca[2]
        coord_z = ca[3]
        image_coord = np.array((coord_z, coord_y, coord_x))

        # determine voxel coordinate given the worldcoordinate
        image_coord = world_2_voxel(image_coord, origin, spacing)

        # determine the range of the nodule
        noduleRange = seq(-radius, radius, RESIZE_SPACING[0])

        # create the mask
        for x in noduleRange:
            for y in noduleRange:
                for z in noduleRange:
                    coords = world_2_voxel(np.array((coord_z + z, coord_y + y, coord_x + x)), origin, spacing)
                    if (np.linalg.norm(image_coord - coords) * RESIZE_SPACING[0]) < radius:
                        # print(int(np.round(coords[0])), int(np.round(coords[1])), int(np.round(coords[2])))
                        image_mask[int(np.round(coords[0])), int(np.round(coords[1])), int(np.round(coords[2]))] = int(1)


                        # tmp = voxel_2_world(coords, origin, spacing)
                        # tmp2 = world_2_voxel(tmp, origin, spacing)
                        # print(int(np.round(tmp[0])), int(np.round(tmp[1])), int(np.round(tmp[2])))
                        # print('................')

    # resize_factor = spacing / RESIZE_SPACING
    # new_real_shape = image_mask.shape * resize_factor
    # new_shape = np.round(new_real_shape)
    # real_resize = new_shape / image_mask.shape
    # new_spacing = spacing / real_resize
    #
    # # resize image
    # image_mask = scipy.ndimage.interpolation.zoom(image_mask, real_resize)

    return image_mask


def get_segmented_lungs(im, plot=False):
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(1, 9, figsize=(40, 40))
    orig = np.copy(im)
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < -400
    # binary = im < 600

    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone)
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone)
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone)

    if plot == True:
        plots[8].axis('off')
        plots[8].imshow(orig)

    plt.show()
    return im


'''
This function is much faster than step1_python.
'''
def segment_lung_from_ct_scan(ct_scan):
    return np.asarray([get_segmented_lungs(slice, plot=False) for slice in ct_scan])


'''
This function takes the path to a '.mhd' file as input and 
is used to create the nodule masks and segmented lungs after 
rescaling to 1mm size in all directions. It saved them in the .npz
format. It also takes the list of nodule locations in that CT Scan as 
input.
'''
def create_nodule_mask(imagePath, maskPath, cands):
    # if os.path.isfile(imagePath.replace('original',SAVE_FOLDER_image)) == False:
    img, origin, spacing = load_itk(imagePath)

    # calculate resize factor


    RESIZE_SPACING = [1, 1, 1]
    resize_factor = spacing / RESIZE_SPACING
    new_real_shape = img.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize = new_shape / img.shape
    new_spacing = spacing / real_resize

    # resize image
    lung_img = zoom(img, real_resize)

    # Segment the lung structure
    lung_img = lung_img + 1024
    lung_mask = segment_lung_from_ct_scan(lung_img)
    lung_img = lung_img - 1024

    # create nodule mask
    nodule_mask = draw_circles(lung_img, cands, origin, new_spacing)

    lung_img_512, lung_mask_512, nodule_mask_512 = np.zeros((lung_img.shape[0], 512, 512)), np.zeros(
        (lung_mask.shape[0], 512, 512)), np.zeros((nodule_mask.shape[0], 512, 512))

    original_shape = lung_img.shape
    for z in range(lung_img.shape[0]):
        offset = (512 - original_shape[1])
        upper_offset = int(np.round(offset / 2))
        lower_offset = offset - upper_offset

        new_origin = voxel_2_world([-upper_offset, -lower_offset, 0], origin, new_spacing)

        lung_img_512[z, upper_offset:-lower_offset, upper_offset:-lower_offset] = lung_img[z, :, :]
        lung_mask_512[z, upper_offset:-lower_offset, upper_offset:-lower_offset] = lung_mask[z, :, :]
        nodule_mask_512[z, upper_offset:-lower_offset, upper_offset:-lower_offset] = nodule_mask[z, :, :]

    # save images.
    imageName = 'aha'
    np.save('F:/lung_experiment/preprocessing/DSB2017-master/preprocessing/masks/' + imageName + '_lung_img.npz', lung_img_512)
    np.save('F:/lung_experiment/preprocessing/DSB2017-master/preprocessing/masks/' + imageName + '_lung_mask.npz', lung_mask_512)
    np.save('F:/lung_experiment/preprocessing/DSB2017-master/preprocessing/masks/' + imageName + '_nodule_mask.npz', nodule_mask_512)


def make_mask(center,diam,z,width,height,spacing,origin):
    '''
    Center : centers of circles px -- list of coordinates x,y,z
    diam : diameters of circles px -- diameter
    widthXheight : pixel dim of image
    spacing = mm/px conversion rate np array x,y,z
    origin = x,y,z mm np.array
    z = z position of slice in world coordinates mm
    '''
    #print("center={}\n,diam={}\n,z={}\n,width={}\n,height={}\n,spacing={}\n,origin={}".format(center,diam,z,width,
    # height,spacing,origin))
    #sys.exit(0)
    mask = np.zeros([height,width]) # 0's everywhere except nodule swapping x,y to match img
    #convert to nodule space from world coordinates
    #print(mask.shape)
    #sys.exit(0)
    # Defining the voxel range in which the nodule falls
    v_center = (center-origin)/spacing
    v_diam = int(diam/spacing[0]+5)
    v_xmin = np.max([0,int(v_center[0]-v_diam)-5])
    v_xmax = np.min([width-1,int(v_center[0]+v_diam)+5])
    v_ymin = np.max([0,int(v_center[1]-v_diam)-5])
    v_ymax = np.min([height-1,int(v_center[1]+v_diam)+5])

    v_xrange = range(v_xmin,v_xmax+1)
    v_yrange = range(v_ymin,v_ymax+1)

    # Convert back to world coordinates for distance calculation
    x_data = [x*spacing[0]+origin[0] for x in range(width)]
    y_data = [x*spacing[1]+origin[1] for x in range(height)]

    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            if np.linalg.norm(center-np.array([p_x,p_y,z]))<=diam:
                mask[int((p_y-origin[1])/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1.0
    return(mask)

def make_nonnode_mask(width,height):
    '''
Center : centers of circles px -- list of coordinates x,y,z
diam : diameters of circles px -- diameter
widthXheight : pixel dim of image
spacing = mm/px conversion rate np array x,y,z
origin = x,y,z mm np.array
z = z position of slice in world coordinates mm
    '''
    mask = np.zeros([height,width]) # 0's everywhere except nodule swapping x,y to match img

    return(mask)

def another_create_nodule_mask(imagePath, maskPath, df_node):
    file_list = read_filelist(imagePath)
    # print(len(file_list))

    for fcount, img_file in enumerate(file_list):
        print("Preprocessing start ******")

        mini_df = df_node[df_node["seriesuid"] + '.mhd' == img_file]  # get all nodules associate with file

        # print(df_node["seriesuid"] == '.'.join(img_file.split('.')[:-1]))
        # print('.'.join(img_file.split('.')[:-1]))

        # tmp = df_node[df_node["seriesuid"] + '.mhd' == '1.3.4']
        # print('tmp: ', tmp.shape[0])

        if mini_df.shape[0] > 0:  # some files may not have a nodule--skipping those
            # load the data once
            # print(mini_df)

            itk_img = sitk.ReadImage(imagePath + img_file)
            origin = np.array(itk_img.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
            spacing = np.array(itk_img.GetSpacing())  # spacing of voxels in world coor. (mm)
            # print(itk_img)

            img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (notice the ordering)
            seg_img_array = segment_lung_from_ct_scan(img_array)

            img_array, _ = resample(img_array, spacing, [1, 1, 1]) # resample original im
            new_img_array, _ = resample(seg_img_array, spacing, [1, 1, 1]) # resample segmented img

            num_z, height, width = new_img_array.shape  # heightXwidth constitute the transverse plane

            # print("mini df={} spacing={}".format(mini_df,spacing))
            # sys.exit(0)
            # plot_3d(img_array[0],400)

            # sys.exit(0)
            # go through all nodes (why just the biggest?)
            for node_idx, cur_row in mini_df.iterrows():
                node_x = cur_row["coordX"]
                node_y = cur_row["coordY"]
                node_z = cur_row["coordZ"]
                diam = cur_row["diameter_mm"]
                # print("##########Node :",cur_row)
                # sys.exit(0)
                # just keep 3 slices
                ori_imgs = np.ndarray([3, height, width], dtype=np.float32)
                seg_imgs = np.ndarray([3, height, width], dtype=np.float32)
                masks = np.ndarray([3, height, width], dtype=np.uint8)

                center = np.array([node_x, node_y, node_z])  # nodule center
                v_center = np.rint((center - origin) / spacing)  # nodule center in voxel space (still x,y,z ordering)
                # print(v_center)
                # print(center)
                # print(num_z)
                # v_center[2] = num_z/2
                # print("Img = {}".format(v_center))
                # sys.exit(0)
                for i, i_z in enumerate(np.arange(int(v_center[2]) - 1,
                                                  int(v_center[2]) + 2).clip(0,
                                                                             num_z - 1)):  # clip prevents going out of bounds in Z
                    mask = make_mask(center, diam, i_z * spacing[2] + origin[2],
                                     width, height, spacing, origin)
                    masks[i] = mask
                    ori_imgs[i] = img_array[i_z]
                    seg_imgs[i] = new_img_array[i_z]

                    # print(i_z)
                    # print("i = {} , i_z = {} , num ={}".format(i,i_z,num_z))
                # print("Img = {}".format(imgs))
                # sys.exit(0)

                # plt.imshow(new_img_array[0])
                # plt.figure()
                # plt.imshow(ori_imgs[0])
                # plt.figure()
                # plt.imshow(seg_imgs[0])
                # plt.figure()
                # plt.imshow(masks[0])
                # plt.show()

                np.save(os.path.join(maskPath, "images_%04d_%04d_%s.npy" % (fcount, node_idx, img_file)), ori_imgs)
                np.save(os.path.join(maskPath, "lungmask_%04d_%04d_%s.npy" % (fcount, node_idx, img_file)), seg_imgs)
                np.save(os.path.join(maskPath, "masks_%04d_%04d_%s.npy" % (fcount, node_idx, img_file)), masks)
                # sys.exit(0)
        elif mini_df.shape[0] == 0:
            # pass
            itk_img = sitk.ReadImage(imagePath + img_file)
            # print(itk_img)

            img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (notice the ordering)
            # print('img_array: ')
            # print(img_array[0])
            origin = np.array(itk_img.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
            spacing = np.array(itk_img.GetSpacing())  # spacing of voxels in world coor. (mm)

            seg_img_array = segment_lung_from_ct_scan(img_array)
            # plt.imshow(seg_img_array[seg_img_array.shape[0] // 2])
            # plt.figure()
            img_array, _ = resample(img_array, spacing, [1, 1, 1])  # resample original im
            new_img_array, _ = resample(seg_img_array, spacing, [1,1,1])
            # plt.imshow(new_img_array[int(new_img_array.shape[0] / 2)])
            # plt.figure()
            num_z, height, width = new_img_array.shape  # heightXwidth constitute the transverse plane

            v_center = num_z // 2
            # v_center = np.rint((center - origin) / spacing)  # nodule center in voxel space (still x,y,z ordering)
            # print(v_center)
            # print(center)
            # print(num_z)
            # v_center[2] = num_z/2
            # print("Img = {}".format(v_center))
            # sys.exit(0)
            masks = np.ndarray([3, height, width], dtype=np.uint8)
            imgs = np.ndarray([3, height, width], dtype=np.float32)
            seg_imgs = np.ndarray([3, height, width], dtype=np.float32)

            for i, i_z in enumerate(np.arange(int(v_center) - 1,
                                              int(v_center) + 2).clip(0,
                                                                         num_z - 1)):  # clip prevents going out of bounds in Z
                mask = make_nonnode_mask(width, height)
                masks[i] = mask
                imgs[i] = img_array[i_z]
                seg_imgs[i] = new_img_array[i_z]

            # plt.imshow(imgs[0])
            # plt.show()

            np.save(os.path.join(maskPath, "images_%04d_%04d_%s.npy" % (0, 0, img_file)), imgs) # not original img, segmented img
            np.save(os.path.join(maskPath, "lungmask_%04d_%04d_%s.npy" % (0, 0, img_file)), seg_imgs)
            np.save(os.path.join(maskPath, "masks_%04d_%04d_%s.npy" % (0, 0, img_file)), masks)

        print(img_file, " > Done")
        print('Preprocessing End ******')


'''
This function will show imgs of segmented lung and nodule mask
'''
def show_im_and_nod(img_path, nodule_path, n=0):
    im = np.load(img_path)
    nm = np.load(nodule_path)
    f, plots = plt.subplots(1, 3, figsize=(40, 40))

    # plt.imshow(im[n])
    # plt.figure()
    plots[0].axis('off')
    plots[0].imshow(im[n])

    get_high_vals = nm[n] == 1
    im[n][get_high_vals] = 255
    res = im[n]
    # plt.imshow(res)
    # plt.figure()
    plots[1].axis('off')
    plots[1].imshow(res)

    # plt.imshow(nm[n])
    plots[2].axis('off')
    plots[2].imshow(nm[n])
    plt.show()

def batch_show_im_and_nod(path):
    name_list = glob(path + "images_*.npy")
    num = len(name_list)
    file_list = list(map(lambda name: np.load(name), name_list))
    f, plots = plt.subplots(int(num/4)+1, 4, figsize=(25, 25))
    for i in range(num):
        plots[int(i / 4), int((i % 4))].axis('off')
        plots[int(i / 4), int((i % 4))].imshow(file_list[i][0])
        print(file_list[i].shape)
    plt.show()

def generate_dataset(working_path):
    file_list = glob(working_path + "lungmask_*.npy")
    out_images = []  # final set of images
    out_nodemasks = []  # final set of nodemasks
    for fname in file_list:
        # print("working on file ", fname)
        imgs_to_process = np.load(fname.replace("lungmask", "images"))
        masks = np.load(fname)
        node_masks = np.load(fname.replace("lungmask", "masks"))
        for i in range(len(imgs_to_process)):
            mask = masks[i]
            node_mask = node_masks[i]
            img = imgs_to_process[i]
            img = mask * img
            out_images.append(img)
            out_nodemasks.append(node_mask)

    num_images = len(out_images)
    #
    #  Writing out images and masks as 1 channel arrays for input into network
    #
    final_images = np.ndarray([num_images, 1, 512, 512], dtype=np.float32)
    final_masks = np.ndarray([num_images, 1, 512, 512], dtype=np.float32)
    for i in range(num_images):
        final_images[i, 0] = out_images[i]
        final_masks[i, 0] = out_nodemasks[i]

    rand_i = np.random.choice(range(num_images), size=num_images, replace=False)
    test_i = int(0.2 * num_images)
    np.save(working_path + "trainImages.npy", final_images[rand_i[test_i:]])
    np.save(working_path + "trainMasks.npy", final_masks[rand_i[test_i:]])
    np.save(working_path + "testImages.npy", final_images[rand_i[:test_i]])
    np.save(working_path + "testMasks.npy", final_masks[rand_i[:test_i]])


if __name__ == '__main__':

    INPUT_FOLDER = 'F:/lung_experiment/preprocessing/data/subset0/'
    HM_SLICES = 2

    '''
    By doing so, we'll have num _clean.npy & _label.npy files in OUTPUT_FOLDER
    '''
    OUTPUT_FOLDER = "F:/lung_experiment/preprocessing/data/u-net_work_path/"

    '''
    Generate dataset
    '''
    df = read_from_csv('F:/lung_experiment/preprocessing/data/CSVFILES/annotations.csv')
    another_create_nodule_mask(INPUT_FOLDER, OUTPUT_FOLDER, df)
    generate_dataset(OUTPUT_FOLDER)


    # ori_im = np.load(OUTPUT_FOLDER + 'images_0000_0000_1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd.npy')
    # print(ori_im.shape)
    # plt.imshow(ori_im[int(ori_im.shape[0] / 2)])
    # plt.show()

    # file_list = full_prep(INPUT_FOLDER, OUTPUT_FOLDER, num=HM_SLICES)

    # test = np.load('F:/lung_experiment/preprocessing/data/res/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd_clean.npy')
    # print(test.shape)
    # plt.imshow(test[0][45])
    # plt.figure()
    # final_shape = np.array([1, 3, 512, 512])
    # imgs = np.copy(test[0][45:48])
    # imgs = imgs[[np.newaxis,...]]
    # print(imgs.shape)
    # new_resize_factor = final_shape / imgs.shape
    # print(new_resize_factor)
    # new_imgs = zoom(imgs, new_resize_factor, mode='nearest', order=2)
    # print(new_imgs.shape)
    # plt.imshow(new_imgs[0][0])
    # plt.figure()
    # orig_im = read_original('F:/lung_experiment/preprocessing/data/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd')
    # plt.imshow(orig_im[int(orig_im.shape[0]/2)])
    # plt.show()
    #
    # np.set_printoptions(threshold=1e6)
    # print(new_imgs[0][0][0])
    # print(orig_im[int(orig_im.shape[0]/2)][0])

    '''
    Show the process of get_segmented_lungs()
    '''
    # orig_im = read_original('F:/lung_experiment/preprocessing/data/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd')
    # plot_3d(orig_im, 400)
    # plot_ct_scan(orig_im)
    # seg_im = get_segmented_lungs(orig_im[int(orig_im.shape[0]/2)], True)
    # plot_ct_scan(seg_im)

    # segment_lung_from_ct_scan(orig_im[int(orig_im.shape[0]/2):int(orig_im.shape[0]/2)+1])


    '''
    Generating segmented lung and nodule masks
    # '''
    # df = read_from_csv('F:/lung_experiment/preprocessing/data/CSVFILES/annotations.csv')
    # # create_nodule_mask('F:/lung_experiment/preprocessing/data/subset0/'
    # #                    '1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd', None, df)
    # another_create_nodule_mask('F:/lung_experiment/preprocessing/data/subset0/',
    #                            'F:/lung_experiment/preprocessing/data/work_path/', df)

    # batch_show_im_and_nod('F:/lung_experiment/preprocessing/data/testset/')

    # show_im_and_nod('F:/lung_experiment/preprocessing/data/new_testset/images_0000_0086_1.3.6.1.4.1.14519.5.2.1.6279.6001.124154461048929153767743874565.mhd.npy',
    #                 'F:/lung_experiment/preprocessing/data/new_testset/masks_0000_0086_1.3.6.1.4.1.14519.5.2.1.6279.6001.124154461048929153767743874565.mhd.npy')


    im = np.load('F:/lung_experiment/preprocessing/data/work_path/images_0005_0086_1.3.6.1.4.1.14519.5.2.1.6279.6001.124154461048929153767743874565.mhd.npy')
    lung_m = np.load('F:/lung_experiment/preprocessing/data/work_path/lungmask_0005_0086_1.3.6.1.4.1.14519.5.2.1.6279.6001.124154461048929153767743874565.mhd.npy')
    nodule_m = np.load('F:/lung_experiment/preprocessing/data/work_path/masks_0005_0086_1.3.6.1.4.1.14519.5.2.1.6279.6001.124154461048929153767743874565.mhd.npy')
    # res = im[0] * lung_m[0]
    res = im[0] * nodule_m[0]
    # plt.imshow(res)
    # plt.figure()
    # plt.imshow(lung_m[0])
    # plt.figure()
    # plt.imshow(im[0])
    # plt.show()



    '''
    plot 3D figure of lung
    '''
    # test_path = 'F:/lung_experiment/preprocessing/data/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd'
    # test = read_original(test_path)
    # plot of original slice
    # plot_3d(test, 0)
    # plot of segmented lung
    # seg_test = segment_lung_from_ct_scan(test)
    # plot_3d(seg_test, threshold=-400)

    '''
    Hounsfield Units (HU)
    '''
    # plt.hist(test.flatten(), bins=80, color='c')
    # plt.xlabel('HU')
    # plt.ylabel('Frequency')
    # plt.show()

    '''
    TEST: comparision betweent resampled slice and not resampled slice
    '''
    # resample_im = read_original('F:/lung_experiment/preprocessing/data/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.124154461048929153767743874565.mhd')
    # print(resample_im.shape)
    # plt.imshow(resample_im[40])
    # plt.figure()
    # itk_im = sitk.ReadImage('F:/lung_experiment/preprocessing/data/testset/1.3.6.1.4.1.14519.5.2.1.6279.6001.108197895896446896160048741492.mhd')
    # spacing = np.array(itk_im.GetSpacing())
    # resample_im, _ = resample(resample_im, spacing, [1,1,1])
    # m = int(32 * spacing[0]) + 1
    # print(resample_im.shape)
    # plt.imshow(resample_im[m])
    # plt.show()
    # seg_resample_im = segment_lung_from_ct_scan(resample_im)
    # plt.imshow(seg_resample_im[m])
    # plt.figure()
    # show_im_and_nod('F:/lung_experiment/preprocessing/data/testset/images_0000_0023_1.3.6.1.4.1.14519.5.2.1.6279.6001.108197895896446896160048741492.mhd.npy',
    #                 'F:/lung_experiment/preprocessing/data/testset/masks_0000_0023_1.3.6.1.4.1.14519.5.2.1.6279.6001.108197895896446896160048741492.mhd.npy')


    '''
    TEST
    '''
    # load_itk(test_path)

    # patients = [file for file in os.listdir(INPUT_FOLDER) if '.mhd' in file]
    # patients.sort()

    '''
    IGNORE: JUST to view some middle steps
    '''
    # m1 is left lung, m2 is right lung
    # case_pixels, m1, m2, spacing = step1_python(os.path.join(INPUT_FOLDER,'LKDS-00002.mhd'))
    # case_pixels, m1, m2, spacing = step1_python(os.path.join(INPUT_FOLDER,'1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd'))
    # print(m1.shape)
    # plt.imshow(m1[80])
    # plt.figure()
    # # plt.imshow(m2[80])
    # # plt.figure()
    # # plt.show()

    # dm1 = process_mask(m1)
    # print(dm1.shape)
    # plt.imshow(dm1[80])
    # plt.show()

    '''
    IGNORE: JUST to view some middle steps too
    '''
    # a = np.load('F:/lung_experiment/preprocessing/data/res/'
    #             '1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd_clean.npy')
    # print(a.shape)
    # print(a)
    # plt.imshow(a[0][80])
    # plt.show()
    #
    # test = read_original('F:/lung_experiment/preprocessing/data/subset0/'
    #             '1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd')
    # plot_ct_scan(test)
    # plt.show()

    '''
    IGNORE: I'm doing this to view the lung, lung mask and nodule mask
        now it's abstracted into the show_im_and_nod function
    '''
    # df = read_from_csv('F:/lung_experiment/preprocessing/data/CSVFILES/annotations.csv')
    # # create_nodule_mask('F:/lung_experiment/preprocessing/data/subset0/'
    # #                    '1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd', None, df)
    # another_create_nodule_mask('F:/lung_experiment/preprocessing/data/testset/', 'F:/lung_experiment/preprocessing/data/testset/', df)


    # test = read_original('F:/lung_experiment/preprocessing/data/subset0/'
    #             '1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd')
    # # plot_ct_scan(test)
    # # np.set_printoptions(threshold=1e6)
    # # print(test[0])
    # get_segmented_lungs(test[80], plot=True)

    # nm = np.load('F:/lung_experiment/preprocessing/data/testset/'
    #             'aha_nodule_mask.npz.npy')
    # im = np.load('F:/lung_experiment/preprocessing/data/testset/'
    #             'images_0000_0023_1.3.6.1.4.1.14519.5.2.1.6279.6001.108197895896446896160048741492.mhd.npy')
    # nm = np.load('F:/lung_experiment/preprocessing/data/testset/'
    #             'masks_0000_0023_1.3.6.1.4.1.14519.5.2.1.6279.6001.108197895896446896160048741492.mhd.npy')
    # # print(nm.shape)
    # # print(im[100])
    # # print(m[100])
    # n = 0
    # # np_im = np.asarray(im[n])
    # # np_m = np.asarray(m[n])
    # # np_nm = np.asarray(nm[n])
    # # np_im = np_im.astype(np.int64)
    # # np_m = np_m.astype(np.int64)
    # # np_nm = np_nm.astype(np.int64)
    #
    #
    # # tmp = im[n].copy()
    # # tmp[m[n] == 0] = 0
    # # plt.imshow(tmp)
    # # plt.figure()
    # plt.imshow(im[n])
    # plt.figure()
    # get_high_vals = nm[n] == 1
    # im[n][get_high_vals] = 255
    # res = im[n]
    # np.set_printoptions(threshold=1e6)
    # # res = np.bitwise_and(np_im, np_m)
    # plt.imshow(res)
    # # plt.figure()
    # plt.show()

