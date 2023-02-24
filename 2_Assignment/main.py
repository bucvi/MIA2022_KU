from pathlib import Path
import SimpleITK as sitk
import numpy as np
from multiprocessing.pool import Pool
import time

def createNewImageFromArray(voxel_array, template_image):
    result_image = sitk.GetImageFromArray(voxel_array)
    result_image.SetOrigin(template_image.GetOrigin())
    result_image.SetSpacing(template_image.GetSpacing())
    result_image.SetDirection(template_image.GetDirection())
    return result_image

def normalize(image):
    voxel_array = sitk.GetArrayViewFromImage(image)
    min_intensity = voxel_array.min()
    max_intensity = voxel_array.max()
    full_intensity_range = max_intensity - min_intensity

    normalized_image = (voxel_array - min_intensity)/full_intensity_range
    return createNewImageFromArray(normalized_image, image)

def getGradient(filter, image):
    filter.SetOrder([1,0,0])
    g_x = sitk.GetArrayFromImage(filter.Execute(image))

    filter.SetOrder([0,1,0])
    g_y = sitk.GetArrayFromImage(filter.Execute(image))

    filter.SetOrder([0,0,1])
    g_z = sitk.GetArrayFromImage(filter.Execute(image))

    g = np.concatenate(
        (g_x[..., np.newaxis], g_y[..., np.newaxis], g_z[..., np.newaxis]), axis=-1)
    
    return g

def getHessian(filter, image):
    filter.SetOrder([2,0,0])
    g_xx = sitk.GetArrayFromImage(filter.Execute(image))

    filter.SetOrder([0,2,0])
    g_yy = sitk.GetArrayFromImage(filter.Execute(image))

    filter.SetOrder([0,0,2])
    g_zz = sitk.GetArrayFromImage(filter.Execute(image))

    filter.SetOrder([1,1,0])
    g_xy = sitk.GetArrayFromImage(filter.Execute(image))

    filter.SetOrder([1,0,1])
    g_xz = sitk.GetArrayFromImage(filter.Execute(image))

    filter.SetOrder([0,1,1])
    g_yz = sitk.GetArrayFromImage(filter.Execute(image))

    H_x = np.concatenate(
        (g_xx[..., np.newaxis], g_xy[..., np.newaxis], g_xz[..., np.newaxis]), axis=-1)
    H_y = np.concatenate(
        (g_xy[..., np.newaxis], g_yy[..., np.newaxis], g_yz[..., np.newaxis]), axis=-1)
    H_z = np.concatenate(
        (g_xz[..., np.newaxis], g_yz[..., np.newaxis], g_zz[..., np.newaxis]), axis=-1)

    H = np.concatenate(
        (H_x[...,np.newaxis, :], H_y[...,np.newaxis, :], H_z[...,np.newaxis, :]), axis=-2)

    return H

def getGradientAndHessian(image):
    derivative_filter = sitk.GradientImageFilter()
    derivative_filter.SetUseImageSpacing(False)
    index_filter = sitk.VectorIndexSelectionCastImageFilter()
    
    
    g = derivative_filter.Execute(image)
    index_filter.SetIndex(0)
    g_x = index_filter.Execute(g)
    index_filter.SetIndex(1)
    g_y = index_filter.Execute(g)
    index_filter.SetIndex(2)
    g_z = index_filter.Execute(g)

    H_x = sitk.GetArrayFromImage(derivative_filter.Execute(g_x))
    H_y = sitk.GetArrayFromImage(derivative_filter.Execute(g_y))
    H_z = sitk.GetArrayFromImage(derivative_filter.Execute(g_z))
    
    g = sitk.GetArrayFromImage(g)
    H = np.concatenate(
        (H_x[...,np.newaxis, :], H_y[...,np.newaxis, :], H_z[...,np.newaxis, :]), axis=-2)
    
    return g, H

def interpolate(step, g):
    x, y, z = step
    x_0, y_0, z_0 = np.floor(step).astype(int)
    x_1, y_1, z_1 = np.ceil(step).astype(int)

    with np.errstate(divide='ignore', invalid='ignore'):
        x_d = np.nan_to_num((x-x_0)/(x_1-x_0))
        y_d = np.nan_to_num((y-y_0)/(y_1-y_0))
        z_d = np.nan_to_num((z-z_0)/(z_1-z_0))

    g_00 = g[x_0, y_0, z_0] * (1-x_d) + g[x_1, y_0, z_0] * x_d
    g_01 = g[x_0, y_0, z_1] * (1-x_d) + g[x_1, y_0, z_1] * x_d
    g_10 = g[x_0, y_1, z_0] * (1-x_d) + g[x_1, y_1, z_0] * x_d
    g_11 = g[x_0, y_1, z_1] * (1-x_d) + g[x_1, y_1, z_1] * x_d

    g_0 = g_00 * (1-y_d) + g_10 * y_d
    g_1 = g_01 * (1-y_d) + g_11 * y_d

    g = g_0 * (1-z_d) + g_1 * z_d

    return g

def init_calculate_medialness_summands(B_local):
    global B
    B = B_local

def calculate_medialness_summands(steps, v_alpha):
    b_i = np.zeros(len(steps))
    for i, step in enumerate(steps):
        B_int = interpolate(step, B)
        b_int = np.linalg.norm(B_int)
        with np.errstate(divide='ignore', invalid='ignore'):
            g_int = np.nan_to_num(B_int/b_int)
        c_i = np.maximum(-g_int @ v_alpha[i], 0)
        b_i[i] = b_int * c_i**2
    return b_i

def tubular_structure_enhancement(image, mask, radii, sigma, gamma, use_gaussian=True):
    tubular_structure = np.zeros((len(radii),) + image.GetSize())
    writer = sitk.ImageFileWriter()
    
    # init gaussian derivative image filter
    gaussian_filter = sitk.DiscreteGaussianDerivativeImageFilter()
    gaussian_filter.SetUseImageSpacing(False)
    gaussian_filter.SetVariance(sigma**2)

    # Construct a scale-space via Guassian kernels and increasing sigma
    gaussian_filter.SetOrder(0)
    multiply = sitk.MultiplyImageFilter()
    boundary_scaled_image = multiply.Execute(sigma**gamma,gaussian_filter.Execute(image))

    # calculate gradient
    B = getGradient(gaussian_filter, boundary_scaled_image)

    lung_mask = sitk.GetArrayFromImage(mask).astype(bool)
    x_idx = np.vstack(np.where(lung_mask)).T

    for idx, radius in enumerate(radii):
        gaussian_filter.SetVariance(radius**2)
        gaussian_filter.SetOrder(0)
        multiply = sitk.MultiplyImageFilter()
        scaled_image = multiply.Execute(radius**gamma,gaussian_filter.Execute(image))

        image_gradient = getGradient(gaussian_filter, scaled_image)
        image_hessian = getHessian(gaussian_filter, scaled_image)

        eig_val, eig_vec = np.linalg.eig(image_hessian[lung_mask])

        order = np.argsort(np.abs(eig_val))
        eig_val = np.take_along_axis(eig_val, order, axis=1)
        eig_vec = np.take_along_axis(eig_vec, np.repeat(order[:,np.newaxis,:], 3, axis=1), axis=2)

        alpha = np.linspace(0, 2*np.pi, num=int(2*np.pi*radius+1))
        v_2 = eig_vec[...,-2]
        v_3 = eig_vec[...,-1]

        v_alpha = np.cos(alpha)[np.newaxis, :, np.newaxis] * v_2[:, np.newaxis, :] \
                + np.sin(alpha)[np.newaxis, :, np.newaxis] * v_3[:, np.newaxis, :]
        v_alpha = np.real(v_alpha)

        x_steps = np.clip(x_idx[:, np.newaxis, :] + radius * v_alpha, 0, image.GetSize()[0]-1)
        b_i = np.zeros(x_steps.shape[:-1])
        
        with Pool(initializer=init_calculate_medialness_summands, initargs=(B,)) as pool:
            for n, res in enumerate(pool.starmap(calculate_medialness_summands, zip(x_steps,v_alpha))):
                b_i[n] = res
        R_i = np.mean(b_i, axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            S = np.nan_to_num(1 - np.var(b_i, axis=-1, ddof=1)/R_i**2)
        R_s = R_i * S
        R_c = np.linalg.norm(image_gradient[lung_mask], axis=-1)
        R = np.maximum(R_s - R_c, 0)
    
        tubular_structure[idx, lung_mask] = R
        writer.SetFileName(f"data/tub_{'g' if use_gaussian else 'd'}_{image.GetSize()[0]}_{idx}_{radius}.mhd")
        writer.Execute(createNewImageFromArray(tubular_structure[idx],image))

    return tubular_structure

def main():
    inputImageFileName = Path("data") / "VESSEL12_05_256.mhd"
    inputRegionGrowingFileName = Path("data") / "VESSEL12_05_256_region_growing.mhd"

    # params
    radii = [1, 1.5, 2]
    sigma = 0.9
    gamma = 1
    sampling_factor = 2
    pyramid_height = 2
    use_gaussian = True

    reader = sitk.ImageFileReader()
    reader.SetImageIO("MetaImageIO")

    reader.SetFileName(str(inputImageFileName))
    image = normalize(reader.Execute())

    reader.SetFileName(str(inputRegionGrowingFileName))
    mask = reader.Execute()

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetInterpolator(sitk.sitkLanczosWindowedSinc)
    #resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
    resample_filter.SetSize(image.GetSize())

    shrink_filter = sitk.ShrinkImageFilter()
    shrink_filter.SetShrinkFactor(sampling_factor)
    
    R = np.zeros(((pyramid_height+1)*len(radii),)+image.GetSize())
    t = time.time()
    print("Start")
    R[:len(radii)] = tubular_structure_enhancement(image, mask, radii, sigma, gamma, use_gaussian)
    print(f"Finished in {time.time()-t:.2f}s")

    downsampled_image = image
    sampled_mask = mask

    for i in range(1,pyramid_height+1):
        t = time.time()
        print("Start")
        downsampled_image = shrink_filter.Execute(downsampled_image)
        sampled_mask = shrink_filter.Execute(sampled_mask)
        
        tubular_structures = tubular_structure_enhancement(downsampled_image, sampled_mask, radii, sigma, gamma, use_gaussian)
        for j in range(len(radii)):
            R[i*len(radii)+j] = sitk.GetArrayFromImage(
                sitk.Expand(createNewImageFromArray(tubular_structures[j], downsampled_image), [sampling_factor**i]*3, sitk.sitkLinear))
        print(f"Finished in {time.time()-t:.2f}s")

    maximum_combination = np.maximum.reduce(R)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(f"data/tub_{'g' if use_gaussian else 'd'}_combined.mhd")
    writer.Execute(createNewImageFromArray(maximum_combination,image))

if __name__ == "__main__":
    main()