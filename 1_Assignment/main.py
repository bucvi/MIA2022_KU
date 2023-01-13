from pathlib import Path
import SimpleITK as sitk
import numpy as np

def createNewImageFromArray(voxel_array, template_image):
    result_image = sitk.GetImageFromArray(voxel_array)
    result_image.SetOrigin(template_image.GetOrigin())
    result_image.SetSpacing(template_image.GetSpacing())
    result_image.SetDirection(template_image.GetDirection())
    return result_image

inputImageFileName = Path("data") / "VESSEL12_05_256.mhd"
outputImageFileName = Path("data") / "VESSEL12_05_256_inverted.mhd"
outputDenoisedFileName = Path("data") / "VESSEL12_05_256_denoised.mhd"
outputOtsuFileName = Path("data") / "VESSEL12_05_256_otsu.mhd"
outputShapeLabelFileName = Path("data") / "VESSEL12_05_256_label.mhd"
outputShapeLabelRegionGrowingFileName = Path("data") / "VESSEL12_05_256_label_region_growing.mhd"
outputLungMaskFileName = Path("data") / "VESSEL12_05_256_lung_mask.mhd"
outputLungMaskRegionGrowingFileName = Path("data") / "VESSEL12_05_256_lung_mask_region_growing.mhd"


reader = sitk.ImageFileReader()
reader.SetImageIO("MetaImageIO")
reader.SetFileName(str(inputImageFileName))
image = reader.Execute()

writer = sitk.ImageFileWriter()

# this gives us a 'view' on the intensity array, which we can manipulate like a numpy array
voxel_array = sitk.GetArrayViewFromImage(image)
min_intensity = voxel_array.min()
max_intensity = voxel_array.max()
full_intensity_range = max_intensity - min_intensity
# inverted_voxel_array = full_intensity_range - (voxel_array - min_intensity) + min_intensity

# # we use the inverted voxel values and create a new ITK image again, which requires a copy of the meta information
# # like origin, spacing and orientation!
# inverted_image = createNewImageFromArray(inverted_voxel_array, image)

# writer = sitk.ImageFileWriter()
# writer.SetFileName(str(outputImageFileName))
# writer.Execute(inverted_image)

# Normalize input volume
normalized_image = (voxel_array - min_intensity)/full_intensity_range
min_normalized_intensity = normalized_image.min() # Just for testing (should be 0.0)
max_normalized_intensity = normalized_image.max() # Just for testing (should be 1.0)

# Implement ROF primal-dual algorithm
tau_p = 0.02
tau_d = 2.0
lam = 1 # Range between 0.1 and 10.0
num_iterations = 300
f = normalized_image

outputDenoisedFileName = Path("data") / f"VESSEL12_05_256_denoised_lambda_{lam}_iterations_{num_iterations}.mhd"

def gradient(u):
    grad_x = np.diff(u, axis=0, append=0)
    grad_y = np.diff(u, axis=1, append=0)
    grad_z = np.diff(u, axis=2, append=0)
    grad = np.vstack((grad_x[np.newaxis, ...], grad_y[np.newaxis, ...], grad_z[np.newaxis, ...]))
    return grad

def div(p):
    dim = p.shape[-1]
    p_1 = np.diff(p[0], axis=0, prepend=p[0,0,:,:].reshape(1,dim,dim))
    p_2 = np.diff(p[1], axis=1, prepend=p[1,:,0,:].reshape(dim,1,dim))
    p_3 = np.diff(p[2], axis=2, prepend=p[2,:,:,0].reshape(dim,dim,1))
    return p_1 + p_2 + p_3

def project_p(p):
    return p/np.maximum(1, np.linalg.norm(p, axis=0))

def rof_denoising(f, lam, num_iterations, tau_p, tau_d):
    print("Start denoising!")
    p = np.zeros((len(f.shape),)+f.shape)
    u = np.copy(f)
    for n in range(0, num_iterations):
        if n%(num_iterations/10) == 0:
            print("Iteration:",n)
        p = project_p(p + tau_d*gradient(u))
        u = (u + tau_p * (div(p) + lam*f))/(1+tau_p * lam)
    print("Iteration:", num_iterations)
    print("Finished denoising!")
    return u, p
load_denoised = True

if load_denoised:
    reader.SetFileName(str(outputDenoisedFileName))
    denoised_image = reader.Execute()
else:
    u, p = rof_denoising(f, lam, num_iterations, tau_p, tau_d)

    denoised_image = createNewImageFromArray(u, image)
    writer.SetFileName(str(outputDenoisedFileName))
    writer.Execute(denoised_image)

# Apply an Otsu Threshold
num_bins = 128

otsu_filter = sitk.OtsuThresholdImageFilter()
segmented_image = otsu_filter.Execute(denoised_image)

writer.SetFileName(str(outputOtsuFileName))
writer.Execute(segmented_image)

# Shape Labeling
connected_component_filter = sitk.ConnectedComponentImageFilter()
shape_label_image = connected_component_filter.Execute(image, segmented_image)

labeled_image = sitk.GetArrayFromImage(shape_label_image)
labels = np.unique(labeled_image)

writer.SetFileName(str(outputShapeLabelFileName))
writer.Execute(shape_label_image)

# Choose lung
segments = sitk.GetArrayFromImage(segmented_image)
components = np.zeros(labels.shape + (3,))
for label in labels:
    mask = labeled_image==label
    components[label, 0] = label
    components[label, 1] = segments[mask][...,0]
    components[label, 2] = mask.sum()

dark_components = np.delete(components[components[:,1]==1], 1, 1)
lung_label = dark_components[dark_components[:,1].argsort()][-2,0]
lung_mask = (labeled_image==lung_label).astype(np.uint8)

lung_mask_image = createNewImageFromArray(lung_mask, image)
writer.SetFileName(str(outputLungMaskFileName))
writer.Execute(lung_mask_image)
