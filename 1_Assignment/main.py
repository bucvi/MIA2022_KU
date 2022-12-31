from pathlib import Path
import SimpleITK as sitk


def createNewImageFromArray(voxel_array, template_image):
    result_image = sitk.GetImageFromArray(voxel_array)
    result_image.SetOrigin(template_image.GetOrigin())
    result_image.SetSpacing(template_image.GetSpacing())
    result_image.SetDirection(template_image.GetDirection())
    return result_image


inputImageFileName = Path("data") / "VESSEL12_05_256.mhd"
outputImageFileName = Path("VESSEL12_05_256_inverted.mhd")

reader = sitk.ImageFileReader()
reader.SetImageIO("MetaImageIO")
reader.SetFileName(str(inputImageFileName))
image = reader.Execute()

# this gives us a 'view' on the intensity array, which we can manipulate like a numpy array
voxel_array = sitk.GetArrayViewFromImage(image)
min_intensity = voxel_array.min()
max_intensity = voxel_array.max()
full_intensity_range = max_intensity - min_intensity
inverted_voxel_array = full_intensity_range - (voxel_array - min_intensity) + min_intensity

# Normalize input volume
normalized_image = (voxel_array - min_intensity)/full_intensity_range
min_normalized_intensity = normalized_image.min() # Just for testing (should be 0.0)
max_normalized_intensity = normalized_image.max() # Just for testing (should be 1.0)

# Implement ROF primal-dual algorithm
# TODO

# we use the inverted voxel values and create a new ITK image again, which requires a copy of the meta information
# like origin, spacing and orientation!
inverted_image = createNewImageFromArray(inverted_voxel_array, image)

writer = sitk.ImageFileWriter()
writer.SetFileName(str(outputImageFileName))
writer.Execute(inverted_image)



