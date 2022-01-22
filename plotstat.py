import nibabel as nib
import os
import matplotlib.pyplot as plt

i=100

base_dir = '/home/fhd/projects/.DATASETS/HealthyControls/' #Enter path to HealthyControls Here
ind_dir = os.path.join(base_dir,'C01')
rec_path = os.path.join(ind_dir,'left_foot_trial_21.nii')
img = nib.load(rec_path)
image_data = img.get_fdata()

fig = plt.figure(figsize=(3,6))
plt.contourf(image_data[:,:,i],cmap='Reds')
plt.title(i)
plt.show()