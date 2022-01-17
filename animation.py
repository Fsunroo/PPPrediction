import nibabel as nib
import os
import matplotlib.pyplot as plt
from functools import reduce
from celluloid import Camera

base_dir = '/home/fhd/projects/.DATASETS/HealthyControls/' #Enter path to HealthyControls Here
ind_dir = os.path.join(base_dir,'C01')
rec_path = os.path.join(ind_dir,'left_foot_trial_21.nii')
img = nib.load(rec_path)
image_data = img.get_fdata()

fig = plt.figure(figsize=(3,6))
camera = Camera(fig)
for i in range (image_data.shape[2]):
    plt.contourf(image_data[:,:,i],cmap='Reds')
    plt.title(i)
    if i:
        camera.snap()
animation = camera.animate()
animation.save(os.path.join('output','animation-1000.gif',fps=80))
    