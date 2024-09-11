import matplotlib as mpl
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from scipy.io import savemat

from skimage import measure
from TC_and_TS_define_param import grid_lower, grid_upper, grid_stride, folder2use, depth_layers, var2use, iso_b27

# Code attribution: https://stackoverflow.com/questions/56864378/how-to-light-and-shade-a-poly3dcollection
# Used via SE's default MIT license: https://meta.stackexchange.com/questions/272956/a-new-code-license-the-mit-this-time-with-attribution-required

rast=True

input_dir = folder2use + '/Outputs'
DEPTH_IDX = depth_layers
prefix = '_LOOCV'
lamb_choice = 'Custom' # need to change this today Oct 28
# lamb_choice = 'AdaptInflate' # need to change this today Oct 28
plot_dir = '../Figures/B27/'

# Define various cases
if DEPTH_IDX > 1 or var2use.startswith("ML"):
    sub_list = (
        # 'all_combined',
        'all_hurricanes',
        # 'all_tstd',
        # 'increasing_combined',
        # 'decreasing_combined',
        'increasing_hurricanes',
        'decreasing_hurricanes',
        # 'increasing_tstd',
        # 'decreasing_tstd',
    )
else: # single level
    sub_list = (
    'all_combined',
    'all_hurricanes',
    'all_tstd',
)
  
    
# --------------------------------------------------------------------------- #
def plot_isosurface(init_elev, init_azim, output_fname): 
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection=Axes3D.name)
    ax.view_init(init_elev, init_azim)

    # Values of the iso surfaces
    lower_lim = -iso_b27
    upper_lim = +iso_b27

    # Restrict to first 10 days, less noisy
    tps_half = tps_preds_mat[:200, :, :]
    mask_half = mask_mat[:200, :, :]
    
    # Iso surface for the lower_lim
    verts, faces, normals, values = measure.marching_cubes(tps_half, lower_lim,
	    mask=mask_half)
    mesh = Poly3DCollection(verts[faces], rasterized=rast)

    ls = LightSource(azdeg=225.0, altdeg=45.0)
    normalsarray = np.array([np.array((np.sum(normals[face[:], 0]/3),
                                       sum(normals[face[:], 1]/3), np.sum(normals[face[:],
	    2]/3))/np.sqrt(np.sum(normals[face[:], 0]/3)**2 +
		np.sum(normals[face[:], 1]/3)**2 + np.sum(normals[face[:],
		2]/3)**2)) for face in faces])

    # Next this is more asthetic, but it prevents the shadows of the image being too dark. (linear interpolation to correct)
    min = np.min(ls.shade_normals(normalsarray, fraction=1.0)) # min shade value
    max = np.max(ls.shade_normals(normalsarray, fraction=1.0)) # max shade value
    diff = max-min
    newMin = 0.3 # 0.3
    newMax = 0.95 # 0.95
    newdiff = newMax-newMin

    # Using a constant color, put in desired RGB values here.
    colourRGB = np.array((0/255.0, 53.0/255.0, 107/255.0, 1.0))

    # The correct shading for shadows are now applied. Use the face normals and light orientation to generate a shading value and apply to the RGB colors for each face.
    rgbNew = np.array([colourRGB*(newMin + newdiff*((shade-min)/diff)) for shade in ls.shade_normals(normalsarray, fraction=1.0)])

    # Apply color to face
    mesh.set_facecolor(rgbNew)

    ax.add_collection3d(mesh)
    
    # Iso surface for the upper_lim
    verts, faces, normals, values = measure.marching_cubes(tps_half, upper_lim,
	    mask=mask_half)
    mesh = Poly3DCollection(verts[faces], rasterized=rast)

    ls = LightSource(azdeg=225.0, altdeg=45.0)
    normalsarray = np.array([np.array((np.sum(normals[face[:], 0]/3),
 	np.sum(normals[face[:], 1]/3), np.sum(normals[face[:],
	    2]/3))/np.sqrt(np.sum(normals[face[:], 0]/3)**2 +
		np.sum(normals[face[:], 1]/3)**2 + np.sum(normals[face[:],
		    2]/3)**2)) for face in faces])

    # Next this is more asthetic, but it prevents the shadows of the image being too dark. (linear interpolation to correct)
    min = np.min(ls.shade_normals(normalsarray, fraction=1.0)) # min shade value
    max = np.max(ls.shade_normals(normalsarray, fraction=1.0)) # max shade value
    diff = max-min
    newMin = 0.3 # 0.3
    newMax = 0.95 # 0.95
    newdiff = newMax-newMin

    # Using a constant color, put in desired RGB values here.
    colourRGB = np.array((255.0/255.0, 54.0/255.0, 57/255.0, 1.0))

    # The correct shading for shadows are now applied. Use the face normals and light orientation to generate a shading value and apply to the RGB colors for each face.
    rgbNew = np.array([colourRGB*(newMin + newdiff*((shade-min)/diff)) for shade in ls.shade_normals(normalsarray, fraction=1.0)])

    # Apply color to face
    mesh.set_facecolor(rgbNew)

    ax.add_collection3d(mesh)

    # Set axes
    ax.set_xlim(0, 200) # 200 is the first dimension of tps_half (400 comes from B36_new, here /2 because we only considered the first 10 days out of 20)
    ax.set_ylim(0, 100) # 67 is the second dimension of tps_half (67 comes from B36_new)
    ax.set_zlim(0, DEPTH_IDX-1) # depth

    ax.set_xlabel(r"Days since TC passage ($\tau$)")
    ax.set_xticks([36, 91, 145, 199])
    ax.set_xticklabels([0, 3, 6, 9])
    ax.set_ylabel(r"Cross-track angle, degrees ($d$)")
    ax.set_yticks([19, 50, 80]) # ([0, 33.5, 67])
    ax.set_yticklabels([-5, 0, 5])
    ax.set_zlabel(r"Pressure, dbars ($z$)")
    ax.set_zticks([0, int((DEPTH_IDX-1)/4*1), int((DEPTH_IDX-1)/4*2), int((DEPTH_IDX-1)/4*3), DEPTH_IDX-1])
    ax.set_zticklabels([200, 150, 100, 50, 10])

    # Save plot
    plt.savefig(output_fname,
            bbox_inches='tight',
            pad_inches=0,
            dpi=300,
            )
    return
   
 
# Loop through all cases
for sub in sub_list:
    print(sub)
    
    # Load inputs
    tps_preds = pkl.load(open(f'{input_dir}/TPS{prefix}_Preds_Block_{lamb_choice}_{sub}.pkl', 'rb'))
    tps_masks = pkl.load(open(f'{input_dir}/TPS{prefix}_Mask_Block_{lamb_choice}_{sub}.pkl', 'rb'))

    tps_preds_mat = np.zeros((400, 100, DEPTH_IDX))
    mask_mat = np.zeros((400, 100, DEPTH_IDX))
    for depth_idx in range(DEPTH_IDX):
        tps_preds_mat[:, :, DEPTH_IDX-1-depth_idx] = tps_preds[:, depth_idx].reshape(400, 100)
        mask_mat[:, :, DEPTH_IDX-1-depth_idx] = tps_masks[:, depth_idx].reshape(400, 100)
    
        # savemat('tps_preds_mat_temperature_all.mat', {'tps_preds_mat': tps_preds_mat})
        # savemat('mask_mat_temperature_all.mat', {'mask_mat': mask_mat})
        # savemat('tps_preds_mat_temperature_incr.mat', {'tps_preds_mat': tps_preds_mat})
        # savemat('mask_mat_temperature_incr.mat', {'mask_mat': mask_mat})
        # savemat('tps_preds_mat_temperature_decr.mat', {'tps_preds_mat': tps_preds_mat})
        # savemat('mask_mat_temperature_decr.mat', {'mask_mat': mask_mat})
        # savemat('tps_preds_mat_salinity_all.mat', {'tps_preds_mat': tps_preds_mat})
        # savemat('mask_mat_salinity_all.mat', {'mask_mat': mask_mat})
        # savemat('tps_preds_mat_salinity_incr.mat', {'tps_preds_mat': tps_preds_mat})
        # savemat('mask_mat_salinity_incr.mat', {'mask_mat': mask_mat})
        # savemat('tps_preds_mat_salinity_decr.mat', {'tps_preds_mat': tps_preds_mat})
        # savemat('mask_mat_salinity_decr.mat', {'mask_mat': mask_mat})
        # savemat('tps_preds_mat_pot_density_all.mat', {'tps_preds_mat': tps_preds_mat})
        # savemat('mask_mat_pot_density_all.mat', {'mask_mat': mask_mat})
        # savemat('tps_preds_mat_pot_density_incr.mat', {'tps_preds_mat': tps_preds_mat})
        # savemat('mask_mat_pot_density_incr.mat', {'mask_mat': mask_mat})
        # savemat('tps_preds_mat_pot_density_decr.mat', {'tps_preds_mat': tps_preds_mat})
        # savemat('mask_mat_pot_density_decr.mat', {'mask_mat': mask_mat})
        
    mask_mat = mask_mat.astype(bool)
    
    # General plotting settings
    fs = 16
    df = 2
    plt.rcParams['font.family'] = 'Liberation Serif'
    plt.rcParams['mathtext.rm'] = 'serif'
    plt.rcParams['mathtext.it'] = 'serif:italic'
    plt.rcParams['mathtext.bf'] = 'serif:bold'
    plt.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams.update({'font.size': fs})

    # Choose angles from which the 3D plot is shown
    # First plot:
    init_elev1 = 21.0
    init_azim1 = 144.1
    
    # Second plot:
    init_elev2 = 25.052
    init_azim2 = 16.083
    
    # Third plot:
    init_elev3 = 18.557
    init_azim3 = 176.412
    
    plot_isosurface(
            init_elev1,
            init_azim1,
            f'{plot_dir}/TPS_Isosurface_{var2use}_{sub}_1.pdf',
            )
    
    plot_isosurface(
            init_elev2,
            init_azim2,
            f'{plot_dir}/TPS_Isosurface_{var2use}_{sub}_2.pdf',
            )
    
    plot_isosurface(
            init_elev3,
            init_azim3,
            f'{plot_dir}/TPS_Isosurface_{var2use}_{sub}_3.pdf',
            )
    plot_isosurface(
            init_elev1,
            init_azim1,
            f'{plot_dir}/TPS_Isosurface_{var2use}_{sub}_1.jpg',
            )
    
    plot_isosurface(
            init_elev2,
            init_azim2,
            f'{plot_dir}/TPS_Isosurface_{var2use}_{sub}_2.jpg',
            )
    
    plot_isosurface(
            init_elev3,
            init_azim3,
            f'{plot_dir}/TPS_Isosurface_{var2use}_{sub}_3.jpg',
            )
    