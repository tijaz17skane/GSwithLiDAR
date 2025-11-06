# ======================================================================#
# Start off with FHF dataset in a folder
# ======================================================================#

# 1 # OPTIONAL: downsample the las file
python /mnt/data/tijaz/gaussian-splatting/OwnUtils/downsample_las.py --input /mnt/data/tijaz/data/betterConverters/section_3/annotated_ftth.las --output /mnt/data/tijaz/data/betterConverters/section_3/annotated_ftth_downsampled.las --factor 64 --seed 32

# 2 # load dataset. fhf to colmap format
python /mnt/data/tijaz/gaussian-splatting/OwnUtils/convert_fhf_to_colmap.py --meta /mnt/data/tijaz/data/betterConverters/section_3/meta.json --calib /mnt/data/tijaz/data/betterConverters/section_3/tabular/calibration.csv --las /mnt/data/tijaz/data/betterConverters/section_3/annotated_ftth_downsampled.las --outdir /mnt/data/tijaz/data/betterConverters/section_3/converted2Colmap --images_folder /mnt/data/tijaz/data/betterConverters/section_3/images

# 3 # OPTIONAL: View colmap model
python /mnt/data/tijaz/gaussian-splatting/OwnUtils/viewTXTcamsAsGPKGandPLY.py
# this will output gpkg and ply and txt files in world format for you view how images.txt look. So input images.txt and get a viewable file in whatever format you like

# ======================================================================#
# Run COLMAP on a set of images with PINHOLE in the convert.py function. OPENCV doesn't work well here.
# Use colmap_output/sparse/0/ folder as input to next steps
# ======================================================================#

# 4 # OPTIONAL: register lidar points and poses to colmap friendly scale for best splatting convergence
python /mnt/data/tijaz/gaussian-splatting/OwnUtils/SRTaligner.py --inputA --inputB --outpudir

# 5 # OPTIONAL: Do this if you did step 4
# align lidar to colmap before splatting to get better quality

# first get the SRT matrices by aligning poses
python /mnt/data/tijaz/gaussian-splatting/OwnUtils/SRTaligner.py --inputA --inputB --outputdir --input_format --save_params --verbose

# Apply the SRT to the points3D on points3D for setA. 
python /mnt/data/tijaz/gaussian-splatting/OwnUtils/apply_SRT_to_points3D_txt.py --input_txt --SRT_transf_matrix --output_txt

# ======================================================================#
# You should now have a folder with images.txt, cameras.txt and points3D.txt ready
# main folder has images folder and sparse/0/ with the txt files
# Make another folder in this directory called depth_images to store depth image output from Depth Anything v2
# ======================================================================#


# 6 # Calculate Depths for depth supervision.
# follow instructions under Depth Regularization mentioned here: https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#depth-regularization
# roughly do the following

python /mnt/data/tijaz/Depth-Anything-V2/metric_depth/run.py --encoder vitl --pred-only --grayscale --img-path <path to input images> --outdir <output path> --load-from /mnt/data/tijaz/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth
# we need colmap in bin format for depth scale calculation
colmap model_converter --input_path <path to colmap sparse model> --output_path <path to colmap sparse model> --output_type BIN
python utils/make_depth_scale.py --base_dir <path to colmap> --depths_dir <path to generated depths>

# 7 # Now you can run Gaussian Splatting training with depth supervision
# Something like the follwing
nohup python train.py -s /mnt/data/tijaz/data/section3ready -m /mnt/data/tijaz/trainingOutputs/sec3att0093im1sph0pc1 --iterations 30000 -d /mnt/data/tijaz/data/section3ready/depth_images > train93.log 2>&1 &




# ======================================================================#
# END OF WORKFLOW
# ======================================================================#


# --- ALTERNAIVE USAGE EXAMPLES BELOW ---


# in the folder put all the fhf files meta calib las and images
python /mnt/data/tijaz/gaussian-splatting/OwnUtils/downsample_las.py --input_dir /mnt/data/tijaz/data/Attempt3 --out_dir /mnt/data/tijaz/data/Attempt3
# delete the original only keep downloaded file
python convert_fhf_to_colmap.py --input_dir /mnt/data/tijaz/data/Attempt3 --output_dir /mnt/data/tijaz/data/Attempt3
# also put in colmapCompleteOutput containing images.txt and points3D.txt from colmap run on images alone
python datasetAligner.py --input_dir /mnt/data/tijaz/data/Attempt3
# now prepare for depth regularization, which runs depth anything and makes depth scale. 
python depth_reg_preps.py --input_dir /mnt/data/tijaz/data/Attempt3
# Now run gaussian splatting training with depth supervision


