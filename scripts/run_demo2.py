# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *
import h5py
import subprocess
import shutil
import cv2


def resize_image(img_chw, target_h, target_w, interpolation=cv2.INTER_LINEAR):
    # img_chw: C x H x W numpy array    
    img_hwc = np.transpose(img_chw, (1, 2, 0))
    resized_hwc = cv2.resize(img_hwc, (target_w, target_h), interpolation=interpolation)
    resized_chw = np.transpose(resized_hwc, (2, 0, 1))
    
    return resized_chw

def resize_batch(batch_nchw, target_h, target_w, interpolation=cv2.INTER_LINEAR):
    return np.stack([resize_image(img, target_h, target_w, interpolation) for img in batch_nchw])



if __name__=="__main__":
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', default=5, type=int)
  parser.add_argument('--left_file', default=f'{code_dir}/../assets/left.png', type=str)
  parser.add_argument('--right_file', default=f'{code_dir}/../assets/right.png', type=str)
  parser.add_argument('--left_h5_file', default="", type=str)
  parser.add_argument('--right_h5_file', default="", type=str)
  parser.add_argument('--stereo_params_npz_file', default = "", type = str)
  parser.add_argument('--intrinsic_file', default=f'{code_dir}/../assets/K.txt', type=str, help='camera intrinsic matrix and baseline file')
  parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth', type=str, help='pretrained model path')
  parser.add_argument('--out_dir', default=f'{code_dir}/../output/', type=str, help='the directory to save results')
  parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
  parser.add_argument('--hiera', default=1, type=int, help='hierarchical inference (only needed for high-resolution images (>1K))')
  parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
  parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
  parser.add_argument('--get_depth', type=int, default=1, help='save depth map output as numpy array in meters')
  parser.add_argument('--get_pc', type=int, default=0, help='save point cloud output')  
  parser.add_argument('--remove_invisible', default=1, type=int, help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')
  parser.add_argument('--denoise_cloud', type=int, default=1, help='whether to denoise the point cloud')
  parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
  parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
  parser.add_argument("--process_only",default=None,type=int)
  args = parser.parse_args()

  if not os.path.exists(f'{code_dir}/../dinov2_gh'):
    subprocess.run(["git", "clone", "https://github.com/facebookresearch/dinov2.git", f'{code_dir}/../dinov2_gh'], check=True)
    if os.path.exists(f'{code_dir}/../dinov2_gh/dinov2'):
      if os.path.exists(f'{code_dir}/../dinov2'):      
        shutil.rmtree(f'{code_dir}/../dinov2')
      shutil.move(f'{code_dir}/../dinov2_gh/dinov2', f'{code_dir}/../')



  set_logging_format()
  set_seed(0)
  torch.autograd.set_grad_enabled(False)
  os.makedirs(args.out_dir, exist_ok=True)

  ckpt_dir = args.ckpt_dir
  cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
  if 'vit_size' not in cfg:
    cfg['vit_size'] = 'vitl'
  for k in args.__dict__:
    cfg[k] = args.__dict__[k]
  args = OmegaConf.create(cfg)
  logging.info(f"args:\n{args}")
  logging.info(f"Using pretrained model from {ckpt_dir}")
  stereo_params = np.load(args.stereo_params_npz_file, allow_pickle=True)
        
  P1 = stereo_params['P1']
  P1[:2] *= args.scale
  f_left = P1[0,0]
  baseline = stereo_params['baseline']
  if args.left_h5_file and args.right_h5_file:
    try:
      with h5py.File(args.left_h5_file, 'r') as f:
        left_all = f['data'][()]   # or np.array(f['left'])
      with h5py.File(args.right_h5_file, 'r') as f:
        right_all = f['data'][()]
    except Exception as e:            
      with h5py.File(args.left_h5_file, 'r') as f:
        left_all = f['left'][()]   # or np.array(f['left'])
      with h5py.File(args.right_h5_file, 'r') as f:
        right_all = f['right'][()]
      
    print(left_all.shape, right_all.shape)

    if left_all.ndim==3:
      left_all = left_all[None]
      right_all = right_all[None]
  N,C,H,W = left_all.shape
  if args.process_only:
    N_stop = args.process_only
  else:
    N_stop = N
  resize_factor = 1.5
  print(max(np.ceil(W/resize_factor/4).astype(int), cfg["max_disp"]))
  cfg["max_disp"] = int(np.ceil(W/resize_factor/4).astype(int), cfg["max_disp"])
  args.max_disp = cfg["max_disp"]

  model = FoundationStereo(args)

  ckpt = torch.load(ckpt_dir)
  logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
  model.load_state_dict(ckpt['model'])

  model.cuda()
  model.eval()

  disp_all = []
  depth_all = []

  for i in range(0, N, args.batch_size):
    img0 = left_all[i:i+args.batch_size]
    img1 = right_all[i:i+args.batch_size]
    img0_ori = img0.copy()

    if len(img0.shape)==3:
      img0 = img0[None,...]

    if len(img1.shape)==3:
      img1 = img1[None,...]

    # image size of about 1500x2300 works with batch_size of 1, 
    # with resize_factor of 1.5 at 28s/image, up to ~25 images.

    img0 = resize_batch(img0, round(H/resize_factor) ,round(W/resize_factor))
    img1 = resize_batch(img1, round(H/resize_factor), round(W/resize_factor))
    logging.info(f"batch {i}, img: {img0.shape}")  
    img0 = torch.as_tensor(img0).cuda().float()
    img1 = torch.as_tensor(img1).cuda().float()

    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)

    with torch.amp.autocast("cuda",enabled=True):
      if not args.hiera:
        disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
      else:
        disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
  
      print(disp.shape)
      disp = padder.unpad(disp.float())
      print(disp.shape)

      print("\n")
      # disparr = disp.data.cpu().numpy().reshape(disp.shape[0],H,W)
      # vis = vis_disparityarr(disparr)
      # # vis = np.concatenate([img0_ori, vis], axis=1)
      # imageio.imwrite(f'{args.out_dir}/vis.png', vis)
      # logging.info(f"Output saved to {args.out_dir}")

      # if args.remove_invisible:
      #   yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
      #   us_right = xx-disp
      #   invalid = us_right<0
      #   disp[invalid] = np.inf

      if args.get_depth:
        depth = f_left*baseline/(disp+1e-6)
        #np.save(f'{args.out_dir}/leftview_depth_meter.npy', depth)
      disp_all.append(disp.data.cpu().numpy())
      depth_all.append(depth.data.cpu().numpy()) 
      if i+batch_size >= N_stop:
        break

  disp_all = np.concatenate(disp_all, axis=0).reshape(N,round(H/resize_factor),round(W/resize_factor)).astype(np.float16)
  depth_all = np.concatenate(depth_all, axis=0).reshape(N,round(H/resize_factor),round(W/resize_factor)).astype(np.float16)

  with h5py.File(f'{args.out_dir}/leftview_disp_depth.h5', 'w') as f:
    f.create_dataset('disp', data=disp_all, compression='gzip')
    f.create_dataset('depth', data=depth_all, compression='gzip')

  
  if args.get_pc:
    with open(args.intrinsic_file, 'r') as f:
      lines = f.readlines()
      K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
      baseline = float(lines[1])
    K[:2] *= scale
    depth = K[0,0]*baseline/disp_all
    np.save(f'{args.out_dir}/depth_meter.npy', depth)
    xyz_map = depth2xyzmap(depth_all, K)
    pcd = toOpen3dCloud(xyz_map.reshape(-1,3), img0_ori.reshape(-1,3))
    keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=args.z_far)
    keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
    pcd = pcd.select_by_index(keep_ids)
    o3d.io.write_point_cloud(f'{args.out_dir}/cloud.ply', pcd)
    logging.info(f"PCL saved to {args.out_dir}")

    if args.denoise_cloud:
      logging.info("[Optional step] denoise point cloud...")
      cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
      inlier_cloud = pcd.select_by_index(ind)
      o3d.io.write_point_cloud(f'{args.out_dir}/cloud_denoise.ply', inlier_cloud)
      pcd = inlier_cloud

    logging.info("Visualizing point cloud. Press ESC to exit.")
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
    vis.run()
    vis.destroy_window()

