import os 
import sys
import json
import cv2 as cv
import numpy as np
import collections as cl
import shutil
import math
import random
import scipy.io as sio
from tqdm import tqdm
'''
{ 'filename': '28503151_5b5b7ec140_b.jpg',
  'regions': {
      '0': {
          'region_attributes': {},
          'shape_attributes': {
              'all_points_x': [...],
              'all_points_y': [...],
              'name': 'polygon'}},
      ... more regions ...
  },
  'size': 100202
}
'''

def dumpJsonFile(file_name, regionx, regiony, category):
    ys = cl.OrderedDict()
    ys['all_points_x'] = regionx
    ys['all_points_y'] = regiony
    ys['name'] = 'polygon'
    ds = cl.OrderedDict()
    ds['shape_attributes'] = ys
    reg_attributes = {}
    reg_attributes['rgbd'] = str(category)
    ds['region_attributes'] = reg_attributes
    return ds


def project(img, object3d, Rt, intrinsics, color):
    min_x = min(object3d[:, 0])
    min_y = min(object3d[:, 1])
    min_z = min(object3d[:, 2])
    max_x = max(object3d[:, 0])
    max_y = max(object3d[:, 1])
    max_z = max(object3d[:, 2])
    bb3d = np.zeros((8, 4), dtype=float)
    ##0-1, 0-2, 0-3, 7-6, 7-5, 7-4, 1-3, 1-6, 2-4, 2-6, 3-4, 3-5
    bb3d[0, :] = [min_x, min_y, min_z, 1.]
    bb3d[1, :] = [min_x, min_y, max_z, 1.]
    bb3d[2, :] = [min_x, max_y, min_z, 1.]
    bb3d[3, :] = [max_x, min_y, min_z, 1.]
    bb3d[4, :] = [max_x, max_y, min_z, 1.]
    bb3d[5, :] = [max_x, min_y, max_z, 1.]
    bb3d[6, :] = [min_x, max_y, max_z, 1.]
    bb3d[7, :] = [max_x, max_y, max_z, 1.]
    cam2d = np.dot(intrinsics, np.dot(Rt, bb3d.transpose()))
    cam2d[0, :] = cam2d[0, :] / cam2d[2, :]
    cam2d[1, :] = cam2d[1, :] / cam2d[2, :]
    cam2d = cam2d.astype(np.int32)
    #cam2d = np.dot(intrinsics, np.dot(Rt, np.hstack((object3d, np.ones((object3d.shape[0], 1), dtype=float))).transpose()))
    cv.line(img, (cam2d[0, 0], cam2d[1, 0]), (cam2d[0, 1], cam2d[1, 1]), (255, 0, 0))
    cv.line(img, (cam2d[0, 0], cam2d[1, 0]), (cam2d[0, 2], cam2d[1, 2]), (255, 0, 0))
    cv.line(img, (cam2d[0, 0], cam2d[1, 0]), (cam2d[0, 3], cam2d[1, 3]), (255, 0, 0))
    cv.line(img, (cam2d[0, 7], cam2d[1, 7]), (cam2d[0, 6], cam2d[1, 6]), (255, 0, 0))
    cv.line(img, (cam2d[0, 7], cam2d[1, 7]), (cam2d[0, 5], cam2d[1, 5]), (255, 0, 0))
    cv.line(img, (cam2d[0, 7], cam2d[1, 7]), (cam2d[0, 4], cam2d[1, 4]), (255, 0, 0))
    cv.line(img, (cam2d[0, 1], cam2d[1, 1]), (cam2d[0, 5], cam2d[1, 5]), (255, 0, 0))
    cv.line(img, (cam2d[0, 1], cam2d[1, 1]), (cam2d[0, 6], cam2d[1, 6]), (255, 0, 0))
    cv.line(img, (cam2d[0, 2], cam2d[1, 2]), (cam2d[0, 4], cam2d[1, 4]), (255, 0, 0))
    cv.line(img, (cam2d[0, 2], cam2d[1, 2]), (cam2d[0, 6], cam2d[1, 6]), (255, 0, 0))
    cv.line(img, (cam2d[0, 3], cam2d[1, 3]), (cam2d[0, 4], cam2d[1, 4]), (255, 0, 0))
    cv.line(img, (cam2d[0, 3], cam2d[1, 3]), (cam2d[0, 5], cam2d[1, 5]), (255, 0, 0))

    cam2d_obj = np.dot(intrinsics, np.dot(Rt, np.hstack((object3d, np.ones((object3d.shape[0], 1), dtype=float))).transpose()))
    cam2d_obj[0, :] = cam2d_obj[0, :] / cam2d_obj[2, :]
    cam2d_obj[1, :] = cam2d_obj[1, :] / cam2d_obj[2, :]
    cam2d_obj = cam2d_obj.astype(np.int32)
    #for i in range(cam2d_obj.shape[1]):
    #    cv.circle(img, (cam2d_obj[0, i], cam2d_obj[1, i]), 2, color)

if __name__ == '__main__':
    output_dir = './dataset/'
    if os.path.isdir(output_dir + 'train'):
        shutil.rmtree(output_dir + 'train')
    os.mkdir(output_dir + 'train')
    if os.path.isdir(output_dir + 'val'):
        shutil.rmtree(output_dir + 'val')
    os.mkdir(output_dir + 'val')

    root_dir = 'E:/Dataset/YCB_Video_Dataset/data'
    object_dir = 'E:/Dataset/YCB_Video_Dataset/models'
    scene_list = os.listdir(root_dir)

    object_list = np.genfromtxt('E:/Dataset/YCB_Video_Dataset/code/obj_list.txt', dtype=str, delimiter=',')

    objects_pointcloud = []
    for obj in object_list:
        obj_path = os.path.join(object_dir, obj, 'points.xyz')
        objects = np.loadtxt(obj_path, delimiter=' ', dtype=float)
        objects_pointcloud.append(objects)

    colors = []
    for i in range(100):
        colors.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

    save_trDict = cl.OrderedDict()
    save_valDict = cl.OrderedDict()

    for scene_idx, scene in enumerate(scene_list):
        print('{} / {}'.format(scene_idx, len(scene_list)))
        instance_list = os.listdir(os.path.join(root_dir, scene))
        frame_num = int(len(instance_list) / 5)

        for fdx in tqdm(range(1, frame_num + 1)):
            if np.random.rand() < 0.8:
                mode = 'train'
            else:
                mode = 'test'
            fdx_str = str(fdx).zfill(6)
            color_path = os.path.join(root_dir, scene, fdx_str + '-color.png')
            box_path = os.path.join(root_dir, scene, fdx_str + '-box.txt')
            label_path = os.path.join(root_dir, scene, fdx_str + '-label.png')
            meta_path = os.path.join(root_dir, scene, fdx_str + '-meta.mat')
            meta_info = sio.loadmat(meta_path)
            classes = np.squeeze(meta_info['cls_indexes'])
            poses = meta_info['poses']
            intrinsics = meta_info['intrinsic_matrix']


            img = cv.imread(color_path)
            img_save = img.copy()
            dst = img.copy()
            label_img = cv.imread(label_path)

            bbox_info = np.genfromtxt(box_path, dtype=str, delimiter=' ')
            ts = cl.OrderedDict()
            size = 0
            for bdx in range(bbox_info.shape[0]):
                obj_id = object_list.tolist().index(bbox_info[bdx, 0])
                #project(img, objects_pointcloud[obj_id], poses[:, :, classes.tolist().index(obj_id + 1)], intrinsics, colors[obj_id])
                x1 = int(float(bbox_info[bdx, 1]))
                y1 = int(float(bbox_info[bdx, 2]))
                x2 = int(float(bbox_info[bdx, 3]))
                y2 = int(float(bbox_info[bdx, 4]))
                cv.rectangle(img, (x1, y1), (x2, y2), colors[obj_id], 3)

                mask_img = np.zeros(label_img.shape[:-1], dtype=np.uint8)
                mask_img[label_img[:, :, 0] == obj_id + 1] = 255
                _, contours, hierarchy = cv.findContours(mask_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                if len(contours) == 0:
                    continue
                contours = contours[0]
                regionx = []
                regiony = []
                contour_p = len(contours)
                for i in range(len(contours)):
                    rx = contours[i][0][0]
                    ry = contours[i][0][1]
                    regionx.append(int(rx))
                    regiony.append(int(ry))
                    cv.circle(img, (rx, ry), 1, colors[obj_id])
                
                rotation_vec, _ = cv.Rodrigues(poses[:3, :3, classes.tolist().index(obj_id + 1)])
                translation_vec = poses[:3, 3, classes.tolist().index(obj_id + 1)]
                #print(rotation_vec)
                #print(translation_vec)
                ys = cl.OrderedDict()
                ys['all_points_x'] = regionx
                ys['all_points_y'] = regiony
                ys['name'] = 'polygon'
                ds = cl.OrderedDict()
                ds['shape_attributes'] = ys
                reg_attributes = {}
                reg_attributes['ycb'] = str(obj_id + 1)
                ds['region_attributes'] = reg_attributes
                pose_attributes = {}
                pose_attributes['rx'] = rotation_vec[0][0]
                pose_attributes['ry'] = rotation_vec[1][0]
                pose_attributes['rz'] = rotation_vec[2][0]
                pose_attributes['tx'] = translation_vec[0]
                pose_attributes['ty'] = translation_vec[1]
                pose_attributes['tz'] = translation_vec[2]
                ds['pose_attributes'] = pose_attributes       
                ts[str(bdx)] = ds
                size += len(regionx)
            zs = cl.OrderedDict()
            zs['fileref'] = ''
            zs['base64_img_data'] = ''
            zs['filename'] = scene + '_' + fdx_str
            zs['regions'] = ts
            zs['size'] = size
            if mode == 'train':
                save_trDict[os.path.join(root_dir, scene + '_' +  fdx_str) + str(size)] = zs
                output_file = output_dir + 'train/' + scene + '_' +  fdx_str + '.jpg'
            else:
                save_valDict[os.path.join(root_dir, scene + '_' + fdx_str) + str(size)] = zs
                output_file = output_dir + 'val/' + scene + '_' +  fdx_str + '.jpg'
            #cv.imshow(object_list[obj_id], mask_img)
            cv.imwrite(output_file, img_save)
            cv.imshow("", img)
            cv.waitKey(1)
    fw = open('./dataset/train/via_region_data.json', 'w')
    fw.write(json.dumps(save_trDict))
    fw.close()
    fw1 = open('./dataset/val/via_region_data.json', 'w')
    fw1.write(json.dumps(save_valDict))
    fw1.close()