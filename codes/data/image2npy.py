import os
import argparse
import skimage.io as sio
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='Pre-processing images to npy')
parser.add_argument('--pathRoot', default='',
                    help='directory of images to convert')
parser.add_argument('--pathTo', default='',
                    help='directory of images to save')
parser.add_argument('--split', default=True,
                    help='save individual images')
parser.add_argument('--select', default='',
                    help='select certain path')

args = parser.parse_args()

def makedir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def image2npy(path_HR_From, path_LR_From, path_HR_To, path_LR_To, split=True, select=''):
  for (path, dirs, files) in os.walk(path_LR_From):
      print(path)
      #targetDir = os.path.join(path_LR_To, path[len(path_LR_From) + 1:])
      if len(select) > 0 and path.find(select) == -1:
          continue
      if len(dirs) == 0:
          pack = {}
          n = 0
          for fileName in files:
              (idx, ext) = os.path.splitext(fileName)
              if ext == '.png' or ext == '.JPG':
                  image = sio.imread(os.path.join(path, fileName))
                  hr_image = sio.imread(os.path.join(path_HR_From, fileName))
                  h, w = image.shape[:2]
                  if h < 384 or w < 384:
                      continue
                  if h % 2 != 0:
                      h = h - 1
                  if w % 2 != 0:
                      w = w - 1
                  image = image[0: h, 0: w, :]
                  hr_image = hr_image[0: 2 * h, 0: 2 * w, :]
                  h = int(h / 2)
                  w = int(w / 2)
                  image = cv2.resize(image, (w, h), cv2.INTER_AREA)
                  hr_image = cv2.resize(hr_image, (w * 2, h * 2), cv2.INTER_AREA)
                  if split:
                      np.save(os.path.join(path_LR_To, idx + '.npy'), image)
                      np.save(os.path.join(path_HR_To, idx + '.npy'), hr_image)
                  n += 1
                  if n % 100 == 0:
                      print('Converted ' + str(n) + ' images.')


if __name__ == '__main__':
    
    hr_path = args.pathRoot + '/HR/'
    lr_path = args.pathRoot + '/LR/'
    #print(hr_path)
    hr_npy_path = args.pathTo + '/DC_train_decoded/HR/'
    lr_npy_path = args.pathTo + '/DC_train_decoded/LR/'
    makedir(hr_npy_path)
    makedir(lr_npy_path)
    print('Converted images.')
    image2npy(hr_path, lr_path, hr_npy_path, lr_npy_path, args.split, args.select)
    
