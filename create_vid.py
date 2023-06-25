import cv2
import os
from game import WIDTH, LENGTH

out_path = 'C:/Users/Michael/Desktop/CGOL/screen'

def main():
    for folder in os.listdir(out_path):
        dir = os.path.join(out_path, folder)
        img_arr = []
        for image in os.listdir(dir):
            img = cv2.imread(os.path.join(dir, image))
            height, width, layers = img.shape
            size = (width,height)
            img_arr.append(img)

        # video = cv2.VideoWriter(f'{folder}.mp4', 0, 1, (LENGTH, WIDTH))
        out = cv2.VideoWriter(f'{dir}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
            
        for i in range(len(img_arr)):
            out.write(img_arr[i])
        out.release()

if __name__ =='__main__':
    main()