import matplotlib.pyplot as plt
from PIL import Image

def anomaly_photo_func(glob, top_farthest_points_idx):
    # выведение топа самых вероятно аномальных изображений
    for i in range(len(top_farthest_points_idx)):
        file_path=glob['image_paths'][top_farthest_points_idx[i]]
        # if 'Неправильный_тип' in glob['image_paths'][top_farthest_points_idx[i]].split('/')[-1]:
        
        print(str('№')+'_'+str(i+1),' ' ,glob['image_paths'][top_farthest_points_idx[i]].split('/')[-1])

        img = Image.open(file_path).convert("RGB")

        plt.imshow(img)
        plt.axis('off') 
        plt.show()
