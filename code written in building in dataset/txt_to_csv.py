import os
import glob
from PIL import Image
import csv


def txt_to_csv(path):
    txt_list = []
    label_path = os.path.join(path, 'Label')
    for file in glob.glob(label_path + '/*.txt'):
        image_name_ext = (os.path.basename(file))
        image_name = os.path.splitext(image_name_ext)[0]
        image_path = os.path.join(path, image_name + '.jpg')
        im = Image.open(image_path)
        width, height = im.size
        with open(file, 'r') as myfile:
            rows = myfile.read().split('\n')
            for row in rows:
                if row is not '':
                    class_label, x_min, y_min, x_max, y_max = row.split(' ', 4)
                    value = [image_name + '.jpg',
                             int(width),
                             int(height),
                             class_label,
                             int(float(x_min)), int(float(y_min)),
                             int(float(x_max)), int(float(y_max))]
                    txt_list.append(value)
    return txt_list



def main():
   for directory in ['train', 'test']:
       images_path = os.path.join('/home/eamonn/FYP/OpenImage/OIDv4_ToolKit/OID/Dataset/', '{}'.format(directory))
       for label in ['Dumbbell', 'Footwear']:
           image_path = os.path.join(images_path, label)
           csv_rows = txt_to_csv(image_path)
           csv_file = ('data2/{}_labels.csv'.format(directory))
           with open(csv_file, 'a') as fd:
               writer = csv.writer(fd)
               for row in csv_rows:
                   writer.writerow(row)


main()