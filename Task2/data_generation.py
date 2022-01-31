import cv2
import os
import csv
import random
import shutil

img_path = 'image/'
label_path = 'label/'
gen_img_path = 'gen_data/images/'
gen_label_path = 'gen_data/labels/'
ori_img_path = 'ori_data/images/'
ori_label_path = 'ori_data/labels/'


def get_photos():
    for root, dirs, files in os.walk(gen_img_path):
        continue
    return files


def read_ori_label(photo_name):
    with open(label_path + photo_name.split('.')[0] + '.txt', "r") as f:
        data = f.read()
        f.close()
    data = data.split(' ')
    return [int(float(data[1])), int(float(data[2])), int(float(data[3])), int(float(data[4]))]


def read_gen_label(photo_name):
    with open(gen_label_path + photo_name.split('.')[0] + '.txt', "r") as f:
        data = f.read()
        f.close()
    data = data.split(' ')
    return [float(data[1]), float(data[2]), float(data[3]), float(data[4])]


def write_label(photo_name, operation, box, ori=False):
    box = [0] + [i for i in box]
    box = str(box).replace(',', '')[1:-1]
    if ori:
        f = open(ori_label_path + operation + photo_name.split('.')[0] + '.txt', 'w', encoding='utf-8')
    else:
        f = open(gen_label_path + operation + photo_name.split('.')[0] + '.txt', 'w', encoding='utf-8')
    f.write(box)
    f.close()


def show_img(photo_path, box, ori=False):
    if ori:
        img = cv2.imread(photo_path)
        cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 225), 1)
        cv2.imshow(photo_path.split('/')[-1], img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cx = int(float(box[0]) * 224)
        cy = int(float(box[1]) * 224)
        w = int(float(box[2]) * 224)
        h = int(float(box[3]) * 224)
        img = cv2.imread(photo_path)
        cv2.rectangle(img, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)), (0, 0, 225), 1)
        cv2.imshow(photo_path.split('/')[-1], img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def delete_data():
    list = [gen_img_path, gen_label_path, ori_img_path, ori_label_path]
    for path in list:
        del_list = os.listdir(path)
        for f in del_list:
            file_path = os.path.join(path, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)


def txt_file():
    img_list = os.listdir(gen_img_path)
    random.shuffle(img_list)
    train_list = []
    valid_list = []
    print(len(img_list))
    for i in range(len(img_list)):
        if i % 7 == 0:
            valid_list.append(img_list[i])
        else:
            train_list.append(img_list[i])
    print(len(train_list), len(valid_list))
    with open("gen_data/train.txt", "w") as f:
        for file in train_list:
            f.write('../../gen_data/images/' + file + '\n')
    with open("gen_data/valid.txt", "w") as f:
        for file in valid_list:
            f.write('../../gen_data/images/' + file + '\n')

    img_list = os.listdir(ori_img_path)
    random.shuffle(img_list)
    train_list = []
    valid_list = []
    print(len(img_list))
    for i in range(len(img_list)):
        if i % 7 == 0:
            valid_list.append(img_list[i])
        else:
            train_list.append(img_list[i])
    print(len(train_list), len(valid_list))
    with open("ori_data/train.txt", "w") as f:
        for file in train_list:
            f.write('../../ori_data/images/' + file + '\n')
    with open("ori_data/valid.txt", "w") as f:
        for file in valid_list:
            f.write('../../ori_data/images/' + file + '\n')


class Data_Generator(object):
    def __init__(self):
        self.photo_list = []
        for root, dirs, files in os.walk(gen_img_path):
            self.photo_list = files

    def resize(self):
        ori_photo_list = []
        for root, dirs, files in os.walk(img_path):
            ori_photo_list = files
        for i in range(len(ori_photo_list)):
            print(i, len(ori_photo_list))
            if ori_photo_list[i].split('.')[-1] != 'jpg':
                continue
            ori_img = cv2.imread(img_path + ori_photo_list[i])
            ori_label = read_ori_label(ori_photo_list[i])
            resize_img = cv2.resize(ori_img, (224, 224))
            resize_w = ori_label[2] / ori_img.shape[1]
            resize_h = ori_label[3] / ori_img.shape[0]
            resize_cx = ori_label[0] / ori_img.shape[1] + resize_w / 2
            resize_cy = ori_label[1] / ori_img.shape[0] + resize_h / 2
            resize_label = [resize_cx, resize_cy, resize_w, resize_h]
            cv2.imwrite(gen_img_path + 'resize' + str(224) + str(0) + ori_photo_list[i], resize_img)
            write_label(ori_photo_list[i], 'resize' + str(224) + str(0), resize_label)
            resize_list = [256]
            for size in resize_list:
                resize_img = cv2.resize(ori_img, (size, size))
                resize_w = ori_label[2] * (size / ori_img.shape[1])
                resize_h = ori_label[3] * (size / ori_img.shape[0])
                resize_cx = ori_label[0] * (size / ori_img.shape[1]) + resize_w / 2
                resize_cy = ori_label[1] * (size / ori_img.shape[0]) + resize_h / 2
                for ii in range(3):
                    y = random.randint(224, size)
                    x = random.randint(224, size)
                    final_img = resize_img[y - 224:y, x - 224:x]
                    final_w = resize_w / 224
                    final_h = resize_h / 224
                    final_cx = (resize_cx - (x - 224)) / 224
                    final_cy = (resize_cy - (y - 224)) / 224
                    final_label = [final_cx, final_cy, final_w, final_h]
                    for label in range(len(final_label)):
                        if final_label[label] < 0:
                            final_label[label] = 0
                    cv2.imwrite(gen_img_path + 'resize' + str(size) + str(ii) + ori_photo_list[i], final_img)
                    write_label(ori_photo_list[i], 'resize' + str(size) + str(ii), final_label)

    def flip(self):
        photo_name_list = get_photos()
        for i in range(len(photo_name_list)):
            if photo_name_list[i].split('.')[-1] != 'jpg':
                continue
            print(i, len(photo_name_list))
            old_img = cv2.imread(gen_img_path + photo_name_list[i])
            flip_img = cv2.flip(old_img, 1)
            cv2.imwrite(gen_img_path + 'flip_' + photo_name_list[i], flip_img)
            old_label = read_gen_label(photo_name_list[i])
            flip_label = [1 - old_label[0], old_label[1],
                          old_label[2], old_label[3]]
            write_label(photo_name_list[i], 'flip_', flip_label)
        show_img(gen_img_path + photo_name_list[0], read_gen_label(photo_name_list[0]))
        show_img(gen_img_path + 'flip_' + photo_name_list[0], read_gen_label('flip_' + photo_name_list[0]))

    def gerate_ori_data(self):
        ori_photo_list = []
        for root, dirs, files in os.walk(img_path):
            ori_photo_list = files
        for i in range(len(ori_photo_list)):
            print(i, len(ori_photo_list))
            if ori_photo_list[i].split('.')[-1] != 'jpg':
                continue
            ori_img = cv2.imread(img_path + ori_photo_list[i])
            ori_label = read_ori_label(ori_photo_list[i])
            resize_img = cv2.resize(ori_img, (224, 224))
            resize_w = ori_label[2] / ori_img.shape[1]
            resize_h = ori_label[3] / ori_img.shape[0]
            resize_cx = ori_label[0] / ori_img.shape[1] + resize_w / 2
            resize_cy = ori_label[1] / ori_img.shape[0] + resize_h / 2
            resize_label = [resize_cx, resize_cy, resize_w, resize_h]
            cv2.imwrite(ori_img_path + ori_photo_list[i], resize_img)
            write_label(ori_photo_list[i], '', resize_label, ori=True)


if __name__ == '__main__':
    delete_data()
    # data_generator = Data_Generator()
    # data_generator.resize()
    # data_generator.flip()
    # data_generator.gerate_ori_data()
    # txt_file()
