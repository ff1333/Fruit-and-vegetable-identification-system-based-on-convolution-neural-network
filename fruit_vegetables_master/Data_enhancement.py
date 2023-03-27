import cv2
import math
import numpy as np
import os
import pdb
import xml.etree.ElementTree as ET
class ImgAugemention():
    def __init__(self):
        self.angle = 90
    # 旋转
    def rotate_image(self, src, angle, scale=1.):
        w = src.shape[1]
        h = src.shape[0]
        # 将角度转换为弧度
        rangle = np.deg2rad(angle)  # 转为弧度角
        # 计算新图像的宽度和高度
        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
        # 向OpenCV查询旋转矩阵
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
        # 计算从旧中心到新中心的移动
        # 旋转
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5, 0]))
        # 更新
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 查找
        return cv2.warpAffine(
            src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))),
            flags=cv2.INTER_LANCZOS4)
    def rotate_xml(self, src, xmin, ymin, xmax, ymax, angle, scale=1.):
        w = src.shape[1]
        h = src.shape[0]
        rangle = np.deg2rad(angle)  # 转为弧度角
        # 计算新图像的宽度和高度
        # 更改图像的宽度和高度
        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
        # 向OpenCV查询旋转矩阵
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
        # 计算从旧中心到新中心的移动
        # 旋转
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5, 0]))
        # 更新
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 在初始martix中获取边的四个中心，并转换坐标
        point1 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymin, 1]))
        point2 = np.dot(rot_mat, np.array([xmax, (ymin+ymax)/2, 1]))
        point3 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymax, 1]))
        point4 = np.dot(rot_mat, np.array([xmin, (ymin+ymax)/2, 1]))
        concat = np.vstack((point1, point2, point3, point4))
        # 变换类型
        concat = concat.astype(np.int32)
        #print(concat)
        rx, ry, rw, rh = cv2.boundingRect(concat)
        return rx, ry, rw, rh
    def process_img(self, imgs_path,  img_save_path,  angle_list):
        # 指定旋转角度
        for angle in angle_list:
            for img_name in os.listdir(imgs_path):
                # 拆分文件名和后缀
                n, s = os.path.splitext(img_name)
                if s == ".jpeg":
                    img_path = os.path.join(imgs_path, img_name)
                    img = cv2.imread(img_path)
                    rotated_img = self.rotate_image(img, angle)
                    save_name = n + "_" + str(angle) + "d.jpeg"
                    # 写入图像
                    cv2.imwrite(img_save_path + save_name, rotated_img)

if __name__ == '__main__':
    img_aug = ImgAugemention()
    imgs_path = r"C:\studysoftware\Anaconda3\pyqt5\pyqt\distinguish\fruit_vegetables_master\data\12" #获取原图路径
    img_save_path = r"C:\studysoftware\Anaconda3\pyqt5\pyqt\distinguish\fruit_vegetables_master\new_data\12\1" #将变换后的文件存入
    angle_list = [90, 180, 270] #变换的角度为90，180，270
    img_aug.process_img(imgs_path,  img_save_path,  angle_list)