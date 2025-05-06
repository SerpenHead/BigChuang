# -*- coding: utf-8 -*-
# 这个脚本是win下的抽帧脚本，将data_mp4文件夹中的所有视频数据抽出20张图片来，并用（02），（03）似的命名后缀来区分同一视频抽帧出来的不同图像，规定每6帧抽一张，可自定义，运行该脚本之前要确认data_mp4的子目录中以地点名字命名的各个文件夹不含中文字符，为此，可以运行rename(from hanzi to pinyin).py 脚本

import os
import cv2
# 填入data_mp4文件夹的存储路劲
base_path = "F:\\bic"
source_path = os.join(base_path, "\data\data_mp4")

def get_dirs_list_and_save_dirs_list(source_path):
    dirs_list = os.listdir(source_path)
    for name in dirs_list:
        name = name + "\Done"
        full_path = source_path + "\\" + name
        if not os.path.exists(full_path):
            # 如果文件夹不存在，则创建新的文件夹
            os.makedirs(full_path)
            print("{}文件夹已创建".format(full_path))
    return dirs_list

dirs_list = get_dirs_list_and_save_dirs_list(source_path)
for dir_name in dirs_list:
    videos_src_path = source_path + "\\" + dir_name
    videos_save_path = source_path + "\\" + dir_name + "\Done"
    videos = os.listdir(videos_src_path)
    for each_video in videos:
        #获取每个视频的名称
        each_video_name = (each_video.split('.'))[0]
        #获取保存图片的完整路径，每个视频的图片帧存在以视频名为文件名的文件夹中
        each_video_save_full_path = videos_save_path + "\\" + each_video_name
        #每个视频的完整路径
        each_video_full_path = videos_src_path + "\\" + each_video
        #读入视频
        cap = cv2.VideoCapture(each_video_full_path)
        print(each_video_full_path)
        frame_count = 1
        count = 2
        success = True
        while (success):
            #提取视频帧，success为是否成功获取视频帧（true/false），第二个返回值为返回的视频帧
            success, frame = cap.read()
            # 如果想间隔比如25帧抽一张，可以在下方if里加条件：frame_count % 25 == 0
            if success == True and frame_count % 6 == 0:
                #存储视频帧,%04d则每张图片以2位数命名，比如01.jpg
                each_video_save_full_name = each_video_save_full_path + "(" + "%02d).jpg" % count
                cv2.imwrite(each_video_save_full_name, frame)
                count += 1
                print(each_video_save_full_name + " 已下载在 " + each_video_save_full_path)
            frame_count += 1

