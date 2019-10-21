from glob import glob
import cv2, os

class mx_Class_Convert_Video_To_Image:
    def __init__(self, dataset_dir, dataset_name, image_save_folder_dir, image_save_folder_name):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.image_save_path = os.path.join(image_save_folder_dir, image_save_folder_name)

    def mx_Convert(self):
        filepath = os.path.join(self.dataset_dir, self.dataset_name)
        namelist = os.listdir(filepath)
        for FileName in namelist:
            mx_VideoName = os.path.join(filepath, FileName)
            mx_Capture = cv2.VideoCapture(mx_VideoName)

            FolderName = FileName.split('.')
            image_folder_path = os.path.join(self.image_save_path, FolderName[0])
            if not os.path.exists(image_folder_path):
                os.makedirs(image_folder_path)


            mx_ImgIdex = 1
            while 1:
                mx_ret, mx_ImgOri = mx_Capture.read()
                if mx_ret:
                    if mx_ImgIdex < 10:
                        m4_ImgName = 'image_' + '0000' + str(mx_ImgIdex) + '.jpeg'
                    elif mx_ImgIdex < 100:
                        m4_ImgName = 'image_' + '000' + str(mx_ImgIdex) + '.jpeg'
                    elif mx_ImgIdex < 1000:
                        m4_ImgName = 'image_' + '00' + str(mx_ImgIdex) + '.jpeg'
                    elif mx_ImgIdex < 10000:
                        m4_ImgName = 'image_' + '0' + str(mx_ImgIdex) + '.jpeg'
                    else:
                        m4_ImgName = 'image_' + str(mx_ImgIdex) + '.jpeg'
                    m4_ImgNamePath = os.path.join(image_folder_path ,m4_ImgName)
                    cv2.imwrite(m4_ImgNamePath, mx_ImgOri)
                    print('[* Save ', "'", m4_ImgNamePath, "'", ' successful! *]')
                    mx_ImgIdex += 1

                else:
                    mx_Capture.release()
                    break




if __name__ == '__main__':
    dataset_dir = '/media/yang/F/DataSet/action_detection'
    dataset_name = 'validation'
    image_save_folder_dir = '/media/yang/F/DataSet/action_detection'
    image_save_folder_name = 'SRC_FOLDER'
    mx_ConvertVideoToImage = mx_Class_Convert_Video_To_Image(dataset_dir, dataset_name, image_save_folder_dir, image_save_folder_name)
    mx_ConvertVideoToImage.mx_Convert()