import face_alignment
from PIL import Image
import numpy as np


class WearFaceMask(object):
    def __init__(
        self,
        face_path,
        mask_path,
        save_path,
        enlarge_ratio=0.9,
        use_gpu=True,
        show=False,
    ):
        self.face_path = face_path
        self.mask_path = mask_path
        self.save_path = save_path
        self.enlarge_ratio = enlarge_ratio
        self.use_gpu = use_gpu
        self.show = show
        if use_gpu:
            self.fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D, device="cuda"
            )
        else:
            self.fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D, device="cpu"
            )

    def get_key_landmarks(self, face_landmarks):
        """从68个关键点中获取4个关键点的位置
        用来定口罩佩戴的位置
        :param face_landmarks:人脸68个关键点
        :return:
        """
        # 获取下巴右边佩戴口罩的位置(关键点2)
        self.left_chin_point = face_landmarks[1]
        # 获取鼻梁佩戴口罩的位置(关键点28)
        self.nose_point = face_landmarks[27]
        # 获取下巴左边佩戴口罩的位置(关键点16)
        self.right_chin_point = face_landmarks[15]
        # 获取下巴最下面佩戴口罩的位置(关键点9)
        self.bottom_chin_point = face_landmarks[8]

    @staticmethod
    def cal_point_to_line_dist(point, line_point1, line_point2):
        """计算点到直线的距离
        :param point: 点的坐标
        :param line_point1: 直线上第一点的坐标
        :param line_point2: 直线上另一点的坐标
        :return: 点到直线的距离
        """
        # 计算点和直线上点组成的向量
        vec1 = line_point1 - point
        vec2 = line_point2 - point
        dist = abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point2 - line_point1)
        return dist

    def wear_face_mask(self):
        self._face_img = Image.open(self.face_path)
        self._mask_img = Image.open(self.mask_path)
        face_landmarks = self.fa.get_landmarks(np.asarray(self._face_img))[0].astype(
            np.int32
        )
        # 获取需要的关键点信息
        self.get_key_landmarks(face_landmarks)

        # 获取口罩的宽和高
        mask_width = self._mask_img.width
        mask_height = self._mask_img.height
        # 计算口罩适应人脸后和高度
        new_mask_height = int(np.linalg.norm(self.bottom_chin_point - self.nose_point))

        # 将口罩分割为左右两部分用来适配人脸
        # 左边口罩人脸
        mask_left_img = self._mask_img.crop((0, 0, mask_width // 2, mask_height))
        mask_left_width = self.cal_point_to_line_dist(
            self.left_chin_point, self.nose_point, self.bottom_chin_point
        )
        mask_left_width = int(mask_left_width * self.enlarge_ratio)
        mask_left_img = mask_left_img.resize((mask_left_width, new_mask_height))

        # 右边口罩人脸
        mask_right_img = self._mask_img.crop(
            (mask_width // 2, 0, mask_width, mask_height)
        )
        mask_right_width = self.cal_point_to_line_dist(
            self.right_chin_point, self.nose_point, self.bottom_chin_point
        )
        mask_right_width = int(mask_right_width * self.enlarge_ratio)
        mask_right_img = mask_right_img.resize((mask_right_width, new_mask_height))

        # 合并口罩
        size = (mask_left_width + mask_right_width, new_mask_height)
        mask_img = Image.new("RGBA", size)
        mask_img.paste(mask_left_img, (0, 0), mask_left_img)
        mask_img.paste(mask_right_img, (mask_left_width, 0), mask_right_img)

        # 计算人脸的旋转角度
        angle = np.arctan2(
            self.bottom_chin_point[1] - self.nose_point[1],
            self.bottom_chin_point[0] - self.nose_point[0],
        )
        # 旋转口罩
        rotated_mask_img = mask_img.rotate(angle, expand=True)

        # 计算mask的位置
        mask_center_x = (self.nose_point[0] + self.bottom_chin_point[0]) // 2
        mask_center_y = (self.nose_point[1] + self.bottom_chin_point[1]) // 2
        offset = mask_img.width // 2 - mask_left_width
        # 将弧度转换为角度
        radian = angle * np.pi / 180
        box_x = (
            mask_center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
        )
        box_y = (
            mask_center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2
        )

        self._face_img.paste(mask_img, (box_x, box_y), mask_img)
        self.save()

    def save(self):
        self._face_img.save(self.save_path)
        print(f"Save to {self.save_path}")


face_path = "imgs/test.jpg"
mask_path = "mask_images/mask.png"
save_path = "imgs/face_mask.jpg"
WearFaceMask(face_path, mask_path, save_path).wear_face_mask()
