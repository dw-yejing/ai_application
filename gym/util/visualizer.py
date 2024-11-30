from typing import Any
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2

class TensorboardVisualizer:
    def __init__(self, log_dir: str="."):
        """
        初始化函数，创建一个SummaryWriter对象用于记录训练数据到指定的日志目录。

        Args:
            log_dir (str): 存储tensorboard日志文件的目录路径。
        """
        self.writer = SummaryWriter(log_dir)

    def add_scalar(self, tag, scalar_value, global_step):
        """
        向tensorboard添加标量数据。

        Args:
            tag (str): 数据的标签，用于在tensorboard中区分不同的数据系列。
            scalar_value (float or int): 要记录的标量值。
            global_step (int): 当前的训练步数。
        """
        self.writer.add_scalar(tag, scalar_value, global_step)

    def add_image(self, tag, img_tensor, global_step):
        """
        向tensorboard添加图像数据。

        Args:
            tag (str): 图像数据的标签。
            img_tensor (torch.Tensor or tf.Tensor): 要记录的图像张量，形状应为 [batch_size, height, width, channels] 或 [height, width, channels]。
            global_step (int): 当前的训练步数。
        """
        self.writer.add_image(tag, img_tensor, global_step)

    def close(self):
        """
        关闭SummaryWriter，完成日志记录后需要调用此方法来释放资源。
        """
        self.writer.close()
        

def generate_video(frames:np.array, rewards:np.array, video_name:str="a.mp4"):
    # frames = np.transpose(frames, axes=[1,2,0])
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(video_name, fourcc, 25.0, (width, height))
    # 遍历图片列表并写入视频
    success = 0
    failure = 0
    for index, frame in enumerate(frames):
        success += rewards[index]
        failure += 1 - rewards[index]
        fid_label = f"fid: {index+1}"
        success_label = f"success: {int(success)}"
        failure_label = f"failure: {int(failure)}"
        cv2.putText(frame, fid_label, org=(10, 30), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness = 1)
        cv2.putText(frame, success_label, org=(10, 60), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness = 1)
        cv2.putText(frame, failure_label, org=(10, 90), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness = 1)
        if rewards[index].item() < 1:
            video.write(frame)
            video.write(frame)
            
        video.write(frame)
    video.release()
    cv2.destroyAllWindows()
    print("=== complete ===")
    
    
if __name__ == "__main__":
    frames = np.load("frames.npy")
    rewards = np.load("rewards.npy")
    generate_video(frames, rewards)