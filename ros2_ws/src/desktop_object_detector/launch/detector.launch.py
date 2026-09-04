from launch import LaunchDescription
from launch_ros.actions import Node
def generate_launch_description(): return LaunchDescription([Node(package="desktop_object_detector",executable="desktop_object_detector",parameters=[{"camera_index":1,"model_path":"runs/detect/yolo11n_final/weights/best.pt","conf":0.25,"iou":0.45,"device":"0","show_window":True,"publish_image":False}])])
