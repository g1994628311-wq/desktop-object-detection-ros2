"""Single-open USB camera YOLO11n ROS2 publisher; inference only."""
import json, platform, time
import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from ultralytics import YOLO

NAMES={0:"mouse",1:"keyboard",2:"laptop",3:"cup",4:"headphones"}
class Detector(Node):
 def __init__(self):
  super().__init__("desktop_object_detector")
  for n,v in (("model_path","runs/detect/yolo11n_final/weights/best.pt"),("camera_index",1),("imgsz",640),("conf",0.25),("iou",0.45),("device","0"),("publish_image",False),("show_window",True)): self.declare_parameter(n,v)
  p=lambda n:self.get_parameter(n).value; self.model=YOLO(p("model_path"))
  if self.model.names!=NAMES: raise RuntimeError(f"Class mapping mismatch: {self.model.names}")
  backend=cv2.CAP_DSHOW if platform.system()=="Windows" else cv2.CAP_V4L2
  self.cap=cv2.VideoCapture(p("camera_index"),backend)
  if not self.cap.isOpened(): raise RuntimeError("Camera open failed")
  self.det_pub=self.create_publisher(Detection2DArray,"/detections",10); self.json_pub=self.create_publisher(String,"/detections_json",10); self.img_pub=self.create_publisher(Image,"/detection_image",10); self.bridge=CvBridge(); self.frame=0; self.last=time.perf_counter(); self.create_timer(0.001,self.step)
 def step(self):
  start=time.perf_counter(); ok,img=self.cap.read()
  if not ok: self.get_logger().warning("Camera read failed"); return
  p=lambda n:self.get_parameter(n).value; r=self.model.predict(img,imgsz=p("imgsz"),conf=p("conf"),iou=p("iou"),device=p("device"),verbose=False)[0]; arr=Detection2DArray(); arr.header.stamp=self.get_clock().now().to_msg(); ds=[]
  for b in r.boxes:
   x1,y1,x2,y2=map(float,b.xyxy[0].tolist()); c=int(b.cls.item()); q=float(b.conf.item()); d=Detection2D(); d.bbox.center.position.x=(x1+x2)/2; d.bbox.center.position.y=(y1+y2)/2; d.bbox.size_x=x2-x1; d.bbox.size_y=y2-y1; h=ObjectHypothesisWithPose(); h.hypothesis.class_id=str(c); h.hypothesis.score=q; d.results.append(h); arr.detections.append(d); ds.append({"class_id":c,"class_name":NAMES[c],"confidence":q,"bbox":{"x1":x1,"y1":y1,"x2":x2,"y2":y2}}); cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
  fps=1/max(time.perf_counter()-start,1e-9); self.det_pub.publish(arr); self.json_pub.publish(String(data=json.dumps({"frame_id":self.frame,"timestamp":time.time(),"fps":fps,"detections":ds})))
  if p("publish_image"): self.img_pub.publish(self.bridge.cv2_to_imgmsg(img,encoding="bgr8"))
  if p("show_window"): cv2.imshow("ROS2 YOLO11n",img);
  if p("show_window") and cv2.waitKey(1)&255==ord("q"): rclpy.shutdown()
  self.frame+=1
 def destroy_node(self): self.cap.release(); cv2.destroyAllWindows(); return super().destroy_node()
def main(): rclpy.init(); n=Detector(); rclpy.spin(n); n.destroy_node(); rclpy.shutdown()
