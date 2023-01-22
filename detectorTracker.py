import cv2
import time

# from YOLOv3 import YOLOv3
from deep_sort import DeepSort
# from util import draw_bboxes
from detector import YOLOv3
from utils.draw import draw_boxes

yolo3 = YOLOv3("detector/YOLOv3/cfg/yolo_v3.cfg","yolov3.weights","detector/YOLOv3/cfg/coco.names", is_xywh=True)
deepsort = DeepSort("ckpt.t7")

video_capture = cv2.VideoCapture("Dubai Mall.mp4")
# if video_capture.open('video.mp4'):
width, height = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_capture.get(cv2.CAP_PROP_FPS)
# !rm - f output.mp4 output.avi
# can't write out mp4, so try to write into an AVI file
video_writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    start = time.time()
    xmin, ymin, xmax, ymax = 0, 0, width, height
    im = frame[ymin:ymax, xmin:xmax, (2, 1, 0)]
    bbox_xywh, cls_conf, cls_ids = yolo3(im)
    if bbox_xywh is not None:
        mask = cls_ids == 0
        bbox_xywh = bbox_xywh[mask]
        bbox_xywh[:, 3] *= 1.2
        cls_conf = cls_conf[mask]
        outputs = deepsort.update(bbox_xywh, cls_conf, im)
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            frame = draw_boxes(frame, bbox_xyxy, identities, offset=(xmin, ymin))


    end = time.time()
    print("time: {}s, fps: {}".format(end - start, 1 / (end - start)))
    # cv2.putText(frame, text, (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
    cv2.imshow("TrackingFrame", frame)
    video_writer.write(frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

video_capture.release()
video_writer.release()
cv2.destroyAllWindows()

# convert AVI to MP4
# ffmpeg - y - loglevel info - i output.avi output.mp4
# else:
#     print("can't open the given input video file!")

