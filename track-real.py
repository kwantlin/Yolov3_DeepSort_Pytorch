import argparse
from sys import platform

from yolov3.models import *  # set ONNX_EXPORT in models.py
from yolov3.utils.datasets import *
from yolov3.utils.utils import *
from deep_sort import DeepSort
import pyrealsense2 as rs
import numpy as np
import cv2

deepsort = DeepSort("deep_sort/deep/checkpoint/ckpt.t7")
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

# https://lifesaver.codes/answer/intel-realsense-loadwebcam-instead-of-loadstreams-692
class LoadRealSense2:  # Stream from Intel RealSense D435

    def __init__(self, pipe,cfg,profile,rspath,width='640', height='480', fps='30'):

        # Variabels for setup
        self.width = width
        self.height = height
        self.fps = fps
        self.imgs = [None]
        self.depths = [None]
        self.img_size = 416
        self.half = False

        # Setup
        # self.pipe = rs.pipeline()
        # self.cfg = rs.config()
        
        # self.cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        # self.cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        # # Start streaming
        # self.profile = self.pipe.start(self.cfg)
        # self.path = rs.pipeline_profile()
        # print(self.path)
        self.pipe=pipe
        self.cfg=cfg
        self.profile=profile
        self.path=rspath

        print("streaming at w = " + str(self.width) + " h = " + str(self.height) + " fps = " + str(self.fps))

    def update(self):

        while True:
            #Wait for frames and get the data
            self.frames = self.pipe.wait_for_frames()
            self.depth_frame = self.frames.get_depth_frame()
            self.color_frame = self.frames.get_color_frame()

            #Wait until RGB and depth frames are synchronised
            if not self.depth_frame or not self.color_frame:
                continue
            #get RGB data and convert it to numpy array
            img0 = np.asanyarray(self.color_frame.get_data())
            #print("ini image awal: " + str(np.shape(img0)))

            #align + color depth -> for display purpose only
            #udah di convert ke numpy array di def colorizing
            depth0 = self.colorizing(self.aligned(self.frames))

            # aligned depth -> for depth calculation
            # udah di convert ke numpy array di def kedalaman
            distance0 = self.kedalaman(self.frames)

            #get depth_scale
            depth_scale = self.scale(self.profile)

            #Expand dimensi image biar jadi 4 dimensi (biar bisa masuk ke fungsi letterbox)
            self.imgs = np.expand_dims(img0, axis=0)
            #print("ini img expand: " + str(np.shape(self.imgs)))

            #Kalo yang depth gaperlu, karena gaakan dimasukin ke YOLO
            self.depths = depth0
            self.distance = distance0
            break

        #print("ini depth awal: " + str(np.shape(self.depths)))

        s = np.stack([letterbox(x, new_shape=self.img_size)[0].shape for x in self.imgs], 0)  # inference shapes
        #print("ini s: " + str(np.shape(s)))

        self.rect = np.unique(s, axis=0).shape[0] == 1
        #print("ini rect: " + str(np.shape(self.rect)))

        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

        time.sleep(0.01)  # wait time
        return self.rect, depth_scale

    def scale(self, profile):
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        return depth_scale

    def kedalaman(self, frames):
        self.align = rs.align(rs.stream.color)
        frames = self.align.process(frames)
        aligned_depth_frame = frames.get_depth_frame()
        depth_real = np.asanyarray(aligned_depth_frame.get_data())
        return depth_real

    def aligned(self, frames):
        self.align = rs.align(rs.stream.color)
        frames = self.align.process(frames)
        aligned_depth_frame = frames.get_depth_frame()
        return aligned_depth_frame

    def colorizing(self, aligned_depth_frame):
        self.colorizer = rs.colorizer()
        colorized_depth = np.asanyarray(self.colorizer.colorize(aligned_depth_frame).get_data())
        return(colorized_depth)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        self.rect, depth_scale = self.update()
        img0 = self.imgs.copy()
        depth = self.depths.copy()
        distance = self.distance.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        img_path = 'realsense.jpg'

        # Letterbox
        img = [letterbox(x, new_shape=self.img_size, auto=self.rect, interp=cv2.INTER_LINEAR)[0] for x in img0]
        #print("ini img letterbox: " + str(np.shape(img)))

        # Stack
        img = np.stack(img, 0)
        #print("ini img-padding: " + str(np.shape(img)))

        # Convert Image
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to 3x416x416, uint8 to float32
        #print("ini img-RGB: " + str(np.shape(depth)))
        img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        #print("ini img-final: " + str(np.shape(img)))

        # Return depth, depth0, img, img0
        return str(img_path), depth, distance, depth_scale, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years

def bbox_rel(image_width, image_height, bbox_left, bbox_top, bbox_w, bbox_h):
    """" Calculates the relative bounding box from absolute pixel values. """
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0    
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img, (x1, y1),(x2,y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img



def detect(pipe,cfg,profile,rspath,save_img=True):
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Eval mode
    model.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    save_img=False
    view_img=True
    torch.backends.cudnn.benchmark=True
    dataset = LoadRealSense2(pipe,cfg,profile,rspath)
    # if webcam:
    #     save_img = False
    #     view_img = True
    #     torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(source, img_size=img_size, half=half)
    # else:
    #     save_img = True
    #     dataset = LoadImages(source, img_size=img_size, half=half)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    for path, depth, distance, depth_scale, img, im0s, _ in dataset:
        print("path", path)
        print("im0s", im0s.shape)
        print("img", img.shape)
        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(device)
        print(img.shape)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        print("final image size", img.shape)
        pred = model(img)[0]
        print(len(pred))
        if opt.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        print(pred[0].shape)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # print(det[:, :5])
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []

                # Write results
                for *xyxy, conf, cls in det:
                    img_h, img_w, _ = im0.shape  # get image shape
                    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
                    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
                    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
                    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, bbox_left, bbox_top, bbox_w, bbox_h)
                    #print(x_c, y_c, bbox_w, bbox_h)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                    label = '%s %.2f' % (names[int(cls)], conf)
                    #
                    #print('bboxes')
                    #print(torch.Tensor(bbox_xywh))
                    #print('confs')
                    #print(torch.Tensor(confs))
                    outputs = deepsort.update((torch.Tensor(bbox_xywh)), (torch.Tensor(confs)) , im0)
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        draw_boxes(im0, bbox_xyxy, identities)
                    #print('\n\n\t\ttracked objects')
                    #print(outputs)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, time.time() - t))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov3/cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='yolov3/data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='yolov3/weights/yolov3-spp-ultralytics.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default='0', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=608, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        # Setup
        pipe = rs.pipeline()
        cfg = rs.config()
        
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        profile = pipe.start(cfg)
        path = rs.pipeline_profile()
        detect(pipe,cfg,profile,path)
