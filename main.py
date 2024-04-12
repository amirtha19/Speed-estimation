import argparse
import os
import csv
from collections import defaultdict, deque
from tqdm import tqdm
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO
from constants import SOURCE,TARGET
from coordinates import view_transformer
import streamlit as st

ROOT_OUTPUT_DIR = "detections/"
TARGET_VIDEO_PATH = "output/vehicles-result.mp4"
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.3
MODEL_RESOLUTION = 960

def init_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='Runs speed detection on video')
    parser.add_argument("-s", "--source", default=None, help="The source for the detector to run on, \
                              can be the path to a directory or a file.")
    parser.add_argument("--no-visual", default=False, help="Whether to run the detection in background without visual feedback.")
    parser.add_argument('-v', '--verbose', type=bool, default=False, help="Whether to print additional detected information to the stdout.")
    parser.add_argument('--hide', type=bool, default=False, help="Hide detected information on the video output.")
    args = parser.parse_args()
    return args
class speed_detector:
    def __init__(self, args, SOURCE_VIDEO_PATH, output_dir):
        self.args = args
        self.SOURCE_VIDEO_PATH = SOURCE_VIDEO_PATH
        self.output_dir = output_dir
        self.ready()
         
    def ready(self):
        # use gpu if available else use cpu
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.device(device)

        # load the trained model
        self.model = YOLO("yolov8n.pt", task="detect")
        print(f"Running on device type: {device}")
        self.prepare()

    def prepare(self):
        video_info = sv.VideoInfo.from_video_path(video_path=self.SOURCE_VIDEO_PATH)

        frame_generator = sv.get_video_frames_generator(source_path=self.SOURCE_VIDEO_PATH)

        # tracer initiation
        byte_track = sv.ByteTrack(frame_rate=video_info.fps, track_activation_threshold=CONFIDENCE_THRESHOLD)

        # annotators configuration
        thickness = sv.calculate_dynamic_line_thickness(
            resolution_wh=video_info.resolution_wh
        )
        text_scale = sv.calculate_dynamic_text_scale(
            resolution_wh=video_info.resolution_wh
        )
        bounding_box_annotator = sv.BoundingBoxAnnotator(
            thickness=thickness
        )
        label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=thickness,
            text_position=sv.Position.BOTTOM_CENTER
        )
        trace_annotator = sv.TraceAnnotator(
            thickness=thickness,
            trace_length=video_info.fps * 2,
            position=sv.Position.BOTTOM_CENTER
        )

        polygon_zone = sv.PolygonZone(
            polygon=SOURCE,
            frame_resolution_wh=video_info.resolution_wh
        )

        coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
        self.speed_detect(video_info, frame_generator, byte_track, bounding_box_annotator, label_annotator, trace_annotator, polygon_zone, coordinates)

    def speed_detect(self, video_info, frame_generator, byte_track, bounding_box_annotator, label_annotator, trace_annotator, polygon_zone, coordinates):
        with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
            print("Calculating speed")

            with open('frame.csv', mode='w', newline='') as file:
                writer = csv.writer(file)

                # loop over source video frame
                frame_no = 1
                for frame in tqdm(frame_generator, total=video_info.total_frames):
                    result = self.model(frame, imgsz=MODEL_RESOLUTION, verbose=False)[0]
                    detections = sv.Detections.from_ultralytics(result)

                    # filter out detections by class and confidence
                    detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
                    detections = detections[detections.class_id != 0]

                    # filter out detections outside the zone
                    detections = detections[polygon_zone.trigger(detections)]

                    # refine detections using non-max suppression
                    detections = detections.with_nms(IOU_THRESHOLD)

                    # pass detection through the tracker
                    detections = byte_track.update_with_detections(detections=detections)

                    points = detections.get_anchors_coordinates(
                        anchor=sv.Position.BOTTOM_CENTER
                    )

                    # calculate the detections position inside the target RoI
                    points = view_transformer.transform_points(points=points).astype(int)

                    # store detections position
                    for tracker_id, [_, y] in zip(detections.tracker_id, points):
                        coordinates[tracker_id].append(y)

                    # format labels
                    labels = []

                    for tracker_id in detections.tracker_id:
                        if len(coordinates[tracker_id]) < video_info.fps / 2:
                            labels.append(f"#{tracker_id}")
                        else:
                            # calculate speed
                            coordinate_start = coordinates[tracker_id][-1]
                            coordinate_end = coordinates[tracker_id][0]
                            distance = abs(coordinate_start - coordinate_end)
                            time = len(coordinates[tracker_id]) / video_info.fps
                            speed = distance / time * 3.6
                            labels.append(f"#{tracker_id} {int(speed)} km/h")
                            writer.writerow([frame_no, tracker_id, speed, detections.class_id])

                    # annotate frame
                    annotated_frame = frame.copy()
                    annotated_frame = trace_annotator.annotate(
                        scene=annotated_frame, detections=detections
                    )
                    annotated_frame = bounding_box_annotator.annotate(
                        scene=annotated_frame, detections=detections
                    )
                    annotated_frame = label_annotator.annotate(
                        scene=annotated_frame, detections=detections, labels=labels
                    )
                    frame_no +=1
                    # add frame to target video
                    sink.write_frame(annotated_frame)


# def main():

#     # Create a file uploader
#     videos_folder = 'videos/'

#     # Get the list of files in the videos folder
#     files = os.listdir(videos_folder)

#     # Sort the files by modification time (most recent first)
#     files.sort(key=lambda x: os.path.getmtime(os.path.join(videos_folder, x)), reverse=True)

#     # Get the path to the latest video file
#     latest_video_path = os.path.join(videos_folder, files[0]) if files else None
#     if latest_video_path is not None:
#         # Process the video using speed_detector
#         args = init_argparse()
#         speed_detector(args, latest_video_path, "output_dir/")

# if __name__ == '__main__':
#     main()
