import cv2
import argparse
from ultralytics import YOLO

def main(video_source, output_path=None):
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Try to open the video source
    try:
        source = int(video_source) # if it's an integer, use it for webcam
    except ValueError:
        source = video_source

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    # Get video properties for output if needed
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30 # Default if cannot be read

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video from {video_source}...")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # We process fewer frames if output_path is not given, just for test speed in headless
        if not output_path and frame_count > 100:
            print("Finished processing 100 frames. Exiting.")
            break

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, verbose=False)

        # Plot the results on the frame
        annotated_frame = results[0].plot()

        if out:
            out.write(annotated_frame)
        else:
            # If not saving to file, try to display (may fail in headless envs)
            try:
                # Need to import os to check DISPLAY
                import os
                if os.environ.get('DISPLAY'):
                    cv2.imshow("YOLOv8 Tracking", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            except cv2.error:
                # Silently catch cv2.imshow error in headless environments
                pass

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("Video processing complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Object Detection and Tracking using YOLOv8")
    parser.add_argument('source', nargs='?', default='0', help="Video source (file path or camera index, default 0)")
    parser.add_argument('--output', help="Output video path (e.g., output.mp4)")
    args = parser.parse_args()

    main(args.source, args.output)
