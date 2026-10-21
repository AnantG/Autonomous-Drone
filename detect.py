"""Run the rocket-part detector on an image, video file, folder, or camera.

Examples
--------
    python software/detect.py --source your_photo.jpg
    python software/detect.py --source flight_clip.mp4 --save
    python software/detect.py --source 0 --show          # webcam / USB camera

For each frame it prints every detection and, if a body tube is found, the
normalized centering error and the velocity command the controller would send.
It does NOT talk to the flight controller.
"""

import argparse

from ultralytics import YOLO

from software.centering import (
    Detection,
    compute_velocity,
    is_centered,
    pick_target,
    pixel_error,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="model/best.pt", help="path to .pt weights")
    p.add_argument("--source", default="0", help="image/video/folder path, or camera index like 0")
    p.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--target", default="body tube", help="class the drone should center on")
    p.add_argument("--show", action="store_true", help="display annotated frames")
    p.add_argument("--save", action="store_true", help="save annotated output (runs/detect/...)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    model = YOLO(args.weights)
    names = model.names  # {class_id: class_name}
    print(f"Classes: {list(names.values())}")

    results = model.predict(
        source=source,
        conf=args.conf,
        imgsz=args.imgsz,
        stream=True,   # generator, so long videos/cameras don't fill memory
        show=args.show,
        save=args.save,
        verbose=False,
    )

    for i, r in enumerate(results):
        height, width = r.orig_shape
        dets = []
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dets.append(
                Detection(names[int(box.cls)], float(box.conf), x1, y1, x2, y2)
            )

        summary = ", ".join(f"{d.cls} {d.conf:.2f}" for d in dets) or "nothing"
        line = f"frame {i}: {summary}"

        target = pick_target(dets, target_class=args.target, min_conf=args.conf)
        if target is not None:
            ex, ey = pixel_error(target, width, height)
            vx, vy = compute_velocity(ex, ey)
            line += (
                f" | error=({ex:+.2f}, {ey:+.2f}) cmd=(fwd {vx:+.2f}, right {vy:+.2f}) m/s"
                f"{' CENTERED' if is_centered(ex, ey) else ''}"
            )
        print(line)


if __name__ == "__main__":
    main()
