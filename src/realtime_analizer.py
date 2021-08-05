import cv2
import numpy as np

import torch

import _init_paths
from tracker.multitracker import JDETracker
from tracking_utils.timer import Timer
from tracking_utils import visualization as vis
from opts import opts


def main(opt, frame_rate=30, use_cuda=True, show_image=True):
    cap = cv2.VideoCapture(0)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    img0 = None
    frame_id = 0
    while True:
        ret, img0 = cap.read()
        if not ret:
            break
        img = img0[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        timer.tic()
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        timer.toc()

        if show_image:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
            cv2.imshow('online_im', online_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_id += 1


if __name__ == "__main__":
    opt = opts().init()
    main(opt)
