"""Warp a wide pinhole render into an equidistant fisheye image, to match the real wrist camera.

MuJoCo cameras are pinhole. The real wrist camera is a ~185 deg fisheye whose 1280x720 frame is
centre-cropped to 720x720 for the policy; along that crop's edge-centres the view angle is ~56 deg
and the corners run past 75 deg. A pinhole source of fovy 125 deg covers 62 deg at edge-centres and
~70 deg in the corners; beyond that the output is black, like the real lens's dark corners.
"""
import cv2
import numpy as np


class Fisheye:
    def __init__(self, src_res, src_fovy_deg, out_res, theta_edge_deg=56.0):
        fp = (src_res / 2) / np.tan(np.radians(src_fovy_deg) / 2)          # pinhole focal, px
        ff = (out_res / 2) / np.radians(theta_edge_deg)                    # fisheye px per radian
        v, u = np.mgrid[0:out_res, 0:out_res].astype(np.float32)
        dx, dy = u - (out_res - 1) / 2, v - (out_res - 1) / 2
        r = np.hypot(dx, dy)
        theta = r / ff
        rs = fp * np.tan(np.clip(theta, 0, np.radians(89)))                # radius in the pinhole image
        k = np.where(r > 1e-6, rs / np.maximum(r, 1e-6), fp / ff)
        self.mx = (dx * k + (src_res - 1) / 2).astype(np.float32)
        self.my = (dy * k + (src_res - 1) / 2).astype(np.float32)
        self.valid = (theta < np.radians(88)) & (self.mx >= 0) & (self.mx <= src_res - 1) & (self.my >= 0) & (self.my <= src_res - 1)
        vig = np.clip(1.15 - (theta / np.radians(theta_edge_deg * 1.45)) ** 4, 0, 1)   # soft falloff toward the rim
        self.vig = (vig * self.valid).astype(np.float32)[..., None]

    def rgb(self, img):
        out = cv2.remap(img, self.mx, self.my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return (out.astype(np.float32) * self.vig).astype(np.uint8)

    def ids(self, seg, fill=-1):
        out = cv2.remap(seg.astype(np.float32), self.mx, self.my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
        out[~self.valid] = fill
        return out.astype(np.int32)
