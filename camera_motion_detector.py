import cv2
import numpy as np

class CameraMotionDetector:
    def __init__(self,
                 max_trans_thresh=5.0,
                 max_rot_thresh=2.0,
                 min_inliers=30,
                 max_cum_trans=20.0,
                 drift_decay=0.9,
                 pix_diff_thresh=5,
                 pix_diff_pct=0.02):
        self.orb = cv2.ORB_create(1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.max_trans = max_trans_thresh
        self.max_rot   = max_rot_thresh
        self.min_inliers = min_inliers
        self.max_cum_trans = max_cum_trans
        self.drift_decay = drift_decay
        self.cum_dx = 0.0
        self.cum_dy = 0.0
        self.pix_diff_thresh = pix_diff_thresh
        self.pix_diff_pct = pix_diff_pct

        self.prev_gray = None

    def reset(self):
        self.prev_gray = None
        self.cum_dx = 0.0
        self.cum_dy = 0.0

    def _feature_motion(self, gray):
        kp1, ds1 = self.orb.detectAndCompute(self.prev_gray, None)
        kp2, ds2 = self.orb.detectAndCompute(gray, None)
        if ds1 is None or ds2 is None or len(kp1) < 10 or len(kp2) < 10:
            return 0, 0, 0, False

        matches = self.matcher.match(ds1, ds2)
        if len(matches) < self.min_inliers:
            return 0, 0, 0, False

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        if H is None or mask.ravel().sum() < self.min_inliers:
            return 0, 0, 0, False

        inl = mask.ravel()==1
        p1 = pts1[inl].reshape(-1,2)
        p2 = pts2[inl].reshape(-1,2)
        dxy = (p2 - p1).mean(axis=0)
        dx, dy = dxy.tolist()
        angle = np.degrees(np.arctan2(H[1,0], H[0,0]))
        return dx, dy, angle, True

    def _pixel_diff_motion(self, gray):
        diff = cv2.absdiff(self.prev_gray, gray)
        _, mask = cv2.threshold(diff, self.pix_diff_thresh, 255, cv2.THRESH_BINARY)
        return mask.sum() / (mask.size*255) > self.pix_diff_pct

    def is_camera_moved(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return False

        dx, dy, ang, valid = self._feature_motion(gray)
        if not valid:
            moved = self._pixel_diff_motion(gray)
            if moved: self.reset()
            else:    self.prev_gray = gray
            return moved

        self.cum_dx += dx
        self.cum_dy += dy
        if abs(dx)<1 and abs(dy)<1:
            self.cum_dx *= self.drift_decay
            self.cum_dy *= self.drift_decay

        if (np.hypot(dx,dy)>self.max_trans or abs(ang)>self.max_rot or
            np.hypot(self.cum_dx,self.cum_dy)>self.max_cum_trans):
            self.reset()
            return True

        self.prev_gray = gray
        return False
