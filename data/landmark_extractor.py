"""
Facial Landmark Extractor for GLA-FatigueNet.
Uses MediaPipe Face Mesh to extract 468 landmarks and compute
geometric features: EAR, MAR, head pose, etc.
"""

import numpy as np
import cv2

try:
    import mediapipe as mp
    # Verify that mp.solutions is actually available (fails on Python 3.13+)
    _ = mp.solutions.face_mesh
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError):
    MEDIAPIPE_AVAILABLE = False
    print("[WARNING] MediaPipe not available. Geometric features will be zeros.")


class LandmarkExtractor:
    """
    Extract facial landmarks and compute geometric features using MediaPipe.
    
    Geometric Features (15-dimensional vector):
        0: EAR_left   (Eye Aspect Ratio - Left)
        1: EAR_right  (Eye Aspect Ratio - Right)
        2: EAR_avg    (Average EAR)
        3: MAR        (Mouth Aspect Ratio)
        4: MOR        (Mouth Openness Ratio)
        5: head_pitch (Head rotation around X-axis)
        6: head_yaw   (Head rotation around Y-axis)
        7: head_roll  (Head rotation around Z-axis)
        8: left_eyebrow_eye_dist  (Left eyebrow-eye distance ratio)
        9: right_eyebrow_eye_dist (Right eyebrow-eye distance ratio)
        10: nose_tip_to_chin_dist (Nose-chin distance ratio)
        11: eye_width_ratio       (Eye width / face width)
        12: mouth_width_ratio     (Mouth width / face width)
        13: face_symmetry         (Left-right face symmetry score)
        14: pupil_eccentricity    (Pupil displacement from center)
    """

    # MediaPipe landmark indices for key facial features
    # Left Eye
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    LEFT_EYE_VERTICAL_TOP = [385, 386, 387]
    LEFT_EYE_VERTICAL_BOTTOM = [373, 374, 380]
    LEFT_EYE_HORIZONTAL = [362, 263]

    # Right Eye
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_VERTICAL_TOP = [160, 159, 158]
    RIGHT_EYE_VERTICAL_BOTTOM = [144, 145, 153]
    RIGHT_EYE_HORIZONTAL = [33, 133]

    # Mouth
    MOUTH_OUTER = [61, 291, 0, 17]  # left, right, top, bottom
    MOUTH_INNER_TOP = [13]
    MOUTH_INNER_BOTTOM = [14]
    MOUTH_LEFT = [61]
    MOUTH_RIGHT = [291]

    # Eyebrows
    LEFT_EYEBROW = [276, 283, 282, 295, 300]
    RIGHT_EYEBROW = [46, 53, 52, 65, 70]

    # Face contour points
    FACE_LEFT = [234]
    FACE_RIGHT = [454]
    NOSE_TIP = [1]
    CHIN = [152]
    FOREHEAD = [10]

    # Head pose estimation - 6 key points
    POSE_LANDMARKS = [1, 33, 263, 61, 291, 199]

    def __init__(self, static_image_mode=True, max_num_faces=1, 
                 min_detection_confidence=0.5, num_features=15):
        self.num_features = num_features
        self.face_mesh = None
        
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=static_image_mode,
                max_num_faces=max_num_faces,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
            )

    def _euclidean(self, p1, p2):
        """Compute Euclidean distance between two points."""
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _get_landmark_coords(self, landmarks, idx, img_w, img_h):
        """Get (x, y) coordinates for a landmark index."""
        lm = landmarks[idx]
        return (lm.x * img_w, lm.y * img_h)

    def compute_ear(self, landmarks, eye_indices, img_w, img_h):
        """
        Compute Eye Aspect Ratio (EAR).
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        coords = [self._get_landmark_coords(landmarks, i, img_w, img_h) 
                   for i in eye_indices]
        
        # Vertical distances
        v1 = self._euclidean(coords[1], coords[5])
        v2 = self._euclidean(coords[2], coords[4])
        
        # Horizontal distance
        h = self._euclidean(coords[0], coords[3])
        
        if h < 1e-6:
            return 0.0
        
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def compute_mar(self, landmarks, img_w, img_h):
        """
        Compute Mouth Aspect Ratio (MAR).
        MAR = vertical_distance / horizontal_distance
        """
        top = self._get_landmark_coords(landmarks, 13, img_w, img_h)
        bottom = self._get_landmark_coords(landmarks, 14, img_w, img_h)
        left = self._get_landmark_coords(landmarks, 61, img_w, img_h)
        right = self._get_landmark_coords(landmarks, 291, img_w, img_h)

        vertical = self._euclidean(top, bottom)
        horizontal = self._euclidean(left, right)

        if horizontal < 1e-6:
            return 0.0

        return vertical / horizontal

    def compute_head_pose(self, landmarks, img_w, img_h):
        """
        Estimate head pose (pitch, yaw, roll) using solvePnP.
        Returns normalized angles in [-1, 1] range.
        """
        # 3D model points (generic face model)
        model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (-225.0, 170.0, -135.0),   # Left eye left corner
            (225.0, 170.0, -135.0),    # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left mouth corner
            (150.0, -150.0, -125.0),   # Right mouth corner
            (0.0, -330.0, -65.0),      # Chin
        ], dtype=np.float64)

        # 2D image points
        image_points = np.array([
            self._get_landmark_coords(landmarks, i, img_w, img_h)
            for i in self.POSE_LANDMARKS
        ], dtype=np.float64)

        # Camera internals (approximate)
        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))

        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                rotation_mat, _ = cv2.Rodrigues(rotation_vector)
                pose_mat = cv2.hconcat([rotation_mat, translation_vector])
                _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
                    cv2.hconcat([pose_mat, np.array([[0, 0, 0, 1]], dtype=np.float64).T])[:3]
                )
                
                pitch = euler_angles[0, 0] / 90.0  # Normalize to [-1, 1]
                yaw = euler_angles[1, 0] / 90.0
                roll = euler_angles[2, 0] / 90.0
                
                return np.clip([pitch, yaw, roll], -1.0, 1.0)
        except Exception:
            pass

        return np.array([0.0, 0.0, 0.0])

    def extract_features(self, image):
        """
        Extract all geometric features from an image.
        
        Args:
            image: numpy array (H, W, 3) in RGB format
            
        Returns:
            features: numpy array of shape (num_features,)
        """
        features = np.zeros(self.num_features, dtype=np.float32)

        if not MEDIAPIPE_AVAILABLE or self.face_mesh is None:
            return features

        img_h, img_w = image.shape[:2]

        # Convert to RGB if needed (MediaPipe expects RGB)
        if len(image.shape) == 2:  # Grayscale
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            image_rgb = image

        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return features

        landmarks = results.multi_face_landmarks[0].landmark

        try:
            # EAR (Eye Aspect Ratio)
            ear_left = self.compute_ear(landmarks, self.LEFT_EYE, img_w, img_h)
            ear_right = self.compute_ear(landmarks, self.RIGHT_EYE, img_w, img_h)
            ear_avg = (ear_left + ear_right) / 2.0
            
            features[0] = ear_left
            features[1] = ear_right
            features[2] = ear_avg

            # MAR (Mouth Aspect Ratio)
            features[3] = self.compute_mar(landmarks, img_w, img_h)

            # MOR (Mouth Openness Ratio) - using outer mouth
            top = self._get_landmark_coords(landmarks, 0, img_w, img_h)
            bottom = self._get_landmark_coords(landmarks, 17, img_w, img_h)
            left = self._get_landmark_coords(landmarks, 61, img_w, img_h)
            right = self._get_landmark_coords(landmarks, 291, img_w, img_h)
            mor_h = self._euclidean(left, right)
            features[4] = self._euclidean(top, bottom) / (mor_h + 1e-6)

            # Head Pose
            head_pose = self.compute_head_pose(landmarks, img_w, img_h)
            features[5] = head_pose[0]  # pitch
            features[6] = head_pose[1]  # yaw
            features[7] = head_pose[2]  # roll

            # Eyebrow-Eye Distance (normalized)
            face_h = self._euclidean(
                self._get_landmark_coords(landmarks, 10, img_w, img_h),
                self._get_landmark_coords(landmarks, 152, img_w, img_h)
            )
            
            if face_h > 1e-6:
                # Left eyebrow to eye
                lb_center = self._get_landmark_coords(landmarks, 282, img_w, img_h)
                le_center = self._get_landmark_coords(landmarks, 385, img_w, img_h)
                features[8] = self._euclidean(lb_center, le_center) / face_h

                # Right eyebrow to eye
                rb_center = self._get_landmark_coords(landmarks, 52, img_w, img_h)
                re_center = self._get_landmark_coords(landmarks, 160, img_w, img_h)
                features[9] = self._euclidean(rb_center, re_center) / face_h

                # Nose tip to chin distance (normalized)
                nose = self._get_landmark_coords(landmarks, 1, img_w, img_h)
                chin = self._get_landmark_coords(landmarks, 152, img_w, img_h)
                features[10] = self._euclidean(nose, chin) / face_h

            # Face width
            face_left = self._get_landmark_coords(landmarks, 234, img_w, img_h)
            face_right = self._get_landmark_coords(landmarks, 454, img_w, img_h)
            face_w = self._euclidean(face_left, face_right)

            if face_w > 1e-6:
                # Eye width ratio
                le_l = self._get_landmark_coords(landmarks, 362, img_w, img_h)
                le_r = self._get_landmark_coords(landmarks, 263, img_w, img_h)
                features[11] = self._euclidean(le_l, le_r) / face_w

                # Mouth width ratio
                features[12] = mor_h / face_w

            # Face symmetry (distance between midline and nose)
            midline_x = (face_left[0] + face_right[0]) / 2
            nose_x = self._get_landmark_coords(landmarks, 1, img_w, img_h)[0]
            if face_w > 1e-6:
                features[13] = 1.0 - abs(nose_x - midline_x) / (face_w / 2)

            # Pupil eccentricity (how centered the iris is)
            le_h_left = self._get_landmark_coords(landmarks, 362, img_w, img_h)
            le_h_right = self._get_landmark_coords(landmarks, 263, img_w, img_h)
            le_center_x = (le_h_left[0] + le_h_right[0]) / 2
            iris_center = self._get_landmark_coords(landmarks, 473, img_w, img_h)  # Left iris
            eye_w = self._euclidean(le_h_left, le_h_right)
            if eye_w > 1e-6:
                features[14] = abs(iris_center[0] - le_center_x) / (eye_w / 2)

        except (IndexError, Exception) as e:
            # If any feature extraction fails, return partial features
            pass

        return features

    def get_fatigue_label(self, features, emotion_label=None):
        """
        Derive fatigue label from geometric features and emotion.
        
        Args:
            features: geometric feature vector
            emotion_label: int (0-6) emotion class index
            
        Returns:
            fatigue_label: int (0=alert, 1=drowsy, 2=fatigued)
        """
        ear_avg = features[2]
        mar = features[3]
        
        # Classes: 0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral
        alert_emotions = {3, 5}      # happy, surprise
        drowsy_emotions = {6, 4}     # neutral, sad
        fatigued_emotions = {4, 2, 1}  # sad, fear, disgust

        # Primary: geometric indicators
        if ear_avg > 0 and ear_avg <= 0.18:
            return 2  # fatigued
        elif ear_avg > 0 and ear_avg <= 0.25:
            if mar > 0.5:
                return 1  # drowsy (yawning)
            elif emotion_label in drowsy_emotions:
                return 1  # drowsy
            else:
                return 0  # alert
        else:
            # EAR is normal or not detected
            if emotion_label is not None:
                if emotion_label in alert_emotions:
                    return 0
                elif emotion_label in fatigued_emotions:
                    return 2
                elif emotion_label in drowsy_emotions:
                    return 1
            return 0  # default: alert

    def close(self):
        """Release resources."""
        if self.face_mesh:
            self.face_mesh.close()
