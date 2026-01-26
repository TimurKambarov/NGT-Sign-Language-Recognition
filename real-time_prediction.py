"""
NGT Sign Language Recognition - Real-Time Prediction
Uses sliding window with 63 features (21 landmarks × 3 coordinates).

Controls:
    Q - Quit
    M - Toggle mirror mode
    Z - Toggle signing zone visibility

Usage:
    python real_time_prediction.py
"""

import cv2
import numpy as np
import joblib
import os
from collections import deque

try:
    import hand_face_detection as detection
except ImportError:
    print("Error: hand_face_detection.py not found in current directory.")
    print("Make sure you're running from the repository root.")
    exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Model path
MODEL_PATH = "models/random_forest.joblib"

# Sliding window settings
WINDOW_SIZE = 5          # Number of frames to average
MIN_FRAMES = 5          # Minimum frames before predicting

# Prediction settings
PREDICTION_INTERVAL = 3   # Predict every N frames
STABILITY_COUNT = 5       # Same prediction N times = stable
CONFIDENCE_THRESHOLD = 0.75

# Signing zone boundaries (relative to face)
SIGNING_ZONE = {
    'x_min': -2.5,
    'x_max': 2.5,
    'y_min': -0.7,
    'y_max': 1.7,
}

# Display
DISPLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_ORANGE = (0, 165, 255)


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(model_path):
    """Load trained model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run train_model.ipynb first."
        )
    
    model = joblib.load(model_path)
    print(f"Model loaded: {model_path}")
    return model


# =============================================================================
# SLIDING WINDOW (63 features)
# =============================================================================

class SlidingWindow:
    """
    Maintains sliding window of frames.
    Averages frames to produce 63 features for prediction.
    """
    
    def __init__(self, max_size=15):
        self.max_size = max_size
        self.frames = deque(maxlen=max_size)
    
    def add_frame(self, landmarks_normalized):
        """Add a frame (21 landmarks with x, y, z)."""
        frame_data = []
        for lm in landmarks_normalized:
            frame_data.extend([lm['x'], lm['y'], lm['z']])
        self.frames.append(frame_data)
    
    def clear(self):
        """Clear all frames."""
        self.frames.clear()
    
    def is_ready(self, min_frames=10):
        """Check if enough frames collected."""
        return len(self.frames) >= min_frames
    
    def get_features(self):
        """
        Average all frames in window to get 63 features.
        Returns numpy array of shape (1, 63).
        """
        if len(self.frames) == 0:
            return None
        
        # Stack all frames and compute mean
        data = np.array(list(self.frames))  # Shape: (num_frames, 63)
        averaged = np.mean(data, axis=0)    # Shape: (63,)
        
        return averaged.reshape(1, -1)
    
    def __len__(self):
        return len(self.frames)


# =============================================================================
# PREDICTION STABILIZER
# =============================================================================

class PredictionStabilizer:
    """Requires N consecutive same predictions for stability."""
    
    def __init__(self, required_count=5):
        self.required_count = required_count
        self.history = deque(maxlen=required_count)
        self.stable_letter = None
        self.stable_confidence = 0.0
    
    def update(self, letter, confidence):
        """Update with new prediction."""
        self.history.append((letter, confidence))
        
        if len(self.history) >= self.required_count:
            letters = [h[0] for h in self.history]
            if all(l == letters[0] for l in letters):
                self.stable_letter = letters[0]
                self.stable_confidence = np.mean([h[1] for h in self.history])
        
        return self.stable_letter, self.stable_confidence
    
    def reset(self):
        """Reset stabilizer."""
        self.history.clear()
        self.stable_letter = None
        self.stable_confidence = 0.0


# =============================================================================
# SIGNING ZONE
# =============================================================================

def is_in_signing_zone(relative_position):
    """Check if hand is in signing zone."""
    if relative_position is None:
        return False
    
    x = relative_position['rel_x']
    y = relative_position['rel_y']
    
    return (SIGNING_ZONE['x_min'] <= x <= SIGNING_ZONE['x_max'] and
            SIGNING_ZONE['y_min'] <= y <= SIGNING_ZONE['y_max'])


def draw_signing_zone(frame, face_refs, show_zone=True):
    """Draw signing zone rectangle."""
    if not show_zone or face_refs is None:
        return frame
    
    nose = face_refs['nose']
    fw = face_refs['face_width']
    fh = face_refs['face_height']
    
    x1 = int(nose[0] + SIGNING_ZONE['x_min'] * fw)
    x2 = int(nose[0] + SIGNING_ZONE['x_max'] * fw)
    y1 = int(nose[1] + SIGNING_ZONE['y_min'] * fh)
    y2 = int(nose[1] + SIGNING_ZONE['y_max'] * fh)
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_YELLOW, 2)
    return frame


# =============================================================================
# DISPLAY
# =============================================================================

def draw_prediction_box(frame, letter, confidence, window_size, in_zone):
    """Draw prediction result box."""
    h, w, _ = frame.shape
    
    box_x, box_y = w - 200, 10
    box_w, box_h = 190, 130
    
    # Background
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), COLOR_WHITE, 2)
    
    # Window fill bar
    fill_pct = min(window_size / WINDOW_SIZE, 1.0)
    bar_width = int((box_w - 20) * fill_pct)
    cv2.rectangle(frame, (box_x + 10, box_y + box_h - 20),
                 (box_x + 10 + bar_width, box_y + box_h - 10), COLOR_CYAN, -1)
    cv2.rectangle(frame, (box_x + 10, box_y + box_h - 20),
                 (box_x + box_w - 10, box_y + box_h - 10), COLOR_WHITE, 1)
    
    if not in_zone:
        cv2.putText(frame, "MOVE HAND", (box_x + 25, box_y + 45),
                   DISPLAY_FONT, 0.6, COLOR_RED, 2)
        cv2.putText(frame, "INTO ZONE", (box_x + 30, box_y + 70),
                   DISPLAY_FONT, 0.6, COLOR_RED, 2)
    elif letter is None:
        cv2.putText(frame, "Detecting...", (box_x + 30, box_y + 55),
                   DISPLAY_FONT, 0.6, COLOR_YELLOW, 2)
    elif confidence < CONFIDENCE_THRESHOLD:
        cv2.putText(frame, f"{letter}?", (box_x + 60, box_y + 60),
                   DISPLAY_FONT, 1.8, COLOR_ORANGE, 3)
        cv2.putText(frame, f"{confidence:.0%}", (box_x + 70, box_y + 90),
                   DISPLAY_FONT, 0.5, COLOR_ORANGE, 1)
    else:
        cv2.putText(frame, letter, (box_x + 60, box_y + 65),
                   DISPLAY_FONT, 2.2, COLOR_GREEN, 4)
        cv2.putText(frame, f"{confidence:.0%}", (box_x + 70, box_y + 95),
                   DISPLAY_FONT, 0.6, COLOR_GREEN, 2)
    
    return frame


def draw_status_bar(frame, face_ok, hand_ok, in_zone, mirrored, fps):
    """Draw status bar at bottom."""
    h, w, _ = frame.shape
    
    cv2.rectangle(frame, (10, h - 35), (500, h - 10), (0, 0, 0), -1)
    
    status = f"Face:{'OK' if face_ok else 'NO'} Hand:{'OK' if hand_ok else 'NO'} "
    status += f"Zone:{'OK' if in_zone else 'NO'} Mirror:{'ON' if mirrored else 'OFF'} FPS:{fps:.0f}"
    
    cv2.putText(frame, status, (15, h - 17), DISPLAY_FONT, 0.4, COLOR_WHITE, 1)
    cv2.putText(frame, "Q=quit M=mirror Z=zone", (350, h - 17), DISPLAY_FONT, 0.35, (150, 150, 150), 1)
    
    return frame


# =============================================================================
# MAIN
# =============================================================================

def main():
    
    # Load model
    model = load_model(MODEL_PATH)
    
    # Check model expects 63 features
    if hasattr(model, 'n_features_in_'):
        print(f"Model expects {model.n_features_in_} features")
        if model.n_features_in_ != 63:
            print(f"WARNING: Model expects {model.n_features_in_} features, not 63!")
    
    # Initialize detection
    mp_resources = detection.initialize_mediapipe()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # State
    window = SlidingWindow(max_size=WINDOW_SIZE)
    stabilizer = PredictionStabilizer(required_count=STABILITY_COUNT)
    
    mirrored = True
    show_zone = True
    frame_count = 0
    last_printed_letter = None
    
    # FPS tracking
    fps = 0
    fps_counter = 0
    fps_start = cv2.getTickCount()
    
    print("\nReady! Controls: Q=quit, M=mirror, Z=zone")
    print("Predictions appear automatically when hand is in zone.\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if mirrored:
            frame = cv2.flip(frame, 1)
        
        # Detect face and hands
        frame, face_refs, hands_data = detection.process_frame(frame, mp_resources)
        
        face_ok = face_refs is not None
        hand_ok = len(hands_data) > 0
        in_zone = False
        
        # Process hand if detected
        if hand_ok and face_ok:
            hand_data = hands_data[0]
            rel_pos = hand_data.get('relative_position')
            in_zone = is_in_signing_zone(rel_pos)
            
            if in_zone:
                # Add frame to sliding window
                window.add_frame(hand_data['landmarks_normalized'])
                
                # Predict every N frames
                if frame_count % PREDICTION_INTERVAL == 0 and window.is_ready(MIN_FRAMES):
                    features = window.get_features()
                    
                    if features is not None:
                        # Predict
                        prediction = model.predict(features)[0]
                        proba = model.predict_proba(features)[0]
                        confidence = proba.max()
                        
                        # Stabilize
                        stable_letter, stable_conf = stabilizer.update(prediction, confidence)
                        
                        # Print on change
                        if stable_letter != last_printed_letter and stable_letter is not None:
                            if stable_conf >= CONFIDENCE_THRESHOLD:
                                print(f"Detected: {stable_letter} ({stable_conf:.0%})")
                                last_printed_letter = stable_letter
            else:
                # Hand left zone
                window.clear()
                stabilizer.reset()
                last_printed_letter = None
        else:
            # No detection
            window.clear()
            stabilizer.reset()
            last_printed_letter = None
        
        # Get display values
        display_letter = stabilizer.stable_letter
        display_confidence = stabilizer.stable_confidence
        
        # Draw UI
        frame = draw_signing_zone(frame, face_refs, show_zone)
        frame = draw_prediction_box(frame, display_letter, display_confidence, len(window), in_zone)
        frame = draw_status_bar(frame, face_ok, hand_ok, in_zone, mirrored, fps)
        
        # Hand indicator
        if hand_ok:
            center = hands_data[0]['center_px']
            color = COLOR_GREEN if in_zone else COLOR_RED
            cv2.circle(frame, center, 8, color, -1)
        
        cv2.imshow('NGT Sign Language Recognition', frame)
        
        # FPS calculation
        frame_count += 1
        fps_counter += 1
        if fps_counter >= 30:
            fps_end = cv2.getTickCount()
            fps = 30 / ((fps_end - fps_start) / cv2.getTickFrequency())
            fps_start = fps_end
            fps_counter = 0
        
        # Input handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            mirrored = not mirrored
            print(f"Mirror: {'ON' if mirrored else 'OFF'}")
        elif key == ord('z'):
            show_zone = not show_zone
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    mp_resources['hands'].close()
    mp_resources['face_mesh'].close()
    print("\nStopped.")


if __name__ == "__main__":
    main()