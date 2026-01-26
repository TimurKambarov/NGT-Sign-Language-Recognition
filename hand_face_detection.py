"""
NGT Sign Language Recognition - Hand + Face Detection Module
Detects hands and face to track hand position relative to body.

Key Face Mesh landmarks used:
    - Nose tip (1): Center reference point
    - Chin (152): Lower face boundary
    - Left ear (234): Left side reference
    - Right ear (454): Right side reference
    - Forehead (10): Upper face boundary

Controls:
    - Press 'q' to quit
    - Press 'm' to toggle mirrored view
    - Press 'z' to toggle signing zone visualization
"""

import cv2
import mediapipe as mp
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

# Signing zone boundaries (relative to face)
# These define where the hand should be for proper NGT signing
SIGNING_ZONE = {
    'x_min': -2.5,   # Left boundary (face widths from nose)
    'x_max': 2.5,    # Right boundary
    'y_min': -0.7,   # Upper boundary (above nose)
    'y_max': 1.7,    # Lower boundary (below nose)
}

# Colors (BGR format)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_ORANGE = (0, 165, 255)


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_mediapipe():
    """Initialize MediaPipe Hands and Face Mesh solutions."""
    
    # Hand detection
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    # Face mesh detection
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        refine_landmarks=False  # Set True for iris tracking (not needed here)
    )
    
    # Drawing utilities
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    return {
        'hands': hands,
        'face_mesh': face_mesh,
        'mp_hands': mp_hands,
        'mp_face_mesh': mp_face_mesh,
        'mp_drawing': mp_drawing,
        'mp_drawing_styles': mp_drawing_styles
    }


# =============================================================================
# FACE PROCESSING
# =============================================================================

def extract_face_references(face_landmarks, frame_width, frame_height):
    """
    Extract key face reference points for body-relative positioning.
    
    Returns dict with:
        - nose: (x, y) pixel coordinates of nose tip
        - chin: (x, y) pixel coordinates of chin
        - left_ear: (x, y) pixel coordinates
        - right_ear: (x, y) pixel coordinates
        - forehead: (x, y) pixel coordinates
        - face_width: estimated face width in pixels
        - face_height: estimated face height in pixels
    """
    
    # Key landmark indices in Face Mesh (468 total landmarks)
    NOSE_TIP = 1
    CHIN = 152
    FOREHEAD = 10
    LEFT_EAR = 234
    RIGHT_EAR = 454
    
    def get_point(idx):
        lm = face_landmarks.landmark[idx]
        return (int(lm.x * frame_width), int(lm.y * frame_height))
    
    nose = get_point(NOSE_TIP)
    chin = get_point(CHIN)
    forehead = get_point(FOREHEAD)
    left_ear = get_point(LEFT_EAR)
    right_ear = get_point(RIGHT_EAR)
    
    # Calculate face dimensions
    face_width = abs(right_ear[0] - left_ear[0])
    face_height = abs(chin[1] - forehead[1])
    
    return {
        'nose': nose,
        'chin': chin,
        'forehead': forehead,
        'left_ear': left_ear,
        'right_ear': right_ear,
        'face_width': max(face_width, 1),  # Avoid division by zero
        'face_height': max(face_height, 1)
    }


def draw_face_references(frame, face_refs, show_zone=True):
    """Draw face reference points and signing zone on frame."""
    
    nose = face_refs['nose']
    chin = face_refs['chin']
    forehead = face_refs['forehead']
    fw = face_refs['face_width']
    fh = face_refs['face_height']
    
    # Draw key face points
    cv2.circle(frame, nose, 5, COLOR_BLUE, -1)
    cv2.circle(frame, chin, 5, COLOR_BLUE, -1)
    cv2.circle(frame, forehead, 5, COLOR_BLUE, -1)
    
    # Draw signing zone if enabled
    if show_zone:
        zone_x1 = int(nose[0] + SIGNING_ZONE['x_min'] * fw)
        zone_x2 = int(nose[0] + SIGNING_ZONE['x_max'] * fw)
        zone_y1 = int(nose[1] + SIGNING_ZONE['y_min'] * fh)
        zone_y2 = int(nose[1] + SIGNING_ZONE['y_max'] * fh)
        
        cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), 
                     COLOR_YELLOW, 2)
        cv2.putText(frame, "Signing Zone", (zone_x1, zone_y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_YELLOW, 1)
    
    return frame


# =============================================================================
# HAND PROCESSING
# =============================================================================

def extract_hand_data(hand_landmarks, handedness, frame_width, frame_height):
    """
    Extract hand landmark data.
    
    Returns dict with:
        - landmarks: list of 21 {id, x, y, z} dicts (normalized 0-1)
        - landmarks_px: list of 21 (x, y) pixel coordinates
        - wrist_px: (x, y) pixel coordinates of wrist
        - center_px: (x, y) approximate center of hand
        - handedness: 'Left' or 'Right'
    """
    
    landmarks = []
    landmarks_px = []
    
    for idx, lm in enumerate(hand_landmarks.landmark):
        landmarks.append({
            'id': idx,
            'x': lm.x,
            'y': lm.y,
            'z': lm.z
        })
        landmarks_px.append((int(lm.x * frame_width), int(lm.y * frame_height)))
    
    wrist_px = landmarks_px[0]
    
    # Calculate hand center (average of all landmarks)
    center_x = int(np.mean([p[0] for p in landmarks_px]))
    center_y = int(np.mean([p[1] for p in landmarks_px]))
    
    return {
        'landmarks': landmarks,
        'landmarks_px': landmarks_px,
        'wrist_px': wrist_px,
        'center_px': (center_x, center_y),
        'handedness': handedness
    }


def calculate_relative_position(hand_data, face_refs):
    """
    Calculate hand position relative to face.
    
    Returns dict with:
        - rel_x: horizontal offset from nose (in face_widths, negative=left)
        - rel_y: vertical offset from nose (in face_heights, negative=above)
        - distance_to_nose: Euclidean distance to nose (in face_widths)
        - in_signing_zone: boolean, True if hand is in proper position
    """
    
    hand_center = hand_data['center_px']
    nose = face_refs['nose']
    fw = face_refs['face_width']
    fh = face_refs['face_height']
    
    # Calculate relative position (normalized by face size)
    rel_x = (hand_center[0] - nose[0]) / fw
    rel_y = (hand_center[1] - nose[1]) / fh
    
    # Euclidean distance
    distance = np.sqrt(rel_x**2 + rel_y**2)
    
    # Check if in signing zone
    in_zone = (SIGNING_ZONE['x_min'] <= rel_x <= SIGNING_ZONE['x_max'] and
               SIGNING_ZONE['y_min'] <= rel_y <= SIGNING_ZONE['y_max'])
    
    return {
        'rel_x': rel_x,
        'rel_y': rel_y,
        'distance_to_nose': distance,
        'in_signing_zone': in_zone
    }


def normalize_landmarks_to_wrist(landmarks, include_z=True):
    wrist = landmarks[0]
    normalized = []
    
    for lm in landmarks:
        norm_lm = {
            'id': lm['id'],
            'x': lm['x'] - wrist['x'],
            'y': lm['y'] - wrist['y'],
            'z': lm['z'] - wrist['z']  # Always included now
        }
        normalized.append(norm_lm)
    
    return normalized


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_frame(frame, mp_resources):
    """
    Process a single frame: detect face and hands, calculate relative positions.
    
    Returns:
        - frame: annotated frame
        - face_refs: face reference data (or None)
        - hands_data: list of hand data with relative positions (or empty list)
    """
    
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process face
    face_results = mp_resources['face_mesh'].process(rgb_frame)
    face_refs = None
    
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        face_refs = extract_face_references(face_landmarks, w, h)
    
    # Process hands
    hand_results = mp_resources['hands'].process(rgb_frame)
    hands_data = []
    
    if hand_results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            # Get handedness
            handedness = hand_results.multi_handedness[idx].classification[0].label
            
            # Extract hand data
            hand_data = extract_hand_data(hand_landmarks, handedness, w, h)
            
            # Add normalized landmarks (relative to wrist)
            hand_data['landmarks_normalized'] = normalize_landmarks_to_wrist(
                hand_data['landmarks']
            )
            
            # Calculate position relative to face (if face detected)
            if face_refs:
                hand_data['relative_position'] = calculate_relative_position(
                    hand_data, face_refs
                )
            else:
                hand_data['relative_position'] = None
            
            # Draw hand landmarks
            mp_resources['mp_drawing'].draw_landmarks(
                frame,
                hand_landmarks,
                mp_resources['mp_hands'].HAND_CONNECTIONS,
                mp_resources['mp_drawing_styles'].get_default_hand_landmarks_style(),
                mp_resources['mp_drawing_styles'].get_default_hand_connections_style()
            )
            
            hands_data.append(hand_data)
    
    return frame, face_refs, hands_data


def draw_hand_status(frame, hand_data, y_offset):
    """Draw hand status information on frame."""
    
    rel_pos = hand_data['relative_position']
    handedness = hand_data['handedness']
    center = hand_data['center_px']
    
    if rel_pos:
        in_zone = rel_pos['in_signing_zone']
        color = COLOR_GREEN if in_zone else COLOR_RED
        status = "IN ZONE" if in_zone else "OUT OF ZONE"
        
        # Draw hand center point
        cv2.circle(frame, center, 8, color, -1)
        
        # Draw status text
        cv2.putText(frame, f"{handedness}: {status}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"  Pos: ({rel_pos['rel_x']:.2f}, {rel_pos['rel_y']:.2f})",
                   (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
    else:
        cv2.putText(frame, f"{handedness}: NO FACE REF", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ORANGE, 2)
    
    return frame


def display_info(frame, face_detected, num_hands, mirrored, show_zone):
    """Display status information overlay."""
    
    h, w, _ = frame.shape
    
    # Background
    cv2.rectangle(frame, (10, 10), (300, 80), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (300, 80), COLOR_WHITE, 1)
    
    # Status
    face_status = "Yes" if face_detected else "No"
    face_color = COLOR_GREEN if face_detected else COLOR_RED
    
    cv2.putText(frame, f"Face: {face_status} | Hands: {num_hands}", (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 1)
    cv2.putText(frame, f"Mirror: {'ON' if mirrored else 'OFF'} (m) | Zone: {'ON' if show_zone else 'OFF'} (z)",
               (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
    cv2.putText(frame, "Press 'q' to quit", (20, 75),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)
    
    return frame


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function to run hand + face detection."""
    
    print("Initializing hand + face detection...")
    mp_resources = initialize_mediapipe()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Ready!")
    print("Controls: 'q'=quit, 'm'=mirror, 'z'=toggle zone")
    
    mirrored = True
    show_zone = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if mirrored:
            frame = cv2.flip(frame, 1)
        
        # Process frame
        frame, face_refs, hands_data = process_frame(frame, mp_resources)
        
        # Draw face references and zone
        if face_refs:
            frame = draw_face_references(frame, face_refs, show_zone)
        
        # Draw hand status
        for i, hand_data in enumerate(hands_data):
            frame = draw_hand_status(frame, hand_data, 120 + i * 60)
        
        # Draw info overlay
        frame = display_info(frame, face_refs is not None, len(hands_data), 
                            mirrored, show_zone)
        
        cv2.imshow('NGT Sign Language - Hand + Face Detection', frame)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            mirrored = not mirrored
        elif key == ord('z'):
            show_zone = not show_zone
    
    cap.release()
    cv2.destroyAllWindows()
    mp_resources['hands'].close()
    mp_resources['face_mesh'].close()
    print("Stopped.")

# Make file import-friendly

if __name__ == "__main__":
    main()