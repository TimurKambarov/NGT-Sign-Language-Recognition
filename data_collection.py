"""
NGT Sign Language Recognition - Data Collection Module
Collects hand landmark sequences for training the classifier.

Controls:
    A-Y keys  : Select letter directly 
    SPACE     : Record 100 frames
    1         : Quit and save
    2         : Toggle signing zone on/off
    3         : Toggle mirror mode
    4         : Show guide for current letter
    5         : Reset dataset 

Output:
    - data/samples.csv: All collected samples
    - Format: 100 rows per sample with sample_id, frame_id, 63 features (21 × 3), label
"""

import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime

try:
    import hand_face_detection as detection
except ImportError:
    print("Error: hand_face_detection.py not found in current directory.")
    print("Make sure you're running from the repository root.")
    exit(1)

try:
    from guide import show_guide
except ImportError:
    print("Warning: guide.py not found. Letter guides will be unavailable.")
    show_guide = None


# =============================================================================
# CONFIGURATION
# =============================================================================

# Output settings
DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "samples.csv")

# Recording settings
FRAMES_PER_SAMPLE = 30  # Number of frames to record per sample (~3 sec at 30fps)

# Static letters only (excluding dynamic: H, J, Z)
STATIC_LETTERS = list("ABCDEFGIKLMNOPQRSTUVWXY")

# Letters per row in the UI display
LETTERS_PER_ROW = 8

# Signing zone boundaries (relative to face)
SIGNING_ZONE = {
    'x_min': -2.5,
    'x_max': 2.5,
    'y_min': -0.7,
    'y_max': 1.7,
}

# Colors
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_ORANGE = (0, 165, 255)


# =============================================================================
# DATA MANAGEMENT
# =============================================================================

def create_column_names():
    """
    Create column names for the CSV.
    Format: sample_id, frame_id, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20, label
    """
    columns = ['sample_id', 'frame_id']
    for i in range(21):
        columns.append(f"x{i}")
        columns.append(f"y{i}")
        columns.append(f"z{i}")
    columns.append("label")
    return columns


def landmarks_to_row(landmarks_normalized, sample_id, frame_id, label):
    """
    Convert normalized landmarks to a flat row for CSV.
    
    Args:
        landmarks_normalized: List of 21 dicts with 'x', 'y', 'z' keys
        sample_id: Unique identifier for this sample
        frame_id: Frame number within the sample (0-99)
        label: Letter label (e.g., 'A')
    
    Returns:
        List of values: [sample_id, frame_id, x0, y0, z0, ..., x20, y20, z20, label]
    """
    row = [sample_id, frame_id]
    for lm in landmarks_normalized:
        row.append(lm['x'])
        row.append(lm['y'])
        row.append(lm['z'])
    row.append(label)
    return row


def load_existing_data(filepath):
    """Load existing CSV data or create empty DataFrame."""
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} existing rows from {filepath}")
        return df
    else:
        columns = create_column_names()
        print("Starting with empty dataset")
        return pd.DataFrame(columns=columns)


def save_data(df, filepath):
    """Save DataFrame to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(df)} rows to {filepath}")


def get_next_sample_id(df):
    """Get the next available sample_id."""
    if len(df) == 0:
        return 0
    return df['sample_id'].max() + 1


def get_sample_counts(df):
    """Get count of complete samples per letter."""
    if len(df) == 0:
        return {}
    # Count unique sample_ids per label
    samples_per_label = df.groupby('label')['sample_id'].nunique()
    return samples_per_label.to_dict()


def reset_dataset(filepath):
    """Delete existing dataset and create empty one."""
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"Deleted {filepath}")
    columns = create_column_names()
    df = pd.DataFrame(columns=columns)
    return df


# =============================================================================
# SIGNING ZONE CHECK
# =============================================================================

def is_in_signing_zone(relative_position):
    """Check if hand is within the signing zone."""
    if relative_position is None:
        return False
    
    x = relative_position['rel_x']
    y = relative_position['rel_y']
    
    return (SIGNING_ZONE['x_min'] <= x <= SIGNING_ZONE['x_max'] and
            SIGNING_ZONE['y_min'] <= y <= SIGNING_ZONE['y_max'])


# =============================================================================
# UI DRAWING
# =============================================================================

def draw_signing_zone(frame, face_refs, show_zone=True):
    """Draw signing zone rectangle based on face position."""
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


def draw_letter_selector(frame, current_letter, sample_counts):
    """Draw the letter selection UI on the frame."""
    h, w, _ = frame.shape
    
    # Background panel
    panel_height = 180
    cv2.rectangle(frame, (10, h - panel_height - 10), (w - 10, h - 10), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, h - panel_height - 10), (w - 10, h - 10), COLOR_WHITE, 2)
    
    # Title
    cv2.putText(frame, "Press letter key to select | SPACE=record | 1=quit | 2=zone | 3=mirror | 4=guide | 5=reset",
               (20, h - panel_height + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)
    
    # Draw letters in grid
    start_y = h - panel_height + 40
    box_size = 35
    margin = 5
    
    for idx, letter in enumerate(STATIC_LETTERS):
        row = idx // LETTERS_PER_ROW
        col = idx % LETTERS_PER_ROW
        
        x = 20 + col * (box_size + margin)
        y = start_y + row * (box_size + margin)
        
        # Highlight selected letter
        if letter == current_letter:
            cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), COLOR_GREEN, -1)
            text_color = (0, 0, 0)
        else:
            cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), COLOR_WHITE, 1)
            text_color = COLOR_WHITE
        
        # Letter
        cv2.putText(frame, letter, (x + 10, y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # Sample count (small, below letter)
        count = sample_counts.get(letter, 0)
        cv2.putText(frame, str(count), (x + 12, y + box_size - 3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_CYAN, 1)
    
    return frame


def draw_status(frame, current_letter, sample_counts, hand_in_zone, is_recording, frames_recorded, mirrored, show_zone):
    """Draw status information at the top of the frame."""
    h, w, _ = frame.shape
    
    # Current letter and count
    count = sample_counts.get(current_letter, 0)
    
    if is_recording:
        # Recording mode
        status_color = COLOR_ORANGE
        status_text = f"RECORDING: {current_letter} | Frame: {frames_recorded}/{FRAMES_PER_SAMPLE}"
        zone_text = "Hold still..."
        
        # Progress bar
        progress = frames_recorded / FRAMES_PER_SAMPLE
        bar_width = 400
        cv2.rectangle(frame, (20, 75), (20 + bar_width, 90), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 75), (20 + int(bar_width * progress), 90), COLOR_ORANGE, -1)
    else:
        # Normal mode
        status_text = f"Selected: {current_letter} | Samples: {count}"
        
        if hand_in_zone:
            status_color = COLOR_GREEN
            zone_text = "READY - Press SPACE to record"
        else:
            status_color = COLOR_RED
            zone_text = "Move hand into signing zone"
    
    # Background
    cv2.rectangle(frame, (10, 10), (500, 100), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (500, 100), status_color, 2)
    
    # Text
    cv2.putText(frame, status_text, (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    cv2.putText(frame, zone_text, (20, 58),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
    
    # Settings indicators
    settings_text = f"Mirror: {'ON' if mirrored else 'OFF'} | Zone: {'ON' if show_zone else 'OFF'}"
    cv2.putText(frame, settings_text, (300, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)
    
    # Total samples (top right of status box)
    total = sum(sample_counts.values())
    cv2.putText(frame, f"Total: {total}", (420, 58),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
    
    return frame


# =============================================================================
# KEY MAPPING
# =============================================================================

def get_letter_from_key(key):
    """
    Map key press directly to letter.
    
    Args:
        key: Key code from cv2.waitKey()
    
    Returns:
        Uppercase letter if valid, None otherwise
    """
    # Convert to character
    try:
        char = chr(key).upper()
    except:
        return None
    
    # Check if it's a valid static letter
    if char in STATIC_LETTERS:
        return char
    
    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main data collection loop."""

    print(f"Output file: {OUTPUT_FILE}")
    print(f"Frames per sample: {FRAMES_PER_SAMPLE}")
    print(f"Features per frame: 63 (21 landmarks × 3 coordinates)")
    print(f"Static letters: {', '.join(STATIC_LETTERS)}")
    
    # Initialize MediaPipe
    print("\nInitializing detection...")
    mp_resources = detection.initialize_mediapipe()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Load existing data
    df = load_existing_data(OUTPUT_FILE)
    sample_counts = get_sample_counts(df)
    next_sample_id = get_next_sample_id(df)
    
    # State
    current_letter = STATIC_LETTERS[0]
    mirrored = True
    show_zone = True
    
    # Recording state
    is_recording = False
    recording_frames = []
    frames_recorded = 0
    
    # Session stats
    samples_this_session = 0
    
    print("\nReady! Controls:")
    print("  A-Y      : Select letter directly")
    print("  SPACE    : Record 100 frames")
    print("  1        : Quit and save")
    print("  2        : Toggle signing zone")
    print("  3        : Toggle mirror mode")
    print("  4        : Show guide for current letter")
    print("  5        : Reset dataset")
    print()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if mirrored:
            frame = cv2.flip(frame, 1)
        
        # Process frame (detect face and hands)
        frame, face_refs, hands_data = detection.process_frame(frame, mp_resources)
        
        # Draw signing zone
        if face_refs:
            frame = draw_signing_zone(frame, face_refs, show_zone)
        
        # Check if hand is in signing zone
        hand_in_zone = False
        current_hand_data = None
        
        if hands_data and face_refs:
            for hand_data in hands_data:
                rel_pos = hand_data.get('relative_position')
                if rel_pos and is_in_signing_zone(rel_pos):
                    hand_in_zone = True
                    current_hand_data = hand_data
                    break
        
        # Recording logic
        if is_recording:
            if hand_in_zone and current_hand_data:
                # Capture frame
                landmarks_norm = current_hand_data['landmarks_normalized']
                recording_frames.append(landmarks_norm)
                frames_recorded = len(recording_frames)
                
                # Check if recording complete
                if frames_recorded >= FRAMES_PER_SAMPLE:
                    # Save all frames
                    rows = []
                    for frame_id, landmarks in enumerate(recording_frames):
                        row = landmarks_to_row(landmarks, next_sample_id, frame_id, current_letter)
                        rows.append(row)
                    
                    # Add to DataFrame
                    new_rows = pd.DataFrame(rows, columns=create_column_names())
                    df = pd.concat([df, new_rows], ignore_index=True)
                    
                    # Update state
                    sample_counts[current_letter] = sample_counts.get(current_letter, 0) + 1
                    samples_this_session += 1
                    next_sample_id += 1
                    
                    print(f"Saved: {current_letter} (sample_id: {next_sample_id - 1}, total for letter: {sample_counts[current_letter]})")
                    
                    # Reset recording
                    is_recording = False
                    recording_frames = []
                    frames_recorded = 0
            else:
                # Hand left zone during recording - abort
                print("Recording aborted: Hand left signing zone")
                is_recording = False
                recording_frames = []
                frames_recorded = 0
        
        # Draw UI
        frame = draw_letter_selector(frame, current_letter, sample_counts)
        frame = draw_status(frame, current_letter, sample_counts, hand_in_zone, 
                           is_recording, frames_recorded, mirrored, show_zone)
        
        # Draw hand indicator
        if current_hand_data:
            center = current_hand_data['center_px']
            color = COLOR_GREEN if hand_in_zone else COLOR_RED
            cv2.circle(frame, center, 10, color, -1)
        
        cv2.imshow('NGT Data Collection', frame)
        
        # Handle input (only process non-recording keys when not recording)
        key = cv2.waitKey(1) & 0xFF
        
        # Key: 1 = Quit
        if key == ord('1'):
            if is_recording:
                print("Recording in progress, please wait...")
            else:
                break
        
        # Key: 2 = Toggle zone
        elif key == ord('2') and not is_recording:
            show_zone = not show_zone
            print(f"Signing zone: {'ON' if show_zone else 'OFF'}")
        
        # Key: 3 = Toggle mirror
        elif key == ord('3') and not is_recording:
            mirrored = not mirrored
            print(f"Mirror mode: {'ON' if mirrored else 'OFF'}")
        
        # Key: 4 = Show guide
        elif key == ord('4') and not is_recording:
            if show_guide:
                print(f"Showing guide for: {current_letter}")
                show_guide(current_letter)
            else:
                print("Guide not available (guide.py not found)")
        
        # Key: 5 = Reset dataset
        elif key == ord('5') and not is_recording:
            print("\n" + "=" * 60)
            print("WARNING: This will delete all collected data!")
            confirmation = input("Type 'reset' to confirm: ")
            if confirmation.strip().lower() == 'reset':
                df = reset_dataset(OUTPUT_FILE)
                sample_counts = {}
                next_sample_id = 0
                samples_this_session = 0
                print("Dataset reset successfully!")
            else:
                print("Reset cancelled.")
        
        # Key: SPACE = Start recording
        elif key == ord(' ') and not is_recording:
            if hand_in_zone and current_hand_data:
                is_recording = True
                recording_frames = []
                frames_recorded = 0
                print(f"Recording started for letter: {current_letter}")
            else:
                print("Cannot record: Hand not in signing zone")
        
        # Letter keys = Select letter
        elif not is_recording:
            new_letter = get_letter_from_key(key)
            if new_letter:
                current_letter = new_letter
                print(f"Selected letter: {current_letter}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    mp_resources['hands'].close()
    mp_resources['face_mesh'].close()
    
    # Save data
    if samples_this_session > 0:
        save_data(df, OUTPUT_FILE)
        print(f"\nSession complete: {samples_this_session} new samples collected")
    else:
        print("\nNo new samples collected")
    
    # Print summary
    print("\nSample counts per letter:")
    for letter in STATIC_LETTERS:
        count = sample_counts.get(letter, 0)
        bar = "█" * min(count, 20)
        print(f"  {letter}: {count:3d} {bar}")
    
    total_rows = len(df)
    total_samples = sum(sample_counts.values())
    print(f"\nTotal: {total_samples} samples ({total_rows} rows)")


if __name__ == "__main__":
    main()