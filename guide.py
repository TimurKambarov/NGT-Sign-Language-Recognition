"""
NGT Sign Language Recognition - Letter Guide
Visual guides for Dutch Sign Language (NGT) fingerspelling (static gestures only).

Usage:
    from guide import show_guide
    show_guide('A')  # Shows guide for letter A
"""

import cv2
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

GUIDE_SIZE = 400
BG_COLOR = (50, 50, 50)
TEXT_COLOR = (255, 255, 255)
STEP_COLOR = (200, 200, 200)
HIGHLIGHT_COLOR = (0, 255, 0)
ACCENT_COLOR = (255, 200, 0)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_guide_frame(letter):
    """Create base guide frame with title."""
    frame = np.zeros((GUIDE_SIZE, GUIDE_SIZE, 3), dtype=np.uint8)
    frame[:] = BG_COLOR
    
    cv2.putText(frame, f"How to sign: {letter}", (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_COLOR, 2)
    
    cv2.putText(frame, "Press any key to cycle through all gestures", (20, GUIDE_SIZE - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    cv2.putText(frame, "Press 'q' to close", (20, GUIDE_SIZE - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    return frame


def add_step(frame, step_num, text, y_pos):
    """Add a step instruction to the guide."""
    cv2.putText(frame, f"{step_num}. {text}", (30, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, STEP_COLOR, 1)


def draw_fist(frame, x, y, size=60):
    """Draw a simple fist icon."""
    cv2.ellipse(frame, (x, y), (size//2, size//2 + 10), 0, 0, 360, ACCENT_COLOR, 2)


def draw_hand_open(frame, x, y, size=60):
    """Draw an open hand icon."""
    # Palm
    cv2.ellipse(frame, (x, y + 20), (size//2, size//2), 0, 0, 360, ACCENT_COLOR, 2)
    # Fingers
    for i, offset in enumerate([-20, -10, 0, 10, 20]):
        finger_len = 35 if i != 0 else 25  # Thumb shorter
        cv2.line(frame, (x + offset, y), (x + offset, y - finger_len), ACCENT_COLOR, 2)


# =============================================================================
# LETTER GUIDES
# =============================================================================

def guide_A(frame):
    """A: Fist with thumb resting on side."""
    add_step(frame, 1, "Make a FIST", 100)
    add_step(frame, 2, "Thumb rests on SIDE of fist", 140)
    add_step(frame, 3, "Palm faces forward", 180)
    
    # Draw fist with thumb
    draw_fist(frame, 200, 280)
    cv2.line(frame, (160, 260), (160, 300), HIGHLIGHT_COLOR, 4)  # Thumb
    cv2.putText(frame, "thumb", (120, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.4, HIGHLIGHT_COLOR, 1)


def guide_B(frame):
    """B: Flat hand, fingers up, thumb tucked."""
    add_step(frame, 1, "Hand FLAT, fingers together", 100)
    add_step(frame, 2, "Fingers point UP", 140)
    add_step(frame, 3, "Thumb tucked across palm", 180)
    
    # Draw flat hand
    for i, offset in enumerate([-30, -10, 10, 30]):
        cv2.line(frame, (200 + offset, 320), (200 + offset, 240), ACCENT_COLOR, 3)
    cv2.rectangle(frame, (160, 320), (240, 360), ACCENT_COLOR, 2)
    cv2.line(frame, (160, 340), (140, 340), HIGHLIGHT_COLOR, 3)  # Thumb
    cv2.putText(frame, "thumb tucked", (100, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.4, HIGHLIGHT_COLOR, 1)


def guide_C(frame):
    """C: Curved hand forming C shape."""
    add_step(frame, 1, "Curve fingers and thumb", 100)
    add_step(frame, 2, "Form shape of letter C", 140)
    add_step(frame, 3, "Opening faces right", 180)
    
    # Draw C shape
    cv2.ellipse(frame, (200, 290), (50, 70), 0, 50, 310, ACCENT_COLOR, 4)
    cv2.putText(frame, "C shape", (260, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, HIGHLIGHT_COLOR, 1)


def guide_D(frame):
    """D: Index up, others touch thumb."""
    add_step(frame, 1, "INDEX finger points UP", 100)
    add_step(frame, 2, "Other fingers touch THUMB", 140)
    add_step(frame, 3, "Forms circle with thumb", 180)
    
    # Draw D shape
    cv2.line(frame, (200, 320), (200, 230), ACCENT_COLOR, 4)  # Index up
    cv2.ellipse(frame, (200, 340), (25, 20), 0, 0, 360, ACCENT_COLOR, 2)  # Circle
    cv2.putText(frame, "index", (210, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.4, HIGHLIGHT_COLOR, 1)
    cv2.putText(frame, "circle", (230, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.4, HIGHLIGHT_COLOR, 1)


def guide_E(frame):
    """E: Fingers curled, thumb across."""
    add_step(frame, 1, "Curl ALL fingers down", 100)
    add_step(frame, 2, "Fingertips touch palm", 140)
    add_step(frame, 3, "Thumb across fingertips", 180)
    
    # Draw curled hand
    cv2.ellipse(frame, (200, 290), (40, 50), 0, 0, 360, ACCENT_COLOR, 2)
    cv2.line(frame, (160, 270), (240, 270), HIGHLIGHT_COLOR, 3)  # Thumb across
    cv2.putText(frame, "curled", (250, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.4, ACCENT_COLOR, 1)
    cv2.putText(frame, "thumb", (250, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.4, HIGHLIGHT_COLOR, 1)


def guide_F(frame):
    """F: Thumb and index touch, others up."""
    add_step(frame, 1, "THUMB and INDEX touch", 100)
    add_step(frame, 2, "Form a circle/OK shape", 140)
    add_step(frame, 3, "Other 3 fingers UP", 180)
    
    # Draw F shape
    cv2.circle(frame, (180, 320), 20, HIGHLIGHT_COLOR, 2)  # Circle
    for i, offset in enumerate([0, 15, 30]):
        cv2.line(frame, (200 + offset, 310), (200 + offset, 240), ACCENT_COLOR, 3)
    cv2.putText(frame, "circle", (140, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.4, HIGHLIGHT_COLOR, 1)


def guide_G(frame):
    """G: Index and thumb point sideways."""
    add_step(frame, 1, "INDEX points to SIDE", 100)
    add_step(frame, 2, "THUMB parallel to index", 140)
    add_step(frame, 3, "Other fingers closed", 180)
    
    # Draw G shape
    cv2.line(frame, (150, 290), (280, 290), ACCENT_COLOR, 4)  # Index
    cv2.line(frame, (150, 310), (250, 310), HIGHLIGHT_COLOR, 3)  # Thumb
    cv2.putText(frame, "index", (285, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.4, ACCENT_COLOR, 1)
    cv2.putText(frame, "thumb", (255, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.4, HIGHLIGHT_COLOR, 1)


def guide_I(frame):
    """I: Pinky up, others closed."""
    add_step(frame, 1, "Make a FIST", 100)
    add_step(frame, 2, "Extend PINKY up", 140)
    add_step(frame, 3, "Thumb across fingers", 180)
    
    # Draw I shape
    draw_fist(frame, 190, 300)
    cv2.line(frame, (230, 310), (230, 230), HIGHLIGHT_COLOR, 4)  # Pinky
    cv2.putText(frame, "pinky", (240, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.4, HIGHLIGHT_COLOR, 1)


def guide_K(frame):
    """K: Index up, middle out, thumb between."""
    add_step(frame, 1, "INDEX finger UP", 100)
    add_step(frame, 2, "MIDDLE finger angled out", 140)
    add_step(frame, 3, "THUMB between them", 180)
    
    # Draw K shape
    cv2.line(frame, (180, 320), (180, 230), ACCENT_COLOR, 4)  # Index
    cv2.line(frame, (200, 320), (240, 250), ACCENT_COLOR, 4)  # Middle angled
    cv2.line(frame, (175, 290), (210, 280), HIGHLIGHT_COLOR, 3)  # Thumb
    cv2.putText(frame, "thumb between", (220, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.4, HIGHLIGHT_COLOR, 1)


def guide_L(frame):
    """L: L-shape with index and thumb."""
    add_step(frame, 1, "INDEX finger UP", 100)
    add_step(frame, 2, "THUMB out to side (90°)", 140)
    add_step(frame, 3, "Forms L shape", 180)
    
    # Draw L shape
    cv2.line(frame, (200, 340), (200, 240), ACCENT_COLOR, 4)  # Index up
    cv2.line(frame, (200, 340), (280, 340), ACCENT_COLOR, 4)  # Thumb out
    cv2.putText(frame, "L", (220, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, HIGHLIGHT_COLOR, 2)


def guide_M(frame):
    """M: Three fingers over thumb."""
    add_step(frame, 1, "Thumb UNDER 3 fingers", 100)
    add_step(frame, 2, "Index, middle, ring over thumb", 140)
    add_step(frame, 3, "Pinky tucked in", 180)
    
    # Draw M shape
    for i, offset in enumerate([-25, 0, 25]):
        cv2.line(frame, (200 + offset, 280), (200 + offset, 330), ACCENT_COLOR, 4)
    cv2.line(frame, (170, 340), (240, 340), HIGHLIGHT_COLOR, 3)  # Thumb under
    cv2.putText(frame, "3 fingers", (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.4, ACCENT_COLOR, 1)
    cv2.putText(frame, "thumb under", (250, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.4, HIGHLIGHT_COLOR, 1)


def guide_N(frame):
    """N: Two fingers over thumb."""
    add_step(frame, 1, "Thumb UNDER 2 fingers", 100)
    add_step(frame, 2, "Index and middle over thumb", 140)
    add_step(frame, 3, "Ring and pinky tucked", 180)
    
    # Draw N shape
    for i, offset in enumerate([-15, 15]):
        cv2.line(frame, (200 + offset, 280), (200 + offset, 330), ACCENT_COLOR, 4)
    cv2.line(frame, (180, 340), (230, 340), HIGHLIGHT_COLOR, 3)  # Thumb under
    cv2.putText(frame, "2 fingers", (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.4, ACCENT_COLOR, 1)


def guide_O(frame):
    """O: All fingers touch thumb, forming O."""
    add_step(frame, 1, "ALL fingertips touch thumb", 100)
    add_step(frame, 2, "Form round O shape", 140)
    add_step(frame, 3, "Like holding small ball", 180)
    
    # Draw O shape
    cv2.ellipse(frame, (200, 290), (45, 55), 0, 0, 360, ACCENT_COLOR, 3)
    cv2.putText(frame, "O", (185, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, HIGHLIGHT_COLOR, 2)


def guide_P(frame):
    """P: Like K but pointing down."""
    add_step(frame, 1, "Same as K hand shape", 100)
    add_step(frame, 2, "But pointing DOWN", 140)
    add_step(frame, 3, "Wrist bent forward", 180)
    
    # Draw P shape (K pointing down)
    cv2.line(frame, (180, 250), (180, 340), ACCENT_COLOR, 4)  # Index down
    cv2.line(frame, (200, 250), (240, 320), ACCENT_COLOR, 4)  # Middle angled
    cv2.arrowedLine(frame, (200, 220), (200, 260), HIGHLIGHT_COLOR, 2)
    cv2.putText(frame, "point down", (220, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.4, HIGHLIGHT_COLOR, 1)


def guide_Q(frame):
    """Q: Like G but pointing down."""
    add_step(frame, 1, "Same as G hand shape", 100)
    add_step(frame, 2, "But pointing DOWN", 140)
    add_step(frame, 3, "Index and thumb down", 180)
    
    # Draw Q shape
    cv2.line(frame, (200, 240), (200, 340), ACCENT_COLOR, 4)  # Index down
    cv2.line(frame, (180, 250), (180, 320), HIGHLIGHT_COLOR, 3)  # Thumb
    cv2.arrowedLine(frame, (220, 250), (220, 290), HIGHLIGHT_COLOR, 2)
    cv2.putText(frame, "point down", (230, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.4, HIGHLIGHT_COLOR, 1)


def guide_R(frame):
    """R: Index and middle crossed."""
    add_step(frame, 1, "INDEX and MIDDLE up", 100)
    add_step(frame, 2, "CROSS middle over index", 140)
    add_step(frame, 3, "Other fingers closed", 180)
    
    # Draw crossed fingers
    cv2.line(frame, (190, 330), (180, 240), ACCENT_COLOR, 4)  # Index
    cv2.line(frame, (200, 330), (220, 240), ACCENT_COLOR, 4)  # Middle
    cv2.putText(frame, "crossed", (230, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.4, HIGHLIGHT_COLOR, 1)


def guide_S(frame):
    """S: Fist with thumb over fingers."""
    add_step(frame, 1, "Make a FIST", 100)
    add_step(frame, 2, "Thumb OVER fingers", 140)
    add_step(frame, 3, "Thumb across front", 180)
    
    # Draw S shape
    draw_fist(frame, 200, 290)
    cv2.line(frame, (160, 280), (240, 280), HIGHLIGHT_COLOR, 4)  # Thumb across
    cv2.putText(frame, "thumb over", (250, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.4, HIGHLIGHT_COLOR, 1)


def guide_T(frame):
    """T: Fist with thumb between index and middle."""
    add_step(frame, 1, "Make a FIST", 100)
    add_step(frame, 2, "THUMB pokes between", 140)
    add_step(frame, 3, "Between index and middle", 180)
    
    # Draw T shape
    draw_fist(frame, 200, 290)
    cv2.line(frame, (200, 260), (200, 230), HIGHLIGHT_COLOR, 4)  # Thumb poking up
    cv2.putText(frame, "thumb pokes", (210, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.4, HIGHLIGHT_COLOR, 1)


def guide_U(frame):
    """U: Index and middle up together."""
    add_step(frame, 1, "INDEX and MIDDLE up", 100)
    add_step(frame, 2, "Fingers TOGETHER", 140)
    add_step(frame, 3, "Other fingers closed", 180)
    
    # Draw U shape
    cv2.line(frame, (190, 330), (190, 240), ACCENT_COLOR, 4)  # Index
    cv2.line(frame, (210, 330), (210, 240), ACCENT_COLOR, 4)  # Middle
    cv2.putText(frame, "together", (220, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.4, HIGHLIGHT_COLOR, 1)


def guide_V(frame):
    """V: Index and middle up, spread apart."""
    add_step(frame, 1, "INDEX and MIDDLE up", 100)
    add_step(frame, 2, "Fingers SPREAD (V shape)", 140)
    add_step(frame, 3, "Peace/victory sign", 180)
    
    # Draw V shape
    cv2.line(frame, (200, 330), (160, 240), ACCENT_COLOR, 4)  # Index
    cv2.line(frame, (200, 330), (240, 240), ACCENT_COLOR, 4)  # Middle
    cv2.putText(frame, "V", (190, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, HIGHLIGHT_COLOR, 2)


def guide_W(frame):
    """W: Index, middle, ring up and spread."""
    add_step(frame, 1, "THREE fingers up", 100)
    add_step(frame, 2, "Index, middle, ring", 140)
    add_step(frame, 3, "Spread apart like W", 180)
    
    # Draw W shape
    cv2.line(frame, (200, 330), (150, 240), ACCENT_COLOR, 4)
    cv2.line(frame, (200, 330), (200, 240), ACCENT_COLOR, 4)
    cv2.line(frame, (200, 330), (250, 240), ACCENT_COLOR, 4)
    cv2.putText(frame, "W", (180, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, HIGHLIGHT_COLOR, 2)


def guide_X(frame):
    """X: Index bent like hook."""
    add_step(frame, 1, "INDEX finger up", 100)
    add_step(frame, 2, "BEND index (hook shape)", 140)
    add_step(frame, 3, "Other fingers closed", 180)
    
    # Draw X shape (hooked finger)
    cv2.line(frame, (200, 330), (200, 280), ACCENT_COLOR, 4)
    cv2.line(frame, (200, 280), (230, 260), ACCENT_COLOR, 4)  # Hook
    cv2.putText(frame, "hook", (240, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.4, HIGHLIGHT_COLOR, 1)


def guide_Y(frame):
    """Y: Thumb and pinky out."""
    add_step(frame, 1, "THUMB out to side", 100)
    add_step(frame, 2, "PINKY out to side", 140)
    add_step(frame, 3, "Other 3 fingers closed", 180)
    
    # Draw Y shape
    cv2.line(frame, (200, 300), (130, 260), HIGHLIGHT_COLOR, 4)  # Thumb
    cv2.line(frame, (200, 300), (270, 260), ACCENT_COLOR, 4)  # Pinky
    draw_fist(frame, 200, 320)
    cv2.putText(frame, "thumb", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.4, HIGHLIGHT_COLOR, 1)
    cv2.putText(frame, "pinky", (260, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.4, ACCENT_COLOR, 1)


# Dynamic letters (for future use)
def guide_H(frame):
    """H: V-sign with bouncing motion."""
    add_step(frame, 1, "Make V-sign (peace sign)", 100)
    add_step(frame, 2, "Bounce hand UP and DOWN", 140)
    add_step(frame, 3, "Keep other fingers closed", 180)
    
    cv2.arrowedLine(frame, (200, 250), (200, 200), HIGHLIGHT_COLOR, 3)
    cv2.arrowedLine(frame, (200, 280), (200, 330), HIGHLIGHT_COLOR, 3)
    cv2.putText(frame, "UP", (220, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, HIGHLIGHT_COLOR, 1)
    cv2.putText(frame, "DOWN", (220, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, HIGHLIGHT_COLOR, 1)
    cv2.putText(frame, "[DYNAMIC]", (140, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)


def guide_J(frame):
    """J: Pinky draws J shape."""
    add_step(frame, 1, "Make fist with PINKY out", 100)
    add_step(frame, 2, "Move DOWN", 140)
    add_step(frame, 3, "Then CURVE left/inward", 180)
    
    pts = [(200, 220), (200, 300), (180, 340), (140, 340)]
    for i in range(len(pts)-1):
        cv2.line(frame, pts[i], pts[i+1], (255, 0, 255), 3)
    cv2.circle(frame, pts[0], 8, (255, 0, 255), -1)
    cv2.putText(frame, "START", (210, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    cv2.putText(frame, "[DYNAMIC]", (140, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)


def guide_Z(frame):
    """Z: Index draws Z shape."""
    add_step(frame, 1, "Point with INDEX finger", 100)
    add_step(frame, 2, "Draw Z: right, diagonal, right", 140)
    add_step(frame, 3, "Keep other fingers closed", 180)
    
    pts = [(120, 230), (280, 230), (120, 340), (280, 340)]
    cv2.line(frame, pts[0], pts[1], HIGHLIGHT_COLOR, 3)
    cv2.line(frame, pts[1], pts[2], HIGHLIGHT_COLOR, 3)
    cv2.line(frame, pts[2], pts[3], HIGHLIGHT_COLOR, 3)
    cv2.circle(frame, pts[0], 8, HIGHLIGHT_COLOR, -1)
    cv2.putText(frame, "1", (100, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, HIGHLIGHT_COLOR, 1)
    cv2.putText(frame, "2", (285, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, HIGHLIGHT_COLOR, 1)
    cv2.putText(frame, "3", (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, HIGHLIGHT_COLOR, 1)
    cv2.putText(frame, "[DYNAMIC]", (140, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

# Map letters to their guide functions
GUIDE_FUNCTIONS = {
    'A': guide_A, 'B': guide_B, 'C': guide_C, 'D': guide_D, 'E': guide_E,
    'F': guide_F, 'G': guide_G, 'H': guide_H, 'I': guide_I, 'J': guide_J,
    'K': guide_K, 'L': guide_L, 'M': guide_M, 'N': guide_N, 'O': guide_O,
    'P': guide_P, 'Q': guide_Q, 'R': guide_R, 'S': guide_S, 'T': guide_T,
    'U': guide_U, 'V': guide_V, 'W': guide_W, 'X': guide_X, 'Y': guide_Y,
    'Z': guide_Z
}


def show_guide(letter):
    """
    Show visual guide for how to sign a letter in NGT.
    
    Args:
        letter: Single uppercase letter (A-Z)
    """
    letter = letter.upper()
    
    if letter not in GUIDE_FUNCTIONS:
        print(f"No guide available for letter: {letter}")
        return
    
    frame = create_guide_frame(letter)
    GUIDE_FUNCTIONS[letter](frame)
    
    cv2.imshow("NGT Guide", frame)
    cv2.waitKey(0)
    cv2.destroyWindow("NGT Guide")


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("NGT Letter Guide - Test Mode")
    print("Press any key to cycle through letters, 'q' to quit")
    
    for letter in "ABCDEFGIKLMNOPQRSTUVWXY":
        print(f"Showing guide for: {letter}")
        frame = create_guide_frame(letter)
        GUIDE_FUNCTIONS[letter](frame)
        cv2.imshow("NGT Guide", frame)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()