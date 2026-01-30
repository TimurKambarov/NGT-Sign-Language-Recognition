'''
Guide to help alphabet fingerspelling
'''
import cv2
import os

# Path to gesture images folder
IMAGES_DIR = "data/NGT_gestures"

ALL_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DYNAMIC_LETTERS = ['H', 'J', 'Z']

def show_guide(letter):
    """
    Show reference image for how to sign a letter in NGT
    """
    letter = letter.upper()
    new_dim = (300, 400) 
    
    if letter not in ALL_LETTERS:
        print(f"Invalid letter: {letter}")
        return
    
    # Construct image path
    image_path = os.path.join(IMAGES_DIR, f"{letter}.jpg")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print(f"Please ensure {letter}.jpg exists in {IMAGES_DIR}/")
        return
    
    # Load image
    image = cv2.imread(image_path)
    image = cv2.resize(image, new_dim, interpolation=cv2.INTER_CUBIC)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Add information overlay
    h, w, _ = image.shape
    
    # Add instruction at bottom
    instruction = "Press any key to close"
    cv2.putText(image, instruction, (20, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Display
    cv2.imshow(f"NGT Guide - Letter {letter}", image)
    cv2.waitKey(0)
    cv2.destroyWindow(f"NGT Guide - Letter {letter}")


if __name__ == "__main__":
    print("NGT Letter Guide - Test Mode")
    print("Press any key to cycle through letters, 'q' to quit")
    
    
    # Test all letters
    for letter in ALL_LETTERS:
        image_path = os.path.join(IMAGES_DIR, f"{letter}.jpg")
        if os.path.exists(image_path):
            print(f"Showing guide for: {letter}")
            show_guide(letter)
            
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            if key == ord('q'):
                break
        else:
            print(f"Missing: {image_path}")
    