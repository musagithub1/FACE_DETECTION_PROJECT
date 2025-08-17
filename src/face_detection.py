import cv2
import os
import numpy as np
import argparse
import sys

class FaceDetector:
    def __init__(self, data_dir=None):
        """
        Initialize the face detector with the necessary cascade files
        
        Args:
            data_dir: Directory containing the Haar cascade XML files. If None, uses relative path.
        """
        # If no data directory is specified, use a relative path approach
        if data_dir is None:
            # Get the directory where the script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to the project root
            project_root = os.path.dirname(script_dir)
            # Set data directory relative to project root
            data_dir = os.path.join(project_root, 'data')
        
        self.data_dir = data_dir
        print(f"Looking for cascade files in: {data_dir}")
        
        # Load the face cascade
        face_cascade_path = os.path.join(data_dir, 'haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # Load the eye cascade
        eye_cascade_path = os.path.join(data_dir, 'haarcascade_eye.xml')
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        
        # Load the smile cascade for expression detection
        smile_cascade_path = os.path.join(data_dir, 'haarcascade_smile.xml')
        self.smile_cascade = cv2.CascadeClassifier(smile_cascade_path)
        
        # Check if cascades loaded successfully
        if self.face_cascade.empty():
            raise ValueError(f"Error loading face cascade from {face_cascade_path}")
        if self.eye_cascade.empty():
            raise ValueError(f"Error loading eye cascade from {eye_cascade_path}")
        if self.smile_cascade.empty():
            raise ValueError(f"Error loading smile cascade from {smile_cascade_path}")
            
        print("All cascade classifiers loaded successfully")
    
    def detect_faces(self, img):
        """
        Detect faces in an image
        
        Args:
            img: Input image
            
        Returns:
            Processed image with face detection visualization
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Draw rectangles around faces and detect features
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Region of interest for face
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            # Detect smiles (for expression recognition)
            smiles = self.smile_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.7,
                minNeighbors=22,
                minSize=(25, 25)
            )
            
            # If smile detected, label as "Happy"
            if len(smiles) > 0:
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 1)
                cv2.putText(img, "Happy", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(img, "Neutral", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        
        # Display count of faces
        cv2.putText(img, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return img
    
    def process_image(self, image_path):
        """
        Process a single image file
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Processed image
        """
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Detect faces
        result_img = self.detect_faces(img)
        
        return result_img
    
    def process_webcam(self):
        """
        Process video from webcam with real-time face detection
        """
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise ValueError("Could not open webcam")
        
        print("Webcam initialized. Press 'q' to quit.")
        
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            
            # Detect faces
            result_frame = self.detect_faces(frame)
            
            # Display the result
            cv2.imshow('Face Detection', result_frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()


def main():
    """
    Main function to parse arguments and run face detection
    """
    parser = argparse.ArgumentParser(description='Face Detection using OpenCV')
    parser.add_argument('--image', help='Path to image file')
    parser.add_argument('--webcam', action='store_true', help='Use webcam for real-time detection')
    parser.add_argument('--output', help='Path to save output image (only for image mode)')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode without GUI')
    parser.add_argument('--data-dir', help='Custom directory containing Haar cascade XML files')
    
    args = parser.parse_args()
    
    # Initialize face detector with optional custom data directory
    detector = FaceDetector(args.data_dir)
    
    if args.webcam:
        if args.headless:
            print("Webcam mode cannot be used in headless mode")
            return
        # Process webcam feed
        detector.process_webcam()
    elif args.image:
        # Process single image
        result_img = detector.process_image(args.image)
        
        # Display the result if not in headless mode
        if not args.headless:
            cv2.imshow('Face Detection Result', result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Save output if specified
        if args.output:
            cv2.imwrite(args.output, result_img)
            print(f"Result saved to {args.output}")
    else:
        print("Error: Please specify either --image or --webcam")
        parser.print_help()


if __name__ == "__main__":
    main()
