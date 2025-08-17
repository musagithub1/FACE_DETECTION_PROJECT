import cv2
import os
import numpy as np
import argparse
import sys
import random

# Attempt to import GUI libraries, but allow fallback for headless environments
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import Image, ImageTk
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("Warning: Tkinter or Pillow not found. GUI mode will be unavailable.")

class FaceAnalysisApp:
    def __init__(self, data_dir=None, models_dir=None):
        """
        Initialize the face analysis application with necessary cascade files and models
        
        Args:
            data_dir: Directory containing the Haar cascade XML files. If None, uses relative path.
            models_dir: Directory containing age and gender models. If None, uses relative path.
        """
        # If no data directory is specified, use a relative path approach
        if data_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            data_dir = os.path.join(project_root, 'data')
        
        # If no models directory is specified, use a relative path approach
        if models_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            models_dir = os.path.join(project_root, 'models')
        
        self.data_dir = data_dir
        self.models_dir = models_dir
        
        print(f"Looking for cascade files in: {data_dir}")
        print(f"Looking for models in: {models_dir}")
        
        # Load cascades
        self.face_cascade = self._load_cascade('haarcascade_frontalface_default.xml')
        self.eye_cascade = self._load_cascade('haarcascade_eye.xml')
        self.smile_cascade = self._load_cascade('haarcascade_smile.xml')
        self.glasses_cascade = self._load_cascade('haarcascade_eye_tree_eyeglasses.xml') # For glasses detection
        
        # Load age and gender models
        self.age_net, self.gender_net = self._load_dnn_models()
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.gender_list = ['Male', 'Female']
        
        # Feedback and recommendations
        self.emotion_feedback = {
            "happy": [
                "You seem happy today! That's great!",
                "Your smile brightens the room!",
                "Happiness looks good on you!",
                "Keep smiling, it suits you well!"
            ],
            "neutral": [
                "How are you feeling today?",
                "Take a moment to breathe and relax.",
                "Remember to take breaks throughout your day.",
                "Sometimes a neutral face means deep thinking!"
            ]
        }
        self.haircut_recommendations = {
            "oval": {
                "Male": "Most hairstyles work well with your oval face shape. Try a classic short back and sides with some length on top.",
                "Female": "Your oval face shape suits most hairstyles. Consider layers that frame your face or a medium-length cut."
            },
            "round": {
                "Male": "Styles with height on top and shorter sides help elongate your round face. Try a pompadour or textured quiff.",
                "Female": "Long layers past the chin or an asymmetrical bob can help elongate your round face shape."
            },
            "square": {
                "Male": "Your strong jawline works well with short sides and textured top. Consider a crew cut or textured crop.",
                "Female": "Soft layers, side-swept bangs, or a lob (long bob) can soften your square jawline."
            },
            "unknown": {
                "Male": "A versatile medium-length cut with some texture would be a good choice for your face.",
                "Female": "A shoulder-length cut with layers would frame your face nicely and offer styling versatility."
            }
        }
        
        # Face recognition data (simple implementation)
        self.known_faces = {} # {name: encoding}
        self.next_person_id = 1
        
        # Face tracking data
        self.trackers = []
        self.face_trajectories = {}
        self.frame_count = 0
        
        print("All required models and cascades loaded successfully")

    def _load_cascade(self, filename):
        cascade_path = os.path.join(self.data_dir, filename)
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            raise ValueError(f"Error loading cascade from {cascade_path}")
        return cascade

    def _load_dnn_models(self):
        age_net, gender_net = None, None
        try:
            age_proto = os.path.join(self.models_dir, 'age_deploy.prototxt')
            age_model = os.path.join(self.models_dir, 'age_net.caffemodel')
            age_net = cv2.dnn.readNet(age_proto, age_model)
            
            gender_proto = os.path.join(self.models_dir, 'gender_deploy.prototxt')
            gender_model = os.path.join(self.models_dir, 'gender_net.caffemodel')
            gender_net = cv2.dnn.readNet(gender_proto, gender_model)
            print("Age and gender models loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load age and gender models: {e}")
        return age_net, gender_net

    # --- Feature Implementations --- 
    
    def detect_glasses(self, roi_gray):
        # Simple check using eye region
        glasses = self.glasses_cascade.detectMultiScale(roi_gray, 1.1, 10)
        return len(glasses) > 0

    def analyze_symmetry(self, face_img):
        h, w = face_img.shape[:2]
        if w < 20: return 0 # Too small to analyze
        
        mid = w // 2
        left_half = face_img[:, :mid]
        right_half = cv2.flip(face_img[:, mid:], 1)
        
        # Ensure halves have same width for comparison
        min_w = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_w]
        right_half = right_half[:, :min_w]
        
        # Calculate difference
        diff = cv2.absdiff(left_half, right_half)
        similarity = 1 - (np.sum(diff) / (left_half.size * 255))
        return round(similarity * 100, 1)

    def apply_filter(self, face_img, filter_type):
        if filter_type == 'sepia':
            kernel = np.array([[0.272, 0.534, 0.131],
                               [0.349, 0.686, 0.168],
                               [0.393, 0.769, 0.189]])
            sepia_img = cv2.transform(face_img, kernel)
            sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
            return sepia_img
        elif filter_type == 'bw':
            bw_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(bw_img, cv2.COLOR_GRAY2BGR) # Convert back to BGR for consistency
        elif filter_type == 'cartoon':
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
            color = cv2.bilateralFilter(face_img, 9, 300, 300)
            cartoon = cv2.bitwise_and(color, color, mask=edges)
            return cartoon
        return face_img # No filter or unknown filter

    def track_faces(self, frame, gray, faces):
        # Initialize trackers for new faces
        if self.frame_count % 10 == 0: # Re-detect faces periodically
            self.trackers = []
            self.face_trajectories = {} # Reset trajectories on re-detection
            for (x, y, w, h) in faces:
                try:
                    # Use a more robust tracker if available, CSRT is good but can be slow
                    tracker = cv2.TrackerCSRT_create()
                    # tracker = cv2.TrackerKCF_create() # Alternative tracker
                    tracker.init(frame, (x, y, w, h))
                    self.trackers.append(tracker)
                except Exception as e:
                    print(f"Warning: Could not initialize tracker: {e}")
        else:
            # Update existing trackers
            updated_faces = []
            new_trackers = []
            lost_tracker_ids = set(self.face_trajectories.keys())
            
            for tracker in self.trackers:
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    updated_faces.append((x, y, w, h))
                    new_trackers.append(tracker)
                    
                    # Add to trajectory
                    center = (x + w//2, y + h//2)
                    tracker_id = id(tracker)
                    if tracker_id not in self.face_trajectories:
                        self.face_trajectories[tracker_id] = []
                    self.face_trajectories[tracker_id].append(center)
                    if len(self.face_trajectories[tracker_id]) > 50: # Limit trajectory length
                        self.face_trajectories[tracker_id].pop(0)
                    lost_tracker_ids.discard(tracker_id)
            
            self.trackers = new_trackers # Keep only successful trackers
            faces = updated_faces # Use tracked faces for this frame
            
            # Remove trajectories for lost trackers
            for lost_id in lost_tracker_ids:
                if lost_id in self.face_trajectories:
                    del self.face_trajectories[lost_id]
            
        # Draw trajectories
        for tracker_id, trajectory in self.face_trajectories.items():
            if len(trajectory) > 1:
                for i in range(1, len(trajectory)):
                    cv2.line(frame, trajectory[i-1], trajectory[i], (0, 255, 0), 2)
                
        self.frame_count += 1
        return faces

    def compare_faces(self, face1_img, face2_img):
        # Simple comparison using ORB features (replace with a proper recognition model for better results)
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(face1_img, None)
        kp2, des2 = orb.detectAndCompute(face2_img, None)
        
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 0.0 # Not enough features
            
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Calculate similarity score based on good matches
        good_matches = [m for m in matches if m.distance < 70] # Adjust threshold as needed
        similarity = len(good_matches) / max(len(kp1), len(kp2)) if max(len(kp1), len(kp2)) > 0 else 0
        return round(similarity * 100, 1)

    def detect_smile_intensity(self, roi_gray):
        smiles = self.smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25))
        intensity = 0
        if len(smiles) > 0:
            # Simple intensity based on number/size of detected smiles
            intensity = min(len(smiles), 3) # Cap intensity at 3 levels
            # Could also use average size of smile detections relative to face size
        
        levels = {0: "Neutral", 1: "Slight Smile", 2: "Moderate Smile", 3: "Broad Smile"}
        return levels.get(intensity, "Neutral"), intensity

    def correct_orientation(self, img, face_rect):
        x, y, w, h = face_rect
        # Ensure ROI is valid
        if h <= 0 or w <= 0:
            return img, 0
        face_roi = img[y:y+h, x:x+w]
        if face_roi.size == 0:
             return img, 0
             
        roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])
            left_eye_center = (eyes[0][0] + eyes[0][2]//2, eyes[0][1] + eyes[0][3]//2)
            right_eye_center = (eyes[1][0] + eyes[1][2]//2, eyes[1][1] + eyes[1][3]//2)
            
            dy = right_eye_center[1] - left_eye_center[1]
            dx = right_eye_center[0] - left_eye_center[0]
            if dx == 0: dx = 1e-6 # Avoid division by zero
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Rotate the whole image to align eyes horizontally
            img_h, img_w = img.shape[:2]
            center = (img_w // 2, img_h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_img = cv2.warpAffine(img, M, (img_w, img_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated_img, angle
        return img, 0 # Return original if eyes not detected properly

    def recognize_face(self, face_encoding):
        # Simple recognition: find closest known face encoding
        # Note: This is a very basic implementation. Use a dedicated library for real-world recognition.
        if face_encoding is None:
            return "Unknown"
            
        min_dist = float('inf')
        recognized_name = "Unknown"
        
        for name, known_encoding in self.known_faces.items():
            if known_encoding is None: continue
            dist = np.linalg.norm(face_encoding - known_encoding)
            if dist < min_dist and dist < 0.6: # Threshold for recognition
                min_dist = dist
                recognized_name = name
                
        if recognized_name == "Unknown":
            # Add as new person if unknown
            recognized_name = f"Person_{self.next_person_id}"
            self.known_faces[recognized_name] = face_encoding
            self.next_person_id += 1
            
        return recognized_name

    def get_face_encoding(self, face_img):
        # Simple encoding using average color (replace with actual feature extraction)
        # For real recognition, use libraries like face_recognition or deepface's embedding models
        if face_img is None or face_img.size == 0:
            return None
        # Resize for consistent encoding size
        try:
            resized_face = cv2.resize(face_img, (100, 100))
            # Use average pixel value as a very basic encoding
            encoding = np.mean(resized_face, axis=(0, 1))
            return encoding / 255.0 # Normalize
        except cv2.error as e:
            print(f"Warning: Could not resize face for encoding: {e}")
            return None

    def blur_face(self, img, face_rect):
        x, y, w, h = face_rect
        if w <= 0 or h <= 0: return img # Invalid rectangle
        face_roi = img[y:y+h, x:x+w]
        if face_roi.size == 0: return img # Empty ROI
        # Apply Gaussian blur
        # Ensure kernel size is odd and positive
        ksize = (max(1, w//5) | 1, max(1, h//5) | 1) 
        blurred_face = cv2.GaussianBlur(face_roi, ksize, 30)
        img[y:y+h, x:x+w] = blurred_face
        return img

    def measure_features(self, face_img, face_rect):
        x, y, w, h = face_rect
        if w <= 0 or h <= 0: return {} # Invalid rectangle
        face_roi = face_img[y:y+h, x:x+w]
        if face_roi.size == 0: return {} # Empty ROI
        
        roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(roi_gray)
        measurements = {}
        
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])
            left_eye = eyes[0]
            right_eye = eyes[1]
            
            # Eye distance (between centers)
            left_center = (left_eye[0] + left_eye[2]//2, left_eye[1] + left_eye[3]//2)
            right_center = (right_eye[0] + right_eye[2]//2, right_eye[1] + right_eye[3]//2)
            eye_dist = np.linalg.norm(np.array(left_center) - np.array(right_center))
            measurements['eye_distance'] = round(eye_dist, 1)
            
        # Could add nose length, mouth width etc. if landmarks were available
        # Using simple estimates based on face box for now
        measurements['face_width'] = w
        measurements['face_height'] = h
            
        return measurements

    def estimate_face_shape(self, face_img):
        """Estimates face shape based on width-to-height ratio."""
        if face_img is None or face_img.size == 0:
            return "unknown"
        h, w = face_img.shape[:2]
        if h == 0: return "unknown"
        ratio = w / h
        
        if ratio > 0.95:
            return "round"
        elif ratio < 0.85:
            # Could be square or oblong, distinction needs more features
            return "square/oblong"
        else:
            return "oval"
            
    def estimate_head_pose(self, face_img, face_rect):
        """Estimates head pose (left/right/straight, tilt) based on eye positions."""
        x, y, w, h = face_rect
        if w <= 0 or h <= 0: return "Unknown Pose"
        face_roi = face_img[y:y+h, x:x+w]
        if face_roi.size == 0: return "Unknown Pose"
        
        roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(roi_gray)
        
        pose = "Straight"
        tilt = "Level"
        
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0]) # Sort by x-coordinate
            left_eye = eyes[0]
            right_eye = eyes[1]
            
            # Horizontal Pose (Left/Right)
            face_center_x = w / 2
            left_eye_center_x = left_eye[0] + left_eye[2] / 2
            right_eye_center_x = right_eye[0] + right_eye[2] / 2
            eye_midpoint_x = (left_eye_center_x + right_eye_center_x) / 2
            
            # Thresholds for determining pose (adjust as needed)
            if eye_midpoint_x < face_center_x * 0.85:
                pose = "Looking Right"
            elif eye_midpoint_x > face_center_x * 1.15:
                pose = "Looking Left"
            else:
                pose = "Straight"
                
            # Tilt Estimation
            left_eye_center_y = left_eye[1] + left_eye[3] / 2
            right_eye_center_y = right_eye[1] + right_eye[3] / 2
            delta_y = abs(left_eye_center_y - right_eye_center_y)
            eye_dist = np.linalg.norm(np.array([left_eye_center_x, left_eye_center_y]) - np.array([right_eye_center_x, right_eye_center_y]))
            if eye_dist > 0:
                tilt_ratio = delta_y / eye_dist
                if tilt_ratio > 0.1: # Threshold for significant tilt
                    tilt = "Tilted Left" if left_eye_center_y < right_eye_center_y else "Tilted Right"
                else:
                    tilt = "Level"
                    
        return f"{pose}, {tilt}"

    # --- Core Processing Logic --- 

    def process_frame(self, frame, apply_filter=None, blur_faces=False, recognize=False, track=False):
        """
        Process a single frame (from image or video)
        """
        if frame is None: return None, []
        processed_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Apply tracking if enabled
        if track:
            faces = self.track_faces(processed_frame, gray, faces)
        
        all_face_data = []
        
        for i, (x, y, w, h) in enumerate(faces):
            # Ensure coordinates are valid
            if w <= 0 or h <= 0 or x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
                print(f"Warning: Invalid face rectangle detected: {(x, y, w, h)}")
                continue
                
            face_data = {'rect': (x, y, w, h)}
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = processed_frame[y:y+h, x:x+w]
            
            # Check if ROI is valid
            if roi_gray.size == 0 or roi_color.size == 0:
                print(f"Warning: Empty ROI for face at {(x, y, w, h)}")
                continue
            
            # Apply filter if specified
            if apply_filter:
                roi_color = self.apply_filter(roi_color, apply_filter)
                processed_frame[y:y+h, x:x+w] = roi_color # Update frame with filtered face
            
            # Blur face if specified
            if blur_faces:
                processed_frame = self.blur_face(processed_frame, (x, y, w, h))
                # Skip further analysis if blurred
                cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 0, 0), 2)
                cv2.putText(processed_frame, "Blurred", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                all_face_data.append(face_data)
                continue # Skip analysis for blurred faces
                
            # Draw rectangle around face
            cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # --- Run Analyses --- 
            try:
                face_data['glasses'] = self.detect_glasses(roi_gray)
                face_data['symmetry'] = self.analyze_symmetry(roi_color)
                face_data['emotion'], face_data['smile_intensity'] = self.detect_smile_intensity(roi_gray)
                face_data['measurements'] = self.measure_features(processed_frame, (x, y, w, h)) # Use processed_frame for measurements
                face_data['face_shape'] = self.estimate_face_shape(roi_color)
                face_data['head_pose'] = self.estimate_head_pose(roi_color, (x, y, w, h))
                
                # Age and Gender
                if self.age_net and self.gender_net:
                    blob = cv2.dnn.blobFromImage(roi_color, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
                    self.gender_net.setInput(blob)
                    gender_preds = self.gender_net.forward()
                    face_data['gender'] = self.gender_list[gender_preds[0].argmax()]
                    
                    self.age_net.setInput(blob)
                    age_preds = self.age_net.forward()
                    face_data['age'] = self.age_list[age_preds[0].argmax()]
                else:
                    face_data['gender'] = "N/A"
                    face_data['age'] = "N/A"
                    
                # Recognition
                if recognize:
                    encoding = self.get_face_encoding(roi_color)
                    face_data['name'] = self.recognize_face(encoding)
                
                # --- Display Info --- 
                info_y = y - 10
                def put_text(text, color=(255, 0, 0)):
                    nonlocal info_y
                    # Ensure text position is within frame bounds
                    text_y = max(15, info_y) # Keep text below the top edge
                    cv2.putText(processed_frame, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    info_y -= 18 # Move up for next line
                
                if 'name' in face_data: put_text(f"Name: {face_data['name']}")
                put_text(f"Age: {face_data['age']}")
                put_text(f"Gender: {face_data['gender']}")
                put_text(f"Emotion: {face_data['emotion']} ({face_data['smile_intensity']})", (0, 0, 255))
                put_text(f"Glasses: {'Yes' if face_data['glasses'] else 'No'}")
                put_text(f"Symmetry: {face_data['symmetry']}%")
                put_text(f"Shape: {face_data['face_shape'].capitalize()}")
                put_text(f"Pose: {face_data['head_pose']}")
                if 'eye_distance' in face_data['measurements']:
                    put_text(f"Eye Dist: {face_data['measurements']['eye_distance']}")
                    
                all_face_data.append(face_data)
            except Exception as e:
                print(f"Error analyzing face at {(x, y, w, h)}: {e}")
                # Draw a simple box indicating error
                cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
                cv2.putText(processed_frame, "Error", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                all_face_data.append({'rect': (x, y, w, h), 'error': str(e)})
            
        # Display overall info (face count)
        cv2.putText(processed_frame, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return processed_frame, all_face_data

    def process_image(self, image_path, apply_filter=None, blur_faces=False, recognize=False, correct_orient=False):
        img = cv2.imread(image_path)
        if img is None: raise ValueError(f"Could not read image: {image_path}")
        
        # Optional orientation correction (applied before processing)
        if correct_orient:
            # Need to detect face first to get angle
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            if len(faces) > 0:
                img, angle = self.correct_orientation(img, faces[0]) # Correct based on first face
                print(f"Corrected orientation by {angle:.1f} degrees")
        
        return self.process_frame(img, apply_filter, blur_faces, recognize)

    def process_video(self, video_source=0, apply_filter=None, blur_faces=False, recognize=False, track=True):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened(): raise ValueError("Could not open video source")
        
        print("Processing video. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            processed_frame, _ = self.process_frame(frame, apply_filter, blur_faces, recognize, track)
            
            if processed_frame is not None:
                cv2.imshow('Face Analysis Video', processed_frame)
            else:
                print("Warning: Frame processing returned None")
                
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        cap.release()
        cv2.destroyAllWindows()

    def process_batch(self, input_dir, output_dir, apply_filter=None, blur_faces=False, recognize=False, correct_orient=False):
        if not os.path.isdir(input_dir): raise ValueError(f"Input directory not found: {input_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"processed_{filename}")
                print(f"Processing {input_path}...")
                try:
                    result_img, _ = self.process_image(input_path, apply_filter, blur_faces, recognize, correct_orient)
                    if result_img is not None:
                        cv2.imwrite(output_path, result_img)
                        print(f"Saved result to {output_path}")
                    else:
                        print(f"Skipped saving {output_path} due to processing error.")
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

# --- GUI Implementation --- 

class FaceAnalysisGUI:
    def __init__(self, root, app_logic):
        if not GUI_AVAILABLE:
            raise ImportError("GUI cannot be created because Tkinter or Pillow is not installed.")
            
        self.root = root
        self.app_logic = app_logic
        self.root.title("Face Analysis Tool")
        self.root.geometry("800x600")

        self.image_path = None
        self.original_image = None
        self.processed_image = None

        # --- Controls Frame --- 
        controls_frame = ttk.Frame(root, padding="10")
        controls_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Button(controls_frame, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=5)
        ttk.Button(controls_frame, text="Process Image", command=self.process_gui_image).pack(fill=tk.X, pady=5)
        ttk.Button(controls_frame, text="Start Webcam", command=self.start_webcam).pack(fill=tk.X, pady=5)
        ttk.Button(controls_frame, text="Batch Process", command=self.batch_process_gui).pack(fill=tk.X, pady=5)
        ttk.Button(controls_frame, text="Save Image", command=self.save_image).pack(fill=tk.X, pady=5)

        # Options
        options_frame = ttk.LabelFrame(controls_frame, text="Options", padding="10")
        options_frame.pack(fill=tk.X, pady=10)

        self.filter_var = tk.StringVar(value="none")
        ttk.Label(options_frame, text="Filter:").grid(row=0, column=0, sticky=tk.W)
        filter_menu = ttk.Combobox(options_frame, textvariable=self.filter_var, values=["none", "sepia", "bw", "cartoon"])
        filter_menu.grid(row=0, column=1, sticky=tk.EW)

        self.blur_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Blur Faces", variable=self.blur_var).grid(row=1, column=0, columnspan=2, sticky=tk.W)
        
        self.recognize_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Recognize Faces", variable=self.recognize_var).grid(row=2, column=0, columnspan=2, sticky=tk.W)
        
        self.orient_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Correct Orientation", variable=self.orient_var).grid(row=3, column=0, columnspan=2, sticky=tk.W)

        # --- Image Display Frame --- 
        self.image_frame = ttk.Frame(root)
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        # Bind configure event to resize image when window size changes
        self.image_label.bind('<Configure>', self.on_resize)

    def on_resize(self, event):
        # Redisplay the current image (either original or processed) on resize
        if self.processed_image is not None:
            self.display_image(self.processed_image)
        elif self.original_image is not None:
            self.display_image(self.original_image)

    def load_image(self):
        self.image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*"))
        )
        if self.image_path:
            self.original_image = cv2.imread(self.image_path)
            if self.original_image is None:
                messagebox.showerror("Error", f"Could not load image: {self.image_path}")
                self.image_path = None
                return
            self.display_image(self.original_image)
            self.processed_image = None # Reset processed image

    def process_gui_image(self):
        if self.image_path is None or self.original_image is None:
            messagebox.showerror("Error", "Please load an image first.")
            return
        
        try:
            filter_type = self.filter_var.get()
            if filter_type == "none": filter_type = None
            
            self.processed_image, face_data = self.app_logic.process_image(
                self.image_path,
                apply_filter=filter_type,
                blur_faces=self.blur_var.get(),
                recognize=self.recognize_var.get(),
                correct_orient=self.orient_var.get()
            )
            if self.processed_image is not None:
                self.display_image(self.processed_image)
                # Optionally display face_data in a separate window or panel
                print("Processing complete. Face data:", face_data)
            else:
                 messagebox.showerror("Processing Error", "Image processing failed.")
        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred: {e}")

    def start_webcam(self):
        try:
            filter_type = self.filter_var.get()
            if filter_type == "none": filter_type = None
            # Run video processing in a non-blocking way if possible, or inform user it might freeze GUI
            messagebox.showinfo("Webcam", "Starting webcam processing. Press 'q' in the video window to stop.")
            # Consider running this in a separate thread to avoid freezing the GUI
            self.app_logic.process_video(
                apply_filter=filter_type,
                blur_faces=self.blur_var.get(),
                recognize=self.recognize_var.get(),
                track=True # Enable tracking for video
            )
        except Exception as e:
            messagebox.showerror("Webcam Error", f"Could not start webcam: {e}")

    def batch_process_gui(self):
        input_dir = filedialog.askdirectory(title="Select Input Directory")
        if not input_dir: return
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir: return
        
        try:
            filter_type = self.filter_var.get()
            if filter_type == "none": filter_type = None
            messagebox.showinfo("Batch Processing", f"Starting batch process from {input_dir} to {output_dir}. Check console for progress.")
            # Consider running this in a separate thread
            self.app_logic.process_batch(
                input_dir, 
                output_dir,
                apply_filter=filter_type,
                blur_faces=self.blur_var.get(),
                recognize=self.recognize_var.get(),
                correct_orient=self.orient_var.get()
            )
            messagebox.showinfo("Batch Processing", "Batch processing complete.")
        except Exception as e:
            messagebox.showerror("Batch Processing Error", f"An error occurred: {e}")

    def save_image(self):
        if self.processed_image is None:
            messagebox.showerror("Error", "No processed image to save. Please process an image first.")
            return
        
        save_path = filedialog.asksaveasfilename(
            title="Save Processed Image",
            defaultextension=".png",
            filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*"))
        )
        if save_path:
            try:
                cv2.imwrite(save_path, self.processed_image)
                messagebox.showinfo("Success", f"Image saved successfully to {save_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save image: {e}")

    def display_image(self, cv_img):
        if cv_img is None: return
        # Convert OpenCV image (BGR) to PIL image (RGB)
        try:
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
        except cv2.error as e:
            print(f"Error converting image for display: {e}")
            return
        
        # Resize image to fit label while maintaining aspect ratio
        label_width = self.image_label.winfo_width()
        label_height = self.image_label.winfo_height()
        
        # If label size is not yet determined, use a default or skip resizing
        if label_width <= 1 or label_height <= 1:
            # Use a default size or just display original size if label not ready
            # self.root.after(50, lambda: self.display_image(cv_img)) # Option: Retry later
            # For now, just use original size if label isn't ready
            resized_img = pil_img 
        else:
            img_width, img_height = pil_img.size
            if img_width <= 0 or img_height <= 0: return # Invalid image size
            
            ratio = min(label_width / img_width, label_height / img_height)
            new_size = (max(1, int(img_width * ratio)), max(1, int(img_height * ratio)))
            
            resized_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert PIL image to Tkinter PhotoImage
        try:
            self.tk_image = ImageTk.PhotoImage(resized_img)
            self.image_label.config(image=self.tk_image)
            self.image_label.image = self.tk_image # Keep reference
        except Exception as e:
            print(f"Error creating Tkinter image: {e}")

# --- Main Execution --- 

def main():
    parser = argparse.ArgumentParser(description='Advanced Face Analysis Tool')
    parser.add_argument('--mode', choices=['gui', 'image', 'video', 'batch'], default='gui', help='Execution mode')
    parser.add_argument('--input', help='Path to input image or directory (for image/batch mode)')
    parser.add_argument('--output', help='Path to save output image or directory (for image/batch mode)')
    parser.add_argument('--video_source', default=0, type=int, help='Video source index (for video mode)')
    
    # Processing options for command-line modes
    parser.add_argument('--filter', choices=['none', 'sepia', 'bw', 'cartoon'], default='none', help='Apply image filter')
    parser.add_argument('--blur', action='store_true', help='Blur detected faces')
    parser.add_argument('--recognize', action='store_true', help='Enable basic face recognition')
    parser.add_argument('--track', action='store_true', default=True, help='Enable face tracking in video mode')
    parser.add_argument('--orient', action='store_true', help='Correct face orientation')
    
    parser.add_argument('--data-dir', help='Custom directory for Haar cascades')
    parser.add_argument('--models-dir', help='Custom directory for DNN models')
    
    args = parser.parse_args()

    app_logic = FaceAnalysisApp(args.data_dir, args.models_dir)

    filter_type = args.filter if args.filter != 'none' else None

    if args.mode == 'gui':
        if not GUI_AVAILABLE:
            print("Error: GUI mode selected, but Tkinter or Pillow is not installed.")
            print("Please install Tkinter and Pillow or run in a command-line mode (image, video, batch).")
            sys.exit(1)
        root = tk.Tk()
        gui = FaceAnalysisGUI(root, app_logic)
        root.mainloop()
    elif args.mode == 'image':
        if not args.input: 
            print("Error: Input image path required for image mode.")
            parser.print_help()
            sys.exit(1)
        result_img, face_data = app_logic.process_image(args.input, filter_type, args.blur, args.recognize, args.orient)
        print("Face Data:", face_data)
        if result_img is not None:
            if args.output:
                cv2.imwrite(args.output, result_img)
                print(f"Output saved to {args.output}")
            else:
                # Display image only if not headless/output specified
                try:
                    cv2.imshow("Processed Image", result_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                except cv2.error as e:
                     print(f"Warning: Could not display image. Error: {e}")
                     print("If running in a headless environment, please specify an --output path to save the result.")
        else:
            print("Image processing failed.")
    elif args.mode == 'video':
        app_logic.process_video(args.video_source, filter_type, args.blur, args.recognize, args.track)
    elif args.mode == 'batch':
        if not args.input: 
            print("Error: Input directory required for batch mode.")
            parser.print_help()
            sys.exit(1)
        if not args.output: 
            print("Error: Output directory required for batch mode.")
            parser.print_help()
            sys.exit(1)
        app_logic.process_batch(args.input, args.output, filter_type, args.blur, args.recognize, args.orient)
    else:
        print("Invalid mode selected.")
        parser.print_help()

if __name__ == "__main__":
    main()

