#!/usr/bin/env python3
"""
YARP RFModule for Real-Time Valence/Arousal Estimation using EmoNet

This module subscribes to face annotations and webcam images, runs the EmoNet
model on detected faces, and publishes valence/arousal estimations.

Author: Generated for EmoNet integration with YARP
License: CC BY-NC-ND (same as EmoNet)

Usage:
    python emonet_valence_arousal_rfmodule.py --model_path pretrained/emonet_8.pth --nclasses 8
    
    Or with all options:
    python emonet_valence_arousal_rfmodule.py \
        --model_path pretrained/emonet_8.pth \
        --nclasses 8 \
        --device cuda \
        --input_size 256 \
        --period 0.033 \
        --min_score 0.5 \
        --emonet_root /path/to/emonet
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

try:
    import yarp
except ImportError:
    print("[ERROR] YARP Python bindings not found. Please install YARP with Python bindings.")
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
    import cv2
except ImportError as e:
    print(f"[ERROR] Required package not found: {e}")
    sys.exit(1)


class EmoNetValenceArousalModule(yarp.RFModule):
    """
    YARP RFModule for real-time valence/arousal estimation using EmoNet.
    
    Subscribes to:
        - /faceID/annotations:o (Bottle): Face annotations with bounding boxes
        - /webcam (ImageRgb): Raw RGB images
    
    Publishes to:
        - /emonet/valence_arousal:o (Bottle): Triplets of <name> <valence> <arousal>
    
    Output format:
        Each face produces: name (string), valence (float64), arousal (float64)
        Example: (Nima 0.42 0.31) ("Unknown face" -0.10 0.55)
    """
    
    def __init__(self):
        """Initialize the RFModule."""
        yarp.RFModule.__init__(self)
        
        # YARP ports
        self._annotation_port = None
        self._image_port = None
        self._output_port = None
        
        # Model components
        self._model = None
        self._device = None
        self._input_size = 256
        self._nclasses = 8
        
        # Configuration
        self._period = 0.033  # ~30 FPS
        self._min_score = 0.5
        self._model_path = None
        self._emonet_root = None
        
        # State
        self._is_configured = False
        
        # Timestamp
        self._stamp = yarp.Stamp()
        
    def configure(self, rf: yarp.ResourceFinder) -> bool:
        """
        Configure the module with parameters from ResourceFinder.
        
        Args:
            rf: YARP ResourceFinder with configuration parameters
            
        Returns:
            True if configuration successful, False otherwise
        """
        print("[INFO] Configuring EmoNetValenceArousalModule...")
        
        # Initialize YARP network
        if not yarp.Network.checkNetwork(2.0):
            print("[WARNING] YARP network not available, but continuing...")
        
        # Parse configuration parameters
        self._model_path = rf.find("model_path").asString() if rf.check("model_path") else ""
        self._nclasses = rf.find("nclasses").asInt32() if rf.check("nclasses") else 8
        self._input_size = rf.find("input_size").asInt32() if rf.check("input_size") else 256
        self._period = rf.find("period").asFloat64() if rf.check("period") else 0.033
        self._min_score = rf.find("min_score").asFloat64() if rf.check("min_score") else 0.0
        self._emonet_root = rf.find("emonet_root").asString() if rf.check("emonet_root") else ""
        
        # Device selection
        device_str = rf.find("device").asString() if rf.check("device") else "cuda"
        if device_str == "cuda" and torch.cuda.is_available():
            self._device = torch.device("cuda:0")
            print(f"[INFO] Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            self._device = torch.device("cpu")
            print("[INFO] Using CPU device")
        
        # Add emonet_root to path if specified
        if self._emonet_root and os.path.isdir(self._emonet_root):
            sys.path.insert(0, self._emonet_root)
            print(f"[INFO] Added emonet_root to path: {self._emonet_root}")
        
        # Load the model
        if not self._load_model():
            print("[ERROR] Failed to load EmoNet model")
            return False
        
        # Open YARP ports
        if not self._open_ports():
            print("[ERROR] Failed to open YARP ports")
            return False
        
        # Attempt connections (non-blocking)
        self._connect_ports()
        
        self._is_configured = True
        print("[INFO] Configuration complete")
        return True
    
    def _load_model(self) -> bool:
        """
        Load the EmoNet model from the specified path.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # Determine model path
            if not self._model_path:
                # Try default paths
                script_dir = Path(__file__).parent
                default_path = script_dir / "pretrained" / f"emonet_{self._nclasses}.pth"
                if default_path.exists():
                    self._model_path = str(default_path)
                else:
                    print(f"[ERROR] No model path specified and default not found: {default_path}")
                    return False
            
            model_path = Path(self._model_path)
            if not model_path.exists():
                print(f"[ERROR] Model file not found: {model_path}")
                return False
            
            print(f"[INFO] Loading model from: {model_path}")
            
            # Try to import EmoNet from the emonet package
            try:
                from emonet.models import EmoNet
                
                # Load state dict
                state_dict = torch.load(str(model_path), map_location="cpu")
                
                # Handle DataParallel saved models
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                
                # Create and load model
                self._model = EmoNet(n_expression=self._nclasses).to(self._device)
                self._model.load_state_dict(state_dict, strict=False)
                self._model.eval()
                
                print(f"[INFO] EmoNet model loaded successfully (n_expression={self._nclasses})")
                return True
                
            except ImportError:
                print("[WARNING] Could not import EmoNet from emonet.models")
                print("[INFO] Attempting to load as TorchScript model...")
                
                # Try loading as TorchScript model
                try:
                    self._model = torch.jit.load(str(model_path), map_location=self._device)
                    self._model.eval()
                    print("[INFO] Loaded model as TorchScript")
                    return True
                except Exception as e:
                    print(f"[ERROR] Failed to load as TorchScript: {e}")
                    return False
                    
        except Exception as e:
            print(f"[ERROR] Exception loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _open_ports(self) -> bool:
        """
        Open all YARP ports.
        
        Returns:
            True if all ports opened successfully, False otherwise
        """
        try:
            # Input port for face annotations
            self._annotation_port = yarp.BufferedPortBottle()
            if not self._annotation_port.open("/emonet/faceID/annotations:i"):
                print("[ERROR] Failed to open annotation input port")
                return False
            print("[INFO] Opened port: /emonet/faceID/annotations:i")
            
            # Input port for webcam images
            self._image_port = yarp.BufferedPortImageRgb()
            if not self._image_port.open("/emonet/webcam:i"):
                print("[ERROR] Failed to open image input port")
                return False
            print("[INFO] Opened port: /emonet/webcam:i")
            
            # Output port for valence/arousal
            self._output_port = yarp.BufferedPortBottle()
            if not self._output_port.open("/emonet/valence_arousal:o"):
                print("[ERROR] Failed to open output port")
                return False
            print("[INFO] Opened port: /emonet/valence_arousal:o")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Exception opening ports: {e}")
            return False
    
    def _connect_ports(self) -> None:
        """
        Attempt to connect ports to their sources/destinations.
        Non-blocking: logs warnings if connection fails.
        """
        # Connect annotation input
        if yarp.Network.connect("/faceID/annotations:o", "/emonet/faceID/annotations:i"):
            print("[INFO] Connected /faceID/annotations:o -> /emonet/faceID/annotations:i")
        else:
            print("[WARNING] Could not connect /faceID/annotations:o -> /emonet/faceID/annotations:i")
        
        # Connect image input
        if yarp.Network.connect("/webcam", "/emonet/webcam:i"):
            print("[INFO] Connected /webcam -> /emonet/webcam:i")
        else:
            print("[WARNING] Could not connect /webcam -> /emonet/webcam:i")
    
    def getPeriod(self) -> float:
        """
        Return the module update period in seconds.
        
        Returns:
            Update period (default ~30 FPS)
        """
        return self._period
    
    def updateModule(self) -> bool:
        """
        Main update loop - called periodically.
        
        Reads annotations and images, runs inference, publishes results.
        
        Returns:
            True to continue running, False to stop
        """
        if not self._is_configured:
            return True
        
        try:
            # Read annotation bottle (non-blocking)
            annotation_bottle = self._annotation_port.read(False)
            if annotation_bottle is None:
                return True
            
            # Read image (non-blocking)
            yarp_image = self._image_port.read(False)
            if yarp_image is None:
                return True
            
            # Convert YARP image to numpy array
            frame_rgb = self._yarp_image_to_np(yarp_image)
            if frame_rgb is None:
                print("[WARNING] Failed to convert YARP image to numpy")
                return True
            
            # Parse face annotations
            faces = self._parse_annotations(annotation_bottle)
            if not faces:
                # No valid faces, publish empty or skip
                return True
            
            # Filter by minimum score
            faces = [f for f in faces if f["score"] >= self._min_score]
            if not faces:
                return True
            
            # Extract and preprocess face crops
            face_tensors, valid_faces = self._preprocess_faces(frame_rgb, faces)
            if face_tensors is None or len(valid_faces) == 0:
                return True
            
            # Run inference
            results = self._infer(face_tensors)
            if results is None:
                return True
            
            # Publish results
            self._publish_results(valid_faces, results)
            
        except Exception as e:
            print(f"[ERROR] Exception in updateModule: {e}")
            import traceback
            traceback.print_exc()
        
        return True
    
    def _yarp_image_to_np(self, yarp_image: yarp.ImageRgb) -> Optional[np.ndarray]:
        """
        Convert YARP ImageRgb to numpy array efficiently.
        
        Args:
            yarp_image: YARP ImageRgb object
            
        Returns:
            Numpy array of shape (H, W, 3) in RGB format, or None on failure
        """
        try:
            width = yarp_image.width()
            height = yarp_image.height()
            
            if width <= 0 or height <= 0:
                return None
            
            # Get raw image data pointer
            # YARP ImageRgb stores data as RGB interleaved
            # getRawImage() returns a pointer to the raw data
            
            # Method 1: Use numpy.frombuffer with external pointer
            # This is the efficient way
            try:
                # Get the raw data as bytes
                # YARP stores images row by row, with possible padding
                row_size = yarp_image.getRowSize()
                
                # Create numpy array from raw pointer
                raw_image = yarp_image.getRawImage()
                
                # Convert to numpy using ctypes
                import ctypes
                
                # Calculate total buffer size
                buffer_size = row_size * height
                
                # Create a buffer from the pointer
                buffer_ptr = ctypes.cast(int(raw_image), ctypes.POINTER(ctypes.c_uint8 * buffer_size))
                np_array = np.frombuffer(buffer_ptr.contents, dtype=np.uint8)
                
                # Reshape considering row stride
                if row_size == width * 3:
                    # No padding
                    frame = np_array.reshape((height, width, 3))
                else:
                    # Has padding, need to handle stride
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    for row in range(height):
                        row_start = row * row_size
                        frame[row, :, :] = np_array[row_start:row_start + width * 3].reshape((width, 3))
                
                return frame.copy()  # Return a copy to ensure data persistence
                
            except Exception as e:
                print(f"[WARNING] Fast image conversion failed: {e}, using fallback")
                
                # Method 2: Fallback - pixel by pixel (slow but reliable)
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                for y in range(height):
                    for x in range(width):
                        pixel = yarp_image.pixel(x, y)
                        frame[y, x, 0] = pixel.r
                        frame[y, x, 1] = pixel.g
                        frame[y, x, 2] = pixel.b
                return frame
                
        except Exception as e:
            print(f"[ERROR] Failed to convert YARP image: {e}")
            return None
    
    def _parse_annotations(self, bottle: yarp.Bottle) -> List[Dict[str, Any]]:
        """
        Parse face annotations from YARP Bottle.
        
        Expected format:
            ((class Nima) (score 0.8788) (box (x1 y1 x2 y2)))
            ((class "Unknown face") (score 0.8664) (box (x1 y1 x2 y2)))
        
        Args:
            bottle: YARP Bottle containing face annotations
            
        Returns:
            List of face dictionaries with 'name', 'score', 'box' keys
        """
        faces = []
        
        try:
            # Each element in the bottle is a face annotation (as a list/bottle)
            for i in range(bottle.size()):
                face_element = bottle.get(i)
                
                # Skip non-list elements
                if not face_element.isList():
                    continue
                
                face_bottle = face_element.asList()
                face_data = self._parse_single_face(face_bottle)
                
                if face_data is not None:
                    faces.append(face_data)
                    
        except Exception as e:
            print(f"[WARNING] Error parsing annotations: {e}")
        
        return faces
    
    def _parse_single_face(self, face_bottle: yarp.Bottle) -> Optional[Dict[str, Any]]:
        """
        Parse a single face annotation.
        
        Args:
            face_bottle: YARP Bottle containing single face data
            
        Returns:
            Dictionary with 'name', 'score', 'box' or None if invalid
        """
        name = None
        score = 0.0
        box = None
        
        try:
            for j in range(face_bottle.size()):
                item = face_bottle.get(j)
                
                if not item.isList():
                    continue
                
                item_list = item.asList()
                if item_list.size() < 2:
                    continue
                
                key = item_list.get(0).asString()
                
                if key == "class":
                    # Handle both quoted and unquoted names
                    value = item_list.get(1)
                    if value.isString():
                        name = value.asString()
                    else:
                        # Try to get as string anyway
                        name = str(value.toString())
                        
                elif key == "score":
                    value = item_list.get(1)
                    if value.isFloat64():
                        score = value.asFloat64()
                    elif value.isFloat32():
                        score = float(value.asFloat32())
                    elif value.isInt32():
                        score = float(value.asInt32())
                    else:
                        try:
                            score = float(value.toString())
                        except:
                            score = 0.0
                            
                elif key == "box":
                    # Box can be (x1 y1 x2 y2) as a list or as 4 values
                    if item_list.size() >= 5:
                        # (box x1 y1 x2 y2) format
                        try:
                            x1 = self._get_numeric_value(item_list.get(1))
                            y1 = self._get_numeric_value(item_list.get(2))
                            x2 = self._get_numeric_value(item_list.get(3))
                            y2 = self._get_numeric_value(item_list.get(4))
                            box = (x1, y1, x2, y2)
                        except:
                            pass
                    elif item_list.size() >= 2:
                        # (box (x1 y1 x2 y2)) format - nested list
                        box_value = item_list.get(1)
                        if box_value.isList():
                            box_list = box_value.asList()
                            if box_list.size() >= 4:
                                try:
                                    x1 = self._get_numeric_value(box_list.get(0))
                                    y1 = self._get_numeric_value(box_list.get(1))
                                    x2 = self._get_numeric_value(box_list.get(2))
                                    y2 = self._get_numeric_value(box_list.get(3))
                                    box = (x1, y1, x2, y2)
                                except:
                                    pass
            
            # Validate we have required data
            if name is not None and box is not None:
                return {
                    "name": name,
                    "score": score,
                    "box": box
                }
                
        except Exception as e:
            print(f"[WARNING] Error parsing single face: {e}")
        
        return None
    
    def _get_numeric_value(self, value: yarp.Value) -> float:
        """
        Extract numeric value from YARP Value.
        
        Args:
            value: YARP Value object
            
        Returns:
            Float value
        """
        if value.isFloat64():
            return value.asFloat64()
        elif value.isFloat32():
            return float(value.asFloat32())
        elif value.isInt32():
            return float(value.asInt32())
        elif value.isInt64():
            return float(value.asInt64())
        else:
            return float(value.toString())
    
    def _preprocess_faces(
        self, 
        frame_rgb: np.ndarray, 
        faces: List[Dict[str, Any]]
    ) -> Tuple[Optional[torch.Tensor], List[Dict[str, Any]]]:
        """
        Extract and preprocess face crops for inference.
        
        Args:
            frame_rgb: Full frame as numpy array (H, W, 3) in RGB
            faces: List of face dictionaries
            
        Returns:
            Tuple of (batched tensor, list of valid faces)
        """
        h, w = frame_rgb.shape[:2]
        valid_faces = []
        face_tensors = []
        
        for face in faces:
            box = face["box"]
            
            # Clamp box coordinates to image bounds
            x1 = max(0, min(int(box[0]), w - 1))
            y1 = max(0, min(int(box[1]), h - 1))
            x2 = max(0, min(int(box[2]), w))
            y2 = max(0, min(int(box[3]), h))
            
            # Check for valid box dimensions
            if x2 <= x1 or y2 <= y1:
                print(f"[WARNING] Invalid box for {face['name']}: ({x1}, {y1}, {x2}, {y2})")
                continue
            
            # Minimum size check (at least 10x10 pixels)
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                print(f"[WARNING] Box too small for {face['name']}: {x2-x1}x{y2-y1}")
                continue
            
            try:
                # Extract face crop
                face_crop = frame_rgb[y1:y2, x1:x2, :]
                
                # Resize to model input size
                face_resized = cv2.resize(face_crop, (self._input_size, self._input_size))
                
                # Convert to tensor and normalize
                # EmoNet expects: tensor in [0, 1] range, shape (C, H, W)
                face_tensor = torch.from_numpy(face_resized).float()
                face_tensor = face_tensor.permute(2, 0, 1) / 255.0
                
                # Note: EmoNet demo does not apply ImageNet normalization,
                # it only scales to [0, 1]. If your model requires different
                # normalization, modify here:
                # mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                # std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                # face_tensor = (face_tensor - mean) / std
                
                face_tensors.append(face_tensor)
                valid_faces.append(face)
                
            except Exception as e:
                print(f"[WARNING] Failed to preprocess face {face['name']}: {e}")
                continue
        
        if not face_tensors:
            return None, []
        
        # Stack into batch tensor
        batch_tensor = torch.stack(face_tensors, dim=0).to(self._device)
        
        return batch_tensor, valid_faces
    
    def _infer(self, batch_tensor: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        """
        Run inference on batched face tensor.
        
        Args:
            batch_tensor: Tensor of shape (N, 3, H, W)
            
        Returns:
            Dictionary with 'valence' and 'arousal' tensors, or None on failure
        """
        try:
            with torch.no_grad():
                output = self._model(batch_tensor)
                
                # EmoNet returns dict with 'expression', 'valence', 'arousal', 'heatmap'
                # Clamp valence and arousal to [-1, 1]
                valence = output["valence"].clamp(-1.0, 1.0)
                arousal = output["arousal"].clamp(-1.0, 1.0)
                
                return {
                    "valence": valence.cpu(),
                    "arousal": arousal.cpu()
                }
                
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _publish_results(
        self, 
        faces: List[Dict[str, Any]], 
        results: Dict[str, torch.Tensor]
    ) -> None:
        """
        Publish valence/arousal results to output port with timestamp.
        
        Output format: Repeated triplets of <name> <valence> <arousal>
        Example: Nima 0.42 0.31 "Unknown face" -0.10 0.55
        
        Args:
            faces: List of face dictionaries
            results: Dictionary with 'valence' and 'arousal' tensors
        """
        try:
            out_bottle = self._output_port.prepare()
            out_bottle.clear()
            
            valence_values = results["valence"]
            arousal_values = results["arousal"]
            
            for i, face in enumerate(faces):
                name = face["name"]
                valence = float(valence_values[i].item())
                arousal = float(arousal_values[i].item())
                
                # Add triplet: name, valence, arousal
                out_bottle.addString(name)
                out_bottle.addFloat64(valence)
                out_bottle.addFloat64(arousal)
            
            # Add timestamp
            self._stamp.update()
            self._output_port.setEnvelope(self._stamp)
            
            self._output_port.write()
            
            # Debug logging
            if len(faces) > 0:
                debug_str = ", ".join([
                    f"{f['name']}: v={float(valence_values[i].item()):.3f}, a={float(arousal_values[i].item()):.3f}"
                    for i, f in enumerate(faces)
                ])
                print(f"[INFO] Published: {debug_str}")
                
        except Exception as e:
            print(f"[ERROR] Failed to publish results: {e}")
    
    def interruptModule(self) -> bool:
        """
        Handle interrupt request.
        
        Returns:
            True
        """
        print("[INFO] Interrupt requested")
        
        # Interrupt all ports
        if self._annotation_port is not None:
            self._annotation_port.interrupt()
        if self._image_port is not None:
            self._image_port.interrupt()
        if self._output_port is not None:
            self._output_port.interrupt()
        
        return True
    
    def close(self) -> bool:
        """
        Clean up and close all resources.
        
        Returns:
            True
        """
        print("[INFO] Closing module...")
        
        # Close all ports
        if self._annotation_port is not None:
            self._annotation_port.close()
            print("[INFO] Closed annotation port")
        if self._image_port is not None:
            self._image_port.close()
            print("[INFO] Closed image port")
        if self._output_port is not None:
            self._output_port.close()
            print("[INFO] Closed output port")
        
        # Release model
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[INFO] Module closed")
        return True


def main():
    """Main entry point."""
    # Initialize YARP
    yarp.Network.init()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="YARP RFModule for real-time valence/arousal estimation using EmoNet"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="",
        help="Path to EmoNet .pth model file"
    )
    parser.add_argument(
        "--nclasses", 
        type=int, 
        default=8, 
        choices=[5, 8],
        help="Number of emotion classes (5 or 8)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to run inference on"
    )
    parser.add_argument(
        "--input_size", 
        type=int, 
        default=256,
        help="Input size for face images"
    )
    parser.add_argument(
        "--period", 
        type=float, 
        default=0.033,
        help="Update period in seconds (~30 FPS default)"
    )
    parser.add_argument(
        "--min_score", 
        type=float, 
        default=0.5,
        help="Minimum face detection score threshold"
    )
    parser.add_argument(
        "--emonet_root", 
        type=str, 
        default="",
        help="Path to emonet repository root (added to sys.path)"
    )
    
    args, unknown = parser.parse_known_args()
    
    # Create ResourceFinder
    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext("valence_arousal_estimator")
    
    # Set default values from argparse
    # These can be overridden by --from config.ini or command-line YARP params
    if args.model_path:
        rf.setDefault("model_path", args.model_path)
    rf.setDefault("nclasses", str(args.nclasses))
    rf.setDefault("device", args.device)
    rf.setDefault("input_size", str(args.input_size))
    rf.setDefault("period", str(args.period))
    rf.setDefault("min_score", str(args.min_score))
    if args.emonet_root:
        rf.setDefault("emonet_root", args.emonet_root)
    
    # Configure ResourceFinder from command line (YARP style)
    # This allows loading config files with --from config.ini
    rf.configure(sys.argv)
    
    # Create and run module
    module = EmoNetValenceArousalModule()
    
    print("[INFO] Starting EmoNetValenceArousalModule...")
    print(f"[INFO] Configuration:")
    print(f"       model_path: {args.model_path or '(auto-detect)'}")
    print(f"       nclasses: {args.nclasses}")
    print(f"       device: {args.device}")
    print(f"       input_size: {args.input_size}")
    print(f"       period: {args.period}s ({1.0/args.period:.1f} FPS)")
    print(f"       min_score: {args.min_score}")
    
    # Run the module
    try:
        module.runModule(rf)
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt received")
    finally:
        module.close()
        yarp.Network.fini()
        print("[INFO] Shutdown complete")


if __name__ == "__main__":
    main()
