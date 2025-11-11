import numpy as np
import torch
import cv2
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
import argparse
import os

class HybridSkySegmenter:
    def __init__(self, 
                 sam_checkpoint="sam_vit_h_4b8939.pth",
                 sam_model_type="vit_h",
                 segformer_model="nvidia/segformer-b5-finetuned-ade-640-640"):
        """
        Hybrid segmentation: SegFormer for coarse prediction + SAM for refinement
        
        ADE20K classes:

        - Class 2: sky
        - Class 3: sometimes ceiling (indoor sky-like)

        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load SegFormer for semantic segmentation
        print("Loading SegFormer...")
        self.segformer_processor = SegformerImageProcessor.from_pretrained(segformer_model)
        self.segformer_model = SegformerForSemanticSegmentation.from_pretrained(segformer_model)
        self.segformer_model.to(self.device)
        self.segformer_model.eval()
        
        # Load SAM for refinement
        print("Loading SAM...")
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)
        
        print("Models loaded successfully!")
    
    def segment_sky(self, image_path, visualize=True, num_points=20):
        """
        Main segmentation pipeline
        
        Args:
            image_path: path to image
            visualize: whether to show results
            num_points: number of seed points to sample for SAM
        """
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Step 1: Get coarse sky mask from SegFormer
        print("Step 1: Getting coarse sky prediction from SegFormer...")
        coarse_mask, sky_prob = self._get_coarse_sky_mask(pil_image)
        
        if coarse_mask.sum() == 0:
            print("Warning: No sky detected by SegFormer")
            return np.zeros_like(coarse_mask)
        
        # Step 2: Extract seed points from coarse mask
        print("Step 2: Extracting seed points...")
        sky_points, non_sky_points = self._extract_seed_points(
            coarse_mask, sky_prob, num_points=num_points
        )
        
        # Step 3: Refine with SAM
        print("Step 3: Refining mask with SAM...")
        refined_mask = self._refine_with_sam(
            image_rgb, sky_points, non_sky_points
        )
        
        if visualize:
            self._visualize_pipeline(
                image_rgb, coarse_mask, refined_mask, 
                sky_points, non_sky_points
            )
        
        return refined_mask
    
    def _get_coarse_sky_mask(self, pil_image):
        """
        Get coarse sky mask from SegFormer
        """
        # Preprocess
        inputs = self.segformer_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.segformer_model(**inputs)
            logits = outputs.logits
        
        # Upsample to original size
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=pil_image.size[::-1],  # (height, width)
            mode="bilinear",
            align_corners=False
        )
        
        # Get class predictions and probabilities
        probs = torch.nn.functional.softmax(upsampled_logits, dim=1)
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        
        # Extract sky probability map
        sky_class = 2  # ADE20K sky class
        sky_prob = probs[0, sky_class].cpu().numpy()
        
        # Create binary mask (sky vs non-sky)
        coarse_mask = (pred_seg == sky_class).astype(np.uint8) * 255
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        coarse_mask = cv2.morphologyEx(coarse_mask, cv2.MORPH_CLOSE, kernel)
        coarse_mask = cv2.morphologyEx(coarse_mask, cv2.MORPH_OPEN, kernel)
        
        return coarse_mask, sky_prob
    
    def _extract_seed_points(self, coarse_mask, sky_prob, num_points=20):
        """
        Extract seed points from coarse mask using probability-weighted sampling
        """
        h, w = coarse_mask.shape
        
        # Sky points: sample from high-confidence sky regions
        sky_region = (coarse_mask > 0) & (sky_prob > 0.7)
        if sky_region.sum() > 0:
            # Get coordinates of sky pixels
            sky_coords = np.column_stack(np.where(sky_region))
            sky_probs = sky_prob[sky_region]
            
            # Sample points weighted by probability
            if len(sky_coords) > num_points:
                # Normalize probabilities
                weights = sky_probs / sky_probs.sum()
                indices = np.random.choice(
                    len(sky_coords), 
                    size=num_points, 
                    replace=False,
                    p=weights
                )
                sky_points = sky_coords[indices]
            else:
                sky_points = sky_coords
            
            # Convert from (row, col) to (x, y)
            sky_points = sky_points[:, [1, 0]]
        else:
            # Fallback: sample from top of image
            sky_points = np.array([[w//4, h//6], [w//2, h//6], [3*w//4, h//6]])
        
        # Non-sky points: sample from confident non-sky regions
        non_sky_region = (coarse_mask == 0) & (sky_prob < 0.3)
        if non_sky_region.sum() > 0:
            non_sky_coords = np.column_stack(np.where(non_sky_region))
            
            # Sample from bottom half preferentially
            bottom_mask = non_sky_coords[:, 0] > h // 2
            if bottom_mask.sum() > 0:
                non_sky_coords = non_sky_coords[bottom_mask]
            
            # Sample points
            num_negative = max(3, num_points // 4)
            if len(non_sky_coords) > num_negative:
                indices = np.random.choice(
                    len(non_sky_coords), 
                    size=num_negative, 
                    replace=False
                )
                non_sky_points = non_sky_coords[indices]
            else:
                non_sky_points = non_sky_coords
            
            # Convert from (row, col) to (x, y)
            non_sky_points = non_sky_points[:, [1, 0]]
        else:
            # Fallback: sample from bottom of image
            non_sky_points = np.array([[w//2, 3*h//4]])
        
        return sky_points, non_sky_points
    
    def _refine_with_sam(self, image_rgb, sky_points, non_sky_points):
        """
        Refine mask using SAM with seed points
        """
        # Set image for SAM
        self.sam_predictor.set_image(image_rgb)
        
        # Prepare points and labels
        input_points = np.vstack([sky_points, non_sky_points])
        input_labels = np.array(
            [1] * len(sky_points) + [0] * len(non_sky_points)
        )
        
        # Predict with SAM
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        
        # Select best mask
        best_mask_idx = np.argmax(scores)
        refined_mask = (masks[best_mask_idx] * 255).astype(np.uint8)
        
        return refined_mask
    
    def _visualize_pipeline(self, image, coarse_mask, refined_mask, 
                           sky_points, non_sky_points):
        """
        Visualize the complete pipeline
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image', fontsize=14)
        axes[0, 0].axis('off')
        
        # Coarse mask from SegFormer
        axes[0, 1].imshow(coarse_mask, cmap='gray')
        axes[0, 1].set_title('Coarse Mask (SegFormer)', fontsize=14)
        axes[0, 1].axis('off')
        
        # Seed points
        axes[0, 2].imshow(image)
        axes[0, 2].scatter(sky_points[:, 0], sky_points[:, 1], 
                          c='green', s=100, marker='o', 
                          edgecolors='white', linewidths=2, label='Sky')
        axes[0, 2].scatter(non_sky_points[:, 0], non_sky_points[:, 1], 
                          c='red', s=100, marker='x', 
                          linewidths=3, label='Non-Sky')
        axes[0, 2].set_title('Seed Points for SAM', fontsize=14)
        axes[0, 2].legend()
        axes[0, 2].axis('off')
        
        # Refined mask from SAM
        axes[1, 0].imshow(refined_mask, cmap='gray')
        axes[1, 0].set_title('Refined Mask (SAM)', fontsize=14)
        axes[1, 0].axis('off')
        
        # Overlay coarse
        overlay_coarse = image.copy()
        overlay_coarse[coarse_mask > 0] = overlay_coarse[coarse_mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
        axes[1, 1].imshow(overlay_coarse.astype(np.uint8))
        axes[1, 1].set_title('Coarse Overlay', fontsize=14)
        axes[1, 1].axis('off')
        
        # Overlay refined
        overlay_refined = image.copy()
        overlay_refined[refined_mask > 0] = overlay_refined[refined_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
        axes[1, 2].imshow(overlay_refined.astype(np.uint8))
        axes[1, 2].set_title('Refined Overlay', fontsize=14)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()

from pathlib import Path
from tqdm import tqdm
import json

def batch_process_hybrid(input_dir, output_dir_masks=None, output_dir_masks_overlayed=None, segmenter=None, save_intermediate=False):
    """
    Batch process images with masks and optional overlays
    """
    # Create output directories
    if output_dir_masks:
        Path(output_dir_masks).mkdir(parents=True, exist_ok=True)
        if save_intermediate:
            Path(output_dir_masks, 'coarse').mkdir(exist_ok=True)
        Path(output_dir_masks, 'refined').mkdir(exist_ok=True)
    
    if output_dir_masks_overlayed:
        Path(output_dir_masks_overlayed).mkdir(parents=True, exist_ok=True)
        Path(output_dir_masks_overlayed, 'coarse_overlays').mkdir(exist_ok=True)
        Path(output_dir_masks_overlayed, 'refined_overlays').mkdir(exist_ok=True)
    
    # Get all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(ext))
        image_files.extend(Path(input_dir).glob(ext.upper()))
    
    print(f"Found {len(image_files)} images")
    
    results = []
    
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Get masks
            image = cv2.imread(str(img_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            h, w = image_rgb.shape[:2]
            
            # Coarse mask
            coarse_mask, sky_prob = segmenter._get_coarse_sky_mask(pil_image)
            
            # Initialize variables
            sky_points = None
            non_sky_points = None
            refined_mask = None
            sky_detected = coarse_mask.sum() > 0
            
            if sky_detected:
                # Seed points
                sky_points, non_sky_points = segmenter._extract_seed_points(
                    coarse_mask, sky_prob, num_points=30
                )
                
                # Refined mask
                refined_mask = segmenter._refine_with_sam(
                    image_rgb, sky_points, non_sky_points
                )
            else:
                # Create empty mask when no sky detected
                refined_mask = np.zeros((h, w), dtype=np.uint8)
                # Create default points for visualization
                sky_points = np.array([[w//4, h//6], [w//2, h//6], [3*w//4, h//6]])
                non_sky_points = np.array([[w//2, 3*h//4]])
            
            # Save coarse masks (always)
            if output_dir_masks and save_intermediate:
                cv2.imwrite(
                    str(Path(output_dir_masks, 'coarse', f"{img_path.stem}_coarse.png")),
                    coarse_mask
                )
            
            # Save refined masks (always)
            if output_dir_masks:
                cv2.imwrite(
                    str(Path(output_dir_masks, 'refined', f"{img_path.stem}_mask.png")),
                    refined_mask
                )
            
            # Save overlays (always)
            if output_dir_masks_overlayed:
                # Coarse overlay
                coarse_overlay = image_rgb.copy()
                if sky_detected:
                    coarse_overlay[coarse_mask > 0] = coarse_overlay[coarse_mask > 0] * 0.6 + np.array([255, 0, 0]) * 0.4  # Red overlay
                
                coarse_overlay_bgr = cv2.cvtColor(coarse_overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(Path(output_dir_masks_overlayed, 'coarse_overlays', f"{img_path.stem}_coarse_overlay.jpg")),
                    coarse_overlay_bgr
                )
                
                # Refined overlay with points
                refined_overlay = image_rgb.copy()
                if sky_detected:
                    # Apply mask overlay (cyan for sky)
                    refined_overlay[refined_mask > 0] = refined_overlay[refined_mask > 0] * 0.6 + np.array([0, 255, 255]) * 0.4
                
                # Draw sky points (green circles) - even if no sky detected for visualization
                for point in sky_points:
                    cv2.circle(refined_overlay, tuple(point.astype(int)), 8, (0, 255, 0), -1)  # Green filled
                    cv2.circle(refined_overlay, tuple(point.astype(int)), 10, (255, 255, 255), 2)  # White border
                
                # Draw non-sky points (red X) - even if no sky detected for visualization
                for point in non_sky_points:
                    x, y = point.astype(int)
                    # White outline for X
                    cv2.line(refined_overlay, (x-8, y-8), (x+8, y+8), (255, 255, 255), 5)
                    cv2.line(refined_overlay, (x-8, y+8), (x+8, y-8), (255, 255, 255), 5)
                    # Red X
                    cv2.line(refined_overlay, (x-8, y-8), (x+8, y+8), (0, 0, 255), 3)
                    cv2.line(refined_overlay, (x-8, y+8), (x+8, y-8), (0, 0, 255), 3)
                
                # Convert back to BGR for saving
                refined_overlay_bgr = cv2.cvtColor(refined_overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(Path(output_dir_masks_overlayed, 'refined_overlays', f"{img_path.stem}_refined_overlay.jpg")),
                    refined_overlay_bgr
                )
            
            # Calculate metrics
            sky_percentage = (refined_mask > 0).sum() / refined_mask.size * 100 if refined_mask is not None else 0.0
            
            results.append({
                'filename': img_path.name,
                'sky_percentage': sky_percentage,
                'num_sky_points': len(sky_points) if sky_points is not None else 0,
                'num_non_sky_points': len(non_sky_points) if non_sky_points is not None else 0,
                'sky_detected_by_segformer': sky_detected,
                'success': True
            })
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            
            # Even on error, try to create empty outputs
            try:
                if output_dir_masks:
                    h, w = 1024, 1024  # Default size
                    empty_mask = np.zeros((h, w), dtype=np.uint8)
                    
                    if save_intermediate:
                        cv2.imwrite(
                            str(Path(output_dir_masks, 'coarse', f"{img_path.stem}_coarse.png")),
                            empty_mask
                        )
                    cv2.imwrite(
                        str(Path(output_dir_masks, 'refined', f"{img_path.stem}_mask.png")),
                        empty_mask
                    )
                
                if output_dir_masks_overlayed:
                    # Try to read image for overlay, or create black image
                    try:
                        image = cv2.imread(str(img_path))
                        if image is None:
                            image = np.zeros((1024, 1024, 3), dtype=np.uint8)
                    except:
                        image = np.zeros((1024, 1024, 3), dtype=np.uint8)
                    
                    cv2.imwrite(
                        str(Path(output_dir_masks_overlayed, 'coarse_overlays', f"{img_path.stem}_coarse_overlay.jpg")),
                        image
                    )
                    cv2.imwrite(
                        str(Path(output_dir_masks_overlayed, 'refined_overlays', f"{img_path.stem}_refined_overlay.jpg")),
                        image
                    )
            except:
                pass  # If even empty output creation fails, skip
            
            results.append({
                'filename': img_path.name,
                'error': str(e),
                'sky_detected_by_segformer': False,
                'success': False
            })
    
    # Save results to the mask output directory if it exists, otherwise overlay directory
    results_dir = output_dir_masks if output_dir_masks else output_dir_masks_overlayed
    if results_dir:
        with open(Path(results_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    
    print(f"\nSuccessfully processed: {sum(r['success'] for r in results)}/{len(results)}")
    successful_sky_detection = sum(1 for r in results if r.get('sky_detected_by_segformer', False) and r['success'])
    print(f"Sky detected by SegFormer: {successful_sky_detection}/{len(results)} ({successful_sky_detection/len(results)*100:.1f}%)")
    return results

def main():
    parser = argparse.ArgumentParser(description='Hybrid Sky Segmentation using SegFormer + SAM')
    parser.add_argument('--input_dir', required=True, help='Input directory containing images')
    parser.add_argument('--output_dir_masks', help='Output directory for sky masks (PNG format)')
    parser.add_argument('--output_dir_masks_overlayed', help='Output directory for overlayed images with sky masks and seed points')
    parser.add_argument('--model_checkpoint_path', default='/mnt/data/tijaz/sam_vit_h_4b8939.pth', 
                        help='Path to SAM model checkpoint')
    parser.add_argument('--sam_model_type', default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'],
                        help='SAM model type')
    parser.add_argument('--segformer_model', default='nvidia/segformer-b5-finetuned-ade-640-640',
                        help='SegFormer model name or path')
    parser.add_argument('--save_intermediate', action='store_true', 
                        help='Save intermediate coarse masks from SegFormer')
    parser.add_argument('--num_points', type=int, default=30, 
                        help='Number of seed points for SAM')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.output_dir_masks and not args.output_dir_masks_overlayed:
        print("Error: At least one of --output_dir_masks or --output_dir_masks_overlayed must be specified")
        return
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    if not os.path.exists(args.model_checkpoint_path):
        print(f"Error: Model checkpoint {args.model_checkpoint_path} does not exist")
        return
    
    # Initialize segmenter
    print("Initializing Hybrid Sky Segmenter...")
    segmenter = HybridSkySegmenter(
        sam_checkpoint=args.model_checkpoint_path,
        sam_model_type=args.sam_model_type,
        segformer_model=args.segformer_model
    )
    
    # Process images
    results = batch_process_hybrid(
        input_dir=args.input_dir,
        output_dir_masks=args.output_dir_masks,
        output_dir_masks_overlayed=args.output_dir_masks_overlayed,
        segmenter=segmenter,
        save_intermediate=args.save_intermediate
    )
    
    print(f"\nProcessing complete!")
    successful = sum(r['success'] for r in results)
    total = len(results)
    print(f"Success rate: {successful}/{total} ({successful/total*100:.1f}%)")

if __name__ == "__main__":
    main()

