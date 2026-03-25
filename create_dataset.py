"""
Realistic Traffic Sign Dataset Generator
Simulates real dashcam footage with proper lighting, rotation, and distance
Uses ColorSafeCompositor to guarantee no color shifts
"""

import cv2
import numpy as np
import random
from pathlib import Path
from color_safe_compositor import ColorSafeCompositor


class RealisticSignGenerator:
    """
    Generate realistic traffic sign images

    Features:
    - Different distances (close/medium/distant)
    - Camera angles (rotation, perspective)
    - Lighting scenarios (day/dusk/night/sunny/overcast)
    - Video quality degradation (compression, blur, noise)
    """

    def __init__(self, signs_dir, backgrounds_dir):
        self.compositor = ColorSafeCompositor(backgrounds_dir)
        self.signs_dir = Path(signs_dir)

        # Load all sign files
        self.sign_files = list(self.signs_dir.glob('*.png'))

        if not self.sign_files:
            raise ValueError(f"No sign PNGs found in {signs_dir}")

        print(f"✅ Loaded {len(self.sign_files)} signs")

        # Create class name mapping
        self.class_names = sorted([f.stem for f in self.sign_files])
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}

    # =========================================================================
    # DISTANCE RANGES
    # =========================================================================

    def get_scale_for_distance(self, distance_type):
        """
        Get sign scale based on distance

        Args:
            distance_type: 'close', 'medium', or 'distant'

        Returns:
            scale factor (0.0-1.0)
        """
        if distance_type == 'close':
            # 15-35% of frame (96-224 pixels)
            return random.uniform(0.15, 0.35)

        elif distance_type == 'medium':
            # 8-18% of frame (51-115 pixels)
            return random.uniform(0.08, 0.18)

        elif distance_type == 'distant':
            # 3-10% of frame (19-64 pixels)
            return random.uniform(0.03, 0.10)

        else:
            raise ValueError(f"Unknown distance: {distance_type}")

    # =========================================================================
    # CAMERA ANGLE / ROTATION
    # =========================================================================

    def apply_rotation_perspective(self, sign_bgr, alpha, angle_type='slight'):
        """
        Apply rotation and perspective transform to simulate camera angle

        Args:
            sign_bgr: Sign BGR image
            alpha: Alpha channel
            angle_type: 'none', 'slight', 'moderate', 'angled'

        Returns:
            Transformed sign_bgr, alpha
        """
        if angle_type == 'none':
            return sign_bgr, alpha

        h, w = sign_bgr.shape[:2]

        # Rotation angle
        if angle_type == 'slight':
            angle = random.uniform(-8, 8)
        elif angle_type == 'moderate':
            angle = random.uniform(-15, 15)
        elif angle_type == 'angled':
            angle = random.uniform(-25, 25)
        else:
            angle = 0

        # Rotate
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated_bgr = cv2.warpAffine(sign_bgr, M, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))

        rotated_alpha = cv2.warpAffine(alpha, M, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=0)

        # Perspective transform (simulate viewing angle)
        if angle_type in ['moderate', 'angled']:
            # Random perspective shift
            shift = random.uniform(0.05, 0.15) if angle_type == 'moderate' else random.uniform(0.1, 0.25)

            src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

            # Random corner shifts
            dst_pts = np.float32([
                [random.randint(0, int(w*shift)), random.randint(0, int(h*shift))],
                [w - random.randint(0, int(w*shift)), random.randint(0, int(h*shift))],
                [random.randint(0, int(w*shift)), h - random.randint(0, int(h*shift))],
                [w - random.randint(0, int(w*shift)), h - random.randint(0, int(h*shift))]
            ])

            M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)

            persp_bgr = cv2.warpPerspective(rotated_bgr, M_persp, (w, h),
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=(0, 0, 0))

            persp_alpha = cv2.warpPerspective(rotated_alpha, M_persp, (w, h),
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=0)

            return persp_bgr, persp_alpha

        return rotated_bgr, rotated_alpha

    # =========================================================================
    # LIGHTING SCENARIOS
    # =========================================================================

    def apply_lighting_scenario(self, img, scenario):
        """
        Apply realistic lighting scenario to entire image

        Args:
            img: BGR image
            scenario: 'sunny_day', 'overcast', 'dusk', 'night', 'bright_sun'

        Returns:
            Lit image (BGR)
        """
        if scenario == 'sunny_day':
            # Bright, clear day
            img = self.compositor.apply_brightness(img, random.uniform(1.05, 1.20))
            img = self.compositor.apply_vignette(img, 0.08)

        elif scenario == 'overcast':
            # Cloudy, even lighting
            img = self.compositor.apply_brightness(img, random.uniform(0.85, 0.95))

        elif scenario == 'dusk':
            # Evening, lower light
            img = self.compositor.apply_brightness(img, random.uniform(0.60, 0.75))
            img = self.compositor.apply_noise(img, 10)
            img = self.compositor.apply_vignette(img, 0.15)

        elif scenario == 'night':
            # Night with street lights
            img = self.compositor.apply_brightness(img, random.uniform(0.30, 0.45))
            img = self.compositor.apply_noise(img, 18)
            img = self.compositor.apply_vignette(img, 0.22)

        elif scenario == 'bright_sun':
            # Very bright, some glare
            img = self.compositor.apply_brightness(img, random.uniform(1.25, 1.40))
            img = self.compositor.apply_vignette(img, 0.10)

        return img

    # =========================================================================
    # VIDEO QUALITY
    # =========================================================================

    def apply_video_quality(self, img, quality_level):
        """
        Apply video quality degradation

        Args:
            img: BGR image
            quality_level: 'high', 'medium', 'low', 'poor'

        Returns:
            Degraded image
        """
        if quality_level == 'high':
            # Good dashcam (1080p, high bitrate)
            img = self.compositor.apply_compression(img, random.randint(88, 95))

        elif quality_level == 'medium':
            # Average dashcam (720p, medium bitrate)
            img = self.compositor.apply_compression(img, random.randint(78, 88))
            if random.random() < 0.3:
                img = self.compositor.apply_motion_blur(img, 5)

        elif quality_level == 'low':
            # Older dashcam or high compression
            img = self.compositor.apply_compression(img, random.randint(68, 78))
            img = self.compositor.apply_noise(img, 8)
            if random.random() < 0.5:
                img = self.compositor.apply_motion_blur(img, 7)

        elif quality_level == 'poor':
            # Very compressed or old camera
            img = self.compositor.apply_compression(img, random.randint(55, 68))
            img = self.compositor.apply_noise(img, 12)
            img = self.compositor.apply_motion_blur(img, 7)

        return img

    # =========================================================================
    # REALISTIC SIGN PLACEMENT
    # =========================================================================

    def get_realistic_sign_position(self, sign_width, sign_height, distance, img_width=640, img_height=640):
        """
        Get realistic position for sign based on where they appear in dashcam footage

        Args:
            sign_width: Width of sign in pixels
            sign_height: Height of sign in pixels
            distance: 'close', 'medium', or 'distant'
            img_width: Image width
            img_height: Image height

        Returns:
            (x, y) position, or None if can't place
        """

        # Define realistic placement zones based on distance
        if distance == 'close':
            # Close signs: Roadside (left/right edges), slightly off-center
            # Can appear anywhere from upper-middle to lower (as you're passing them)
            zones = [
                ('left_roadside', 0.05, 0.25, 0.15, 0.70),    # Left edge, mid to lower
                ('right_roadside', 0.70, 0.90, 0.15, 0.70),   # Right edge, mid to lower
                ('left_upper', 0.10, 0.30, 0.05, 0.35),       # Left upper (overhead)
                ('right_upper', 0.65, 0.85, 0.05, 0.35),      # Right upper (overhead)
            ]
            weights = [35, 35, 15, 15]  # Roadside more common than overhead

        elif distance == 'medium':
            # Medium signs: More centered vertically, still on edges horizontally
            # Appear in middle height range (approaching)
            zones = [
                ('left_mid', 0.05, 0.30, 0.20, 0.55),         # Left side, middle height
                ('right_mid', 0.65, 0.90, 0.20, 0.55),        # Right side, middle height
                ('overhead_center', 0.35, 0.65, 0.05, 0.25),  # Overhead gantry signs
                ('left_upper', 0.10, 0.35, 0.10, 0.40),       # Left upper
                ('right_upper', 0.60, 0.85, 0.10, 0.40),      # Right upper
            ]
            weights = [30, 30, 15, 12, 13]

        elif distance == 'distant':
            # Distant signs: More centered (straight ahead on road)
            # Appear in upper-middle (horizon line)
            zones = [
                ('center_upper', 0.30, 0.70, 0.15, 0.40),     # Center, upper third
                ('left_distant', 0.15, 0.40, 0.20, 0.45),     # Left side, distant
                ('right_distant', 0.60, 0.85, 0.20, 0.45),    # Right side, distant
                ('overhead_far', 0.35, 0.65, 0.10, 0.30),     # Overhead, far
            ]
            weights = [40, 25, 25, 10]

        else:
            # Fallback to random
            return (random.randint(50, img_width - sign_width - 50),
                   random.randint(50, img_height - sign_height - 50))

        # Select zone based on weights
        zone_name, x_min, x_max, y_min, y_max = random.choices(zones, weights=weights)[0]

        # Convert normalized coordinates to pixels
        x_min_px = int(x_min * img_width)
        x_max_px = int(x_max * img_width)
        y_min_px = int(y_min * img_height)
        y_max_px = int(y_max * img_height)

        # Ensure sign fits in zone
        if x_max_px - x_min_px < sign_width or y_max_px - y_min_px < sign_height:
            # Zone too small, return None
            return None

        # Random position within zone
        x = random.randint(x_min_px, max(x_min_px, x_max_px - sign_width))
        y = random.randint(y_min_px, max(y_min_px, y_max_px - sign_height))

        return (x, y)

    def generate_realistic_image(self, distance, lighting, quality, rotation,
                                 num_signs=None, degradation='light'):
        """
        Generate a single realistic traffic sign image

        Args:
            distance: 'close', 'medium', 'distant'
            lighting: 'sunny_day', 'overcast', 'dusk', 'night', 'bright_sun'
            quality: 'high', 'medium', 'low', 'poor'
            rotation: 'none', 'slight', 'moderate', 'angled'
            num_signs: Number of signs (None = random 1-3)
            degradation: 'none', 'light', 'medium', 'heavy'

        Returns:
            img (BGR), labels (list of [class_id, x, y, w, h])
        """
        # Load background
        img = self.compositor.load_background()
        img_h, img_w = img.shape[:2]

        # Determine number of signs
        if num_signs is None:
            # Fewer signs for distant (harder to fit), more for close
            if distance == 'distant':
                num_signs = random.choices([1, 2], weights=[80, 20])[0]
            elif distance == 'medium':
                num_signs = random.choices([1, 2, 3], weights=[50, 40, 10])[0]
            else:  # close
                num_signs = random.choices([1, 2, 3], weights=[40, 40, 20])[0]

        # Select random signs
        selected_signs = random.sample(self.sign_files, min(num_signs, len(self.sign_files)))

        labels = []
        placed_boxes = []  # Track placed signs to avoid overlap

        for sign_file in selected_signs:
            # Load sign
            sign_bgr, alpha = self.compositor.load_sign_png(sign_file)

            # Apply rotation/perspective
            sign_bgr, alpha = self.apply_rotation_perspective(sign_bgr, alpha, rotation)

            # Apply degradation to sign
            sign_bgr, alpha = self.compositor.apply_sign_degradation(
                sign_bgr, alpha, degradation
            )

            # Get scale for distance
            scale = self.get_scale_for_distance(distance)

            # Calculate sign size after scaling
            sign_h, sign_w = sign_bgr.shape[:2]
            new_w = int(sign_w * scale)
            new_h = int(sign_h * scale)

            if new_w < 5 or new_h < 5:
                continue  # Too small, skip

            # Try to place sign in realistic position
            max_attempts = 50
            placed = False

            for attempt in range(max_attempts):
                # Get realistic position based on distance
                position = self.get_realistic_sign_position(new_w, new_h, distance, img_w, img_h)

                if position is None:
                    continue  # Zone too small, try again

                x, y = position

                # Check bounds
                if x < 0 or y < 0 or x + new_w > img_w or y + new_h > img_h:
                    continue

                # Check overlap with existing signs
                overlaps = False
                new_box = (x, y, x + new_w, y + new_h)

                for existing_box in placed_boxes:
                    if self._boxes_overlap(new_box, existing_box):
                        overlaps = True
                        break

                if not overlaps:
                    # Composite sign
                    img, bbox = self.compositor.composite_sign_on_background(
                        img, sign_bgr, alpha, x, y, scale
                    )

                    if bbox is not None:
                        # Get class ID
                        class_name = sign_file.stem
                        class_id = self.class_to_id[class_name]

                        # Add label
                        labels.append([class_id] + bbox)
                        placed_boxes.append(new_box)
                        placed = True
                        break

            if not placed:
                # Skip this sign if couldn't place
                continue

        # Apply lighting
        img = self.apply_lighting_scenario(img, lighting)

        # Apply video quality
        img = self.apply_video_quality(img, quality)

        return img, labels

    def _boxes_overlap(self, box1, box2, threshold=0.3):
        """Check if two boxes overlap significantly"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return False

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        iou = intersection / min(area1, area2)

        return iou > threshold

    # =========================================================================
    # DATASET GENERATION
    # =========================================================================

    def generate_dataset(self, output_dir, num_images, scenario_distribution=None):
        """
        Generate complete dataset with varied scenarios

        Args:
            output_dir: Where to save
            num_images: How many images to generate
            scenario_distribution: Dict of scenario weights (optional)
        """
        if scenario_distribution is None:
            scenario_distribution = {
                # (distance, lighting, quality, rotation)
                ('close', 'sunny_day', 'high', 'slight'): 0.15,
                ('close', 'overcast', 'medium', 'moderate'): 0.10,
                ('medium', 'sunny_day', 'high', 'slight'): 0.15,
                ('medium', 'overcast', 'medium', 'moderate'): 0.15,
                ('medium', 'dusk', 'medium', 'slight'): 0.10,
                ('distant', 'sunny_day', 'high', 'none'): 0.10,
                ('distant', 'overcast', 'medium', 'none'): 0.10,
                ('close', 'night', 'low', 'slight'): 0.05,
                ('medium', 'night', 'low', 'slight'): 0.05,
                ('distant', 'dusk', 'medium', 'none'): 0.05,
            }

        print(f"\n🎨 Generating {num_images} images...")
        print("=" * 70)

        # Create scenario list based on weights
        scenarios = []
        for scenario, weight in scenario_distribution.items():
            count = int(num_images * weight)
            scenarios.extend([scenario] * count)

        # Fill remaining with random
        while len(scenarios) < num_images:
            scenarios.append(random.choice(list(scenario_distribution.keys())))

        # Shuffle
        random.shuffle(scenarios)

        generated = 0

        for i, (distance, lighting, quality, rotation) in enumerate(scenarios[:num_images]):
            # Generate image
            img, labels = self.generate_realistic_image(
                distance=distance,
                lighting=lighting,
                quality=quality,
                rotation=rotation,
                degradation=random.choice(['none', 'light', 'light', 'medium'])
            )

            # Save
            filename = f"{distance}_{lighting}_{quality}_{i:06d}"
            self.compositor.save_image(img, labels, output_dir, filename)

            generated += 1

            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_images}...")

        print(f"\nDataset complete: {generated} images")
        print(f"Saved to: {output_dir}")


def main():
    """Generate traffic sign dataset"""

    print("\n🚦 Realistic Traffic Sign Dataset Generator")
    print("=" * 70)

    # Configuration
    signs_dir = input("Path to sign PNGs: ").strip()
    backgrounds_dir = input("Path to backgrounds: ").strip()
    output_dir = input("Output directory: ").strip()
    num_images = int(input("Number of images (default 10000): ").strip() or "10000")

    # Create generator
    generator = RealisticSignGenerator(signs_dir, backgrounds_dir)

    # Generate dataset
    generator.generate_dataset(output_dir, num_images)


if __name__ == "__main__":
    main()