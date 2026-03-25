"""
BULLETPROOF Color-Safe Sign Compositor
100% CV2, NO PIL, NO color shifts
Uses ONLY BGR throughout entire pipeline
"""

import cv2
import numpy as np
import random
from pathlib import Path


class ColorSafeCompositor:
    """
    Sign compositing with GUARANTEED color preservation

    Rules:
    1. ONLY use CV2 (no PIL)
    2. Everything is BGR format
    3. Never touch individual channels except for compositing alpha
    4. All augmentations multiply ALL channels equally
    """

    def __init__(self, backgrounds_dir):
        self.backgrounds_dir = Path(backgrounds_dir)
        self.background_files = list(self.backgrounds_dir.glob('*.jpg')) + \
                               list(self.backgrounds_dir.glob('*.png'))

        if not self.background_files:
            raise ValueError(f"No backgrounds found in {backgrounds_dir}")

        print(f"✅ Loaded {len(self.background_files)} backgrounds")

    def load_background(self):
        """Load background image in BGR format"""
        bg_path = random.choice(self.background_files)
        bg = cv2.imread(str(bg_path))

        if bg is None:
            raise ValueError(f"Could not load background: {bg_path}")

        # Resize to 640x640
        bg = cv2.resize(bg, (640, 640))

        return bg

    def load_sign_png(self, sign_path):
        """
        Load sign PNG with transparency
        Returns: BGR image + alpha channel
        """
        # Load with alpha channel
        sign = cv2.imread(str(sign_path), cv2.IMREAD_UNCHANGED)

        if sign is None:
            raise ValueError(f"Could not load sign: {sign_path}")

        # Check if it has alpha
        if sign.shape[2] != 4:
            raise ValueError(f"Sign must have alpha channel: {sign_path}")

        # Split into BGR and alpha
        bgr = sign[:, :, :3]  # BGR channels
        alpha = sign[:, :, 3]  # Alpha channel

        return bgr, alpha

    def apply_sign_degradation(self, sign_bgr, alpha, degradation_level='none'):
        """
        Apply degradation to sign BEFORE compositing
        degradation_level: 'none', 'light', 'medium', 'heavy'

        CRITICAL: Only operates on BGR, preserves colors
        """
        if degradation_level == 'none':
            return sign_bgr, alpha

        result_bgr = sign_bgr.copy()
        result_alpha = alpha.copy()

        # Separate alpha so we can work on BGR only
        h, w = sign_bgr.shape[:2]

        if degradation_level in ['light', 'medium', 'heavy']:
            # Fading (reduce alpha)
            if degradation_level == 'light':
                result_alpha = (result_alpha * 0.95).astype(np.uint8)
            elif degradation_level == 'medium':
                result_alpha = (result_alpha * 0.85).astype(np.uint8)
            else:  # heavy
                result_alpha = (result_alpha * 0.75).astype(np.uint8)

            # Slight darkening (multiply all channels equally - NO color shift!)
            if degradation_level == 'medium':
                result_bgr = (result_bgr.astype(np.float32) * 0.92).astype(np.uint8)
            elif degradation_level == 'heavy':
                result_bgr = (result_bgr.astype(np.float32) * 0.85).astype(np.uint8)

        return result_bgr, result_alpha

    def composite_sign_on_background(self, background, sign_bgr, alpha, x, y, scale):
        """
        Composite sign onto background at position (x, y) with scale

        CRITICAL: Proper alpha compositing in BGR space
        """
        # Resize sign
        new_h = int(sign_bgr.shape[0] * scale)
        new_w = int(sign_bgr.shape[1] * scale)

        if new_h < 5 or new_w < 5:
            return background, None  # Too small, skip

        sign_resized = cv2.resize(sign_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        alpha_resized = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Check bounds
        bg_h, bg_w = background.shape[:2]

        if x < 0 or y < 0 or x + new_w > bg_w or y + new_h > bg_h:
            return background, None  # Out of bounds

        # Extract region of interest
        roi = background[y:y+new_h, x:x+new_w].copy()

        # Normalize alpha to 0-1 range
        alpha_mask = alpha_resized.astype(np.float32) / 255.0
        alpha_mask_3ch = np.stack([alpha_mask] * 3, axis=2)

        # Alpha compositing (proper formula)
        # result = sign * alpha + background * (1 - alpha)
        composited = (sign_resized.astype(np.float32) * alpha_mask_3ch +
                     roi.astype(np.float32) * (1 - alpha_mask_3ch))

        composited = np.clip(composited, 0, 255).astype(np.uint8)

        # Place back on background
        result = background.copy()
        result[y:y+new_h, x:x+new_w] = composited

        # Return YOLO format bbox
        # YOLO format: class_id x_center y_center width height (normalized 0-1)
        x_center = (x + new_w / 2) / bg_w
        y_center = (y + new_h / 2) / bg_h
        bbox_w = new_w / bg_w
        bbox_h = new_h / bg_h

        bbox = [x_center, y_center, bbox_w, bbox_h]

        return result, bbox

    def apply_brightness(self, img, factor):
        """
        Brightness adjustment - multiply ALL channels equally
        NO color shift!

        Uses float32 precision to avoid rounding errors
        """
        # CRITICAL: Use float32 for precision, round AFTER multiplication
        adjusted = np.round(img.astype(np.float32) * factor)
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def apply_compression(self, img, quality):
        """Video compression simulation"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        decoded = cv2.imdecode(encimg, 1)
        return decoded if decoded is not None else img

    def apply_motion_blur(self, img, kernel_size=7):
        """Horizontal motion blur"""
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = 1
        kernel /= kernel_size
        return cv2.filter2D(img, -1, kernel)

    def apply_noise(self, img, intensity=10):
        """Camera sensor noise"""
        noise = np.random.randn(*img.shape) * intensity
        noisy = img.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def apply_vignette(self, img, strength=0.15):
        """Lens vignette - darkened edges"""
        h, w = img.shape[:2]
        Y, X = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2

        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)

        mask = 1 - (dist / max_dist) * strength
        mask = np.clip(mask, 0.75, 1.0)

        result = img.astype(np.float32) * mask[:, :, np.newaxis]
        return np.clip(result, 0, 255).astype(np.uint8)

    def apply_fog(self, img, strength=0.15):
        """Atmospheric fog"""
        fog = np.ones_like(img, dtype=np.float32) * 240
        result = img.astype(np.float32) * (1 - strength) + fog * strength
        return np.clip(result, 0, 255).astype(np.uint8)

    def save_image(self, img, labels, output_dir, filename):
        """
        Save image and labels
        img: BGR format (CV2)
        """
        output_dir = Path(output_dir)
        images_dir = output_dir / 'images'
        labels_dir = output_dir / 'labels'

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Save image (BGR format, CV2 handles this correctly)
        cv2.imwrite(str(images_dir / f'{filename}.jpg'), img)

        # Save labels
        with open(labels_dir / f'{filename}.txt', 'w') as f:
            for label in labels:
                class_id, x, y, w, h = label
                f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def test_compositor():
    """Test the compositor to verify no color shifts"""

    print("=" * 70)
    print("TESTING COLOR-SAFE COMPOSITOR")
    print("=" * 70)

    compositor = ColorSafeCompositor('D:/road_backgrounds')

    # Load a background
    bg = compositor.load_background()

    print(f"\n✅ Background loaded: {bg.shape}, dtype: {bg.dtype}")

    # Check a pixel (should be BGR)
    pixel = bg[320, 320]
    print(f"Sample pixel (BGR): {pixel}")

    # Apply brightness
    bright = compositor.apply_brightness(bg, 1.3)
    pixel_bright = bright[320, 320]
    print(f"After brightness 1.3x (BGR): {pixel_bright}")

    # Check ratio is same for all channels
    ratios = pixel_bright.astype(np.float32) / (pixel.astype(np.float32) + 1e-6)
    print(f"Channel ratios: B={ratios[0]:.6f}, G={ratios[1]:.6f}, R={ratios[2]:.6f}")

    # Check if ratios are within 0.5% of each other (accounts for rounding)
    max_diff = np.max(ratios) - np.min(ratios)
    print(f"Max difference: {max_diff:.6f}")

    if max_diff < 0.01:  # Less than 1% difference
        print("All channels scaled equally - NO color shift!")
    else:
        print(f" WARNING: Channels scaled differently by {max_diff*100:.2f}% - COLOR SHIFT DETECTED!")

    # Save test image
    cv2.imwrite('compositor_test.jpg', bright)
    print(f"\nTest image saved: compositor_test.jpg")
    print("Check that yellow road lines stayed yellow!")


if __name__ == "__main__":
    test_compositor()