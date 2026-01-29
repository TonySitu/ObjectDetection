#!/usr/bin/env python3
"""
Automatic fix for Conv2D weights parameter error in keras-yolo3 convert.py
This script fixes the incompatibility with newer Keras versions.
"""

import os
import shutil


def fix_convert_py():
    """Fix the Conv2D weights parameter issue"""

    convert_file = 'convert.py'

    if not os.path.exists(convert_file):
        print(f"Error: {convert_file} not found!")
        print("Make sure you're running this from the keras-yolo3 directory")
        return False

    # Backup
    backup_file = f'{convert_file}.backup3'
    if not os.path.exists(backup_file):
        shutil.copy2(convert_file, backup_file)
        print(f"✓ Created backup: {backup_file}")

    # Read file
    with open(convert_file, 'r') as f:
        lines = f.readlines()

    # Find and fix the problematic section
    new_lines = []
    i = 0
    fixes_applied = 0

    while i < len(lines):
        line = lines[i]

        # Look for Conv2D constructor with weights parameter
        if 'Conv2D(' in line and i + 10 < len(lines):
            # Check if this Conv2D has weights parameter
            conv2d_block = ''.join(lines[i:min(i + 15, len(lines))])

            if 'weights=' in conv2d_block and '))(prev_layer)' in conv2d_block:
                # Found the problematic pattern
                # Collect the entire Conv2D call
                block_lines = []
                j = i
                while j < len(lines):
                    block_lines.append(lines[j])
                    if '))(prev_layer)' in lines[j]:
                        j += 1
                        break
                    j += 1

                # Remove weights parameter from the block
                fixed_block = []
                weights_var = None
                for block_line in block_lines:
                    if 'weights=' in block_line:
                        # Extract the weights variable name
                        import re
                        match = re.search(r'weights=([^,\)]+)', block_line)
                        if match:
                            weights_var = match.group(1).strip()
                        # Skip this line (remove weights parameter)
                        continue
                    fixed_block.append(block_line)

                # Add the fixed block
                new_lines.extend(fixed_block)

                # Add the set_weights call if weights variable was found
                if weights_var:
                    indent = '                '  # Match the indentation
                    new_lines.append(f'{indent}if {weights_var} is not None:\n')
                    new_lines.append(f'{indent}    conv_layer.set_weights({weights_var})\n')

                fixes_applied += 1
                i = j
                continue

        new_lines.append(line)
        i += 1

    # Write back
    with open(convert_file, 'w') as f:
        f.writelines(new_lines)

    print(f"✓ Applied {fixes_applied} fix(es) to Conv2D layers")
    return fixes_applied > 0


if __name__ == '__main__':
    print("=" * 60)
    print("Conv2D Weights Parameter Fix for keras-yolo3")
    print("=" * 60)
    print()

    if fix_convert_py():
        print()
        print("=" * 60)
        print("Fix applied successfully!")
        print("Now run: python3 convert.py yolov3.cfg yolov3.weights model_data/yolo.h5")
        print("=" * 60)
    else:
        print("No fixes were applied. The file may already be fixed.")