"""
Utility script to move or copy .nas files to data/raw directory.

Usage:
    python scripts/move_nas_files.py <source_directory> [--copy]
    
    --copy: Copy files instead of moving them (default: move)
"""

import argparse
import os
import shutil
from pathlib import Path


def move_nas_files(source_dir, destination_dir, copy=False):
    """
    Move or copy all .nas files from source to destination.
    
    Args:
        source_dir: Source directory containing .nas files
        destination_dir: Destination directory (data/raw)
        copy: If True, copy files; if False, move files
    """
    source_path = Path(source_dir)
    dest_path = Path(destination_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return
    
    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .nas files
    nas_files = list(source_path.glob("*.nas"))
    nas_files.extend(list(source_path.glob("**/*.nas")))  # Also search subdirectories
    
    if not nas_files:
        print(f"No .nas files found in '{source_dir}'")
        return
    
    print(f"Found {len(nas_files)} .nas file(s)")
    
    action = "Copying" if copy else "Moving"
    for nas_file in nas_files:
        dest_file = dest_path / nas_file.name
        
        # Handle name conflicts
        counter = 1
        while dest_file.exists():
            stem = nas_file.stem
            dest_file = dest_path / f"{stem}_{counter}{nas_file.suffix}"
            counter += 1
        
        try:
            if copy:
                shutil.copy2(nas_file, dest_file)
                print(f"  Copied: {nas_file.name} -> {dest_file.name}")
            else:
                shutil.move(str(nas_file), str(dest_file))
                print(f"  Moved: {nas_file.name} -> {dest_file.name}")
        except Exception as e:
            print(f"  Error processing {nas_file.name}: {e}")
    
    print(f"\n{action.lower().capitalize()} complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Move or copy .nas files to data/raw directory"
    )
    parser.add_argument(
        "source_dir",
        help="Source directory containing .nas files"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them"
    )
    
    args = parser.parse_args()
    
    # Get project root (assuming script is in scripts/ directory)
    project_root = Path(__file__).parent.parent
    destination = project_root / "data" / "raw"
    
    move_nas_files(args.source_dir, destination, copy=args.copy)


if __name__ == "__main__":
    main()
