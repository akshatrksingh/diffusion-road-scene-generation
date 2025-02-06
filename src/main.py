import argparse
import uuid
import os
from datetime import datetime
from generator import RoadSceneGenerator

def generate_unique_filename(base_dir="outputs", extension=".png"):
    """
    Generate a unique filename using UUID and timestamp for better organization.
    The format will be: outputs/YYYY-MM-DD_UUID.png
    """
    # Create the outputs directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Generate current timestamp and UUID
    timestamp = datetime.now().strftime("%Y-%m-%d")
    unique_id = str(uuid.uuid4())
    
    # Combine them into a filename
    filename = f"{timestamp}_{unique_id}{extension}"
    return os.path.join(base_dir, filename)

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Generate road scene images from text descriptions.")
    parser.add_argument("description", type=str, help="Enter the scene description")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Directory to save generated images (default: outputs)")
    args = parser.parse_args()

    try:
        # Initialize the generator
        generator = RoadSceneGenerator()
        
        # Generate the image from description
        image = generator.generate_scene(args.description)
        
        # Generate unique filename and save the image
        output_path = generate_unique_filename(base_dir=args.output_dir)
        image.save(output_path)
        
        # Print both the full path and just the filename for easy reference
        print(f"Generated scene saved as: {output_path}")
        print(f"Filename: {os.path.basename(output_path)}")
        
        # Return the path in case we want to use it programmatically
        return output_path
        
    except Exception as e:
        print(f"Error generating scene: {str(e)}")
        return None

if __name__ == "__main__":
    main()