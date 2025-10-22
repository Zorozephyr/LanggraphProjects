"""
Image extraction and description from MHT files
"""
import email
from email import policy
import base64
import pickle
from typing import Dict
from pathlib import Path


class ImageProcessor:
    """Handles image extraction and AI-powered description"""
    
    def __init__(self, llm, cache_path: Path):
        """
        Initialize ImageProcessor
        
        Args:
            llm: Language model for image description
            cache_path: Path to pickle cache file
        """
        self.llm = llm
        self.cache_path = cache_path
    
    def extract_and_describe_images(self, mht_path: Path, use_cache: bool = True) -> Dict[str, Dict]:
        """
        Extract images from MHT file and get their descriptions
        
        Args:
            mht_path: Path to MHT file
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary mapping image IDs to their data and descriptions
        """
        # Check cache first
        if use_cache and self.cache_path.exists():
            print("Loading images from cache...")
            with open(self.cache_path, 'rb') as f:
                images = pickle.load(f)
            print(f"✓ Loaded {len(images)} images from cache")
            return images
        
        print("Processing images (this may take a while)...")
        images = {}
        
        with open(mht_path, 'rb') as f:
            msg = email.message_from_binary_file(f, policy=policy.default)
        
        image_count = 0
        for part in msg.walk():
            content_type = part.get_content_type()
            
            if content_type.startswith("image/"):
                image_count += 1
                
                # Get image identifier
                cid = part.get('Content-ID', '')
                if cid:
                    cid = cid.strip('<>')
                else:
                    cid = part.get('Content-Location', f'image_{len(images)}')
                
                image_data = part.get_payload(decode=True)
                
                if self.llm:
                    print(f"  Analyzing image {image_count}...")
                    description = self._describe_image(image_data, content_type)
                else:
                    description = "[Image - description unavailable without LLM]"
                
                images[cid] = {
                    'data': image_data,
                    'content_type': content_type,
                    'description': description,
                    'size': len(image_data)
                }
        
        # Save to cache
        print(f"Saving {len(images)} images to cache...")
        with open(self.cache_path, 'wb') as f:
            pickle.dump(images, f)
        print("✓ Cache saved")
        
        return images
    
    def _describe_image(self, image_data: bytes, content_type: str) -> str:
        """
        Use LLM to describe image content
        
        Args:
            image_data: Binary image data
            content_type: Image MIME type
            
        Returns:
            Image description text
        """
        try:
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            response = self.llm.invoke([
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Describe this image in detail for personal note-taking purposes. Include:
- Any text visible in the image
- Diagrams, charts, or visual elements
- Key concepts or information shown
- Any annotations or highlights
Be concise but thorough."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{content_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ])
            
            return response.content
        
        except Exception as e:
            print(f"    ⚠️  Error describing image: {e}")
            return f"[Image - error getting description: {str(e)[:100]}]"
    
    def clear_cache(self):
        """Delete the cache file to force reprocessing"""
        if self.cache_path.exists():
            self.cache_path.unlink()
            print("✓ Cache cleared")
        else:
            print("No cache to clear")

