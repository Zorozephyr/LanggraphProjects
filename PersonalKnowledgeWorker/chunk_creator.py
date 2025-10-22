"""
Create enriched text chunks from MHT files with image descriptions and links
"""
import email
from email import policy
from typing import List, Dict, Any
from pathlib import Path
from bs4 import BeautifulSoup


class ChunkCreator:
    """Creates enriched chunks from MHT files"""
    
    @staticmethod
    def create_enriched_chunks(mht_path: Path, images: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """
        Create chunks with text, image descriptions, and links
        
        Args:
            mht_path: Path to MHT file
            images: Dictionary of processed images
            
        Returns:
            List of enriched chunk dictionaries
        """
        chunks = []
        
        with open(mht_path, 'rb') as f:
            msg = email.message_from_binary_file(f, policy=policy.default)
        
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                html_content = part.get_content()
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Find major sections (excluding paragraphs for better chunking)
                sections = soup.find_all(['div', 'section', 'article'])
                
                for idx, section in enumerate(sections):
                    # Extract text with newline separators
                    text = section.get_text(separator='\n', strip=True)
                    
                    # Skip empty or whitespace-only sections
                    if not text or not text.strip():
                        continue
                    
                    # Extract images in this section
                    section_images = ChunkCreator._extract_section_images(section, images)
                    
                    # Extract links in this section
                    section_links = ChunkCreator._extract_section_links(section)
                    
                    # Build enriched text
                    enriched_text = ChunkCreator._build_enriched_text(
                        text, section_images, section_links
                    )
                    
                    # Create chunk
                    chunks.append({
                        'text': enriched_text,
                        'raw_text': text,
                        'metadata': {
                            'source': str(mht_path),
                            'chunk_id': idx,
                            'images': section_images,
                            'links': section_links,
                            'has_images': len(section_images) > 0,
                            'has_links': len(section_links) > 0
                        }
                    })
        
        return chunks
    
    @staticmethod
    def _extract_section_images(section, images: Dict[str, Dict]) -> List[Dict[str, str]]:
        """Extract and match images in a section"""
        section_images = []
        
        for img in section.find_all('img'):
            img_src = img.get('src', '')
            
            if img_src:
                # Extract filename from path
                filename = img_src.split('/')[-1]
                
                # Find matching image in processed images dict
                matching_key = None
                for key in images.keys():
                    if filename in key:
                        matching_key = key
                        break
                
                if matching_key:
                    section_images.append({
                        'cid': matching_key,
                        'alt': img.get('alt', ''),
                        'description': images[matching_key]['description']
                    })
        
        return section_images
    
    @staticmethod
    def _extract_section_links(section) -> List[Dict[str, str]]:
        """Extract links from a section"""
        section_links = []
        
        for a_tag in section.find_all('a', href=True):
            link_text = a_tag.get_text(strip=True)
            link_url = a_tag['href']
            
            # Skip empty URLs and internal CID references
            if link_url and not link_url.startswith('cid:'):
                section_links.append({
                    'text': link_text or 'Link',
                    'url': link_url,
                    'title': a_tag.get('title', '')
                })
        
        return section_links
    
    @staticmethod
    def _build_enriched_text(text: str, section_images: List[Dict], 
                            section_links: List[Dict]) -> str:
        """Build enriched text with images and links"""
        enriched_text = f"Content: {text}\n\n"
        
        # Add image descriptions
        if section_images:
            enriched_text += "Images in this section:\n"
            for img_info in section_images:
                filename = img_info['cid'].split('/')[-1]
                enriched_text += f"- Image: {filename}\n"
                enriched_text += f"  Description: {img_info['description']}\n"
                if img_info['alt']:
                    enriched_text += f"  Alt text: {img_info['alt']}\n"
            enriched_text += "\n"
        
        # Add links
        if section_links:
            enriched_text += "Links in this section:\n"
            for link_info in section_links:
                enriched_text += f"- Link: {link_info['text']}\n"
                enriched_text += f"  URL: {link_info['url']}\n"
                if link_info['title']:
                    enriched_text += f"  Title: {link_info['title']}\n"
            enriched_text += "\n"
        
        return enriched_text

