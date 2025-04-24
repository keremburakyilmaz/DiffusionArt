import requests
from bs4 import BeautifulSoup
import os
import time

import requests
from bs4 import BeautifulSoup
import os
import time
import json
import re

def download_wikiart_artist(artist_name, output_dir, max_images=30):
    """Download paintings from a specific artist on WikiArt"""
    base_url = f"https://www.wikiart.org/en/{artist_name}/all-works/text-list"
   
    # Create output directory
    artist_dir = os.path.join(output_dir, artist_name.replace('-', '_'))
    os.makedirs(artist_dir, exist_ok=True)
   
    # Get list of paintings with headers that mimic a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(base_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
   
    # Find painting links based on the HTML structure shown
    painting_items = soup.find_all('li', class_='painting-list-text-row')
    
    print(f"Found {len(painting_items)} paintings for {artist_name}")
   
    # Download each painting (up to max_images)
    count = 0
    for item in painting_items:
        if count >= max_images:
            break
        
        # Find the link within each list item
        link = item.find('a')
        if not link or not link.get('href'):
            continue
            
        painting_url = "https://www.wikiart.org" + link['href']
        painting_name = link.text.strip().replace(' ', '_').replace('/', '_')
        
        print(f"Downloading: {painting_name}")
        
        try:
            # Get the painting page with browser-like headers
            painting_response = requests.get(painting_url, headers=headers)
            
            # Method 1: Look for JSON data in the page that contains image URLs
            json_data_match = re.search(r'window\.__INITIAL_STATE__\s*=\s*({.*?});', painting_response.text)
            if json_data_match:
                try:
                    json_data = json.loads(json_data_match.group(1))
                    
                    # Navigate through the JSON structure to find the image URL
                    # This path may need to be adjusted based on the actual structure
                    if 'artist' in json_data and 'artworks' in json_data['artist']:
                        # Find the current artwork in the list
                        for artwork in json_data['artist']['artworks']:
                            if artwork.get('id') and painting_url.endswith(str(artwork['id'])):
                                img_url = artwork.get('image')
                                if img_url:
                                    break
                    
                    # Alternative path in the JSON
                    if not img_url and 'painting' in json_data:
                        img_url = json_data['painting'].get('image')
                        
                except json.JSONDecodeError:
                    img_url = None
            else:
                img_url = None
                
            # Method 2: Direct regex search for image URLs if JSON approach fails
            if not img_url:
                # Look for image URLs in the HTML
                img_url_match = re.search(r'"image":\s*"(https?://[^"]+\.(?:jpg|jpeg|png))"', painting_response.text)
                if img_url_match:
                    img_url = img_url_match.group(1)
                    
            # Method 3: If still no success, try to find the og:image meta tag
            if not img_url:
                soup = BeautifulSoup(painting_response.text, 'html.parser')
                og_image = soup.find('meta', property='og:image')
                if og_image and og_image.get('content'):
                    img_url = og_image['content']
            
            # Download the image if we found a URL
            if img_url:
                # Ensure the URL is complete
                if not img_url.startswith(('http://', 'https://')):
                    img_url = "https:" + img_url if img_url.startswith('//') else "https://www.wikiart.org" + img_url
                
                print(f"Found image URL: {img_url}")
                
                # Download the image
                img_response = requests.get(img_url, headers=headers, timeout=10)
                if img_response.status_code == 200:
                    with open(os.path.join(artist_dir, f"{painting_name}.jpg"), "wb") as f:
                        f.write(img_response.content)
                    count += 1
                    print(f"Successfully downloaded {count}/{max_images}")
                else:
                    print(f"Failed to download image: HTTP status {img_response.status_code}")
            else:
                print(f"Could not find image URL for {painting_name}")
                # Save the response for debugging
                with open(os.path.join(artist_dir, f"{painting_name}_debug.html"), "w", encoding="utf-8") as f:
                    f.write(painting_response.text)
                
        except Exception as e:
            print(f"Error downloading {painting_name}: {str(e)}")
        
        # Add a small delay to avoid overwhelming the server
        time.sleep(2)

# Main execution
artists = [
    "vincent-van-gogh",      # Post-Impressionism with swirling brushstrokes
    "pablo-picasso",         # Cubism and various experimental styles
    "claude-monet",          # Impressionism with focus on light and atmosphere
    "edvard-munch",          # Expressionism with emotional intensity
    "salvador-dali",         # Surrealism with dreamlike imagery
    "frida-kahlo",           # Symbolic self-portraits with Mexican influences
    "andy-warhol",           # Pop Art with bright colors and celebrity imagery
    "henri-matisse",         # Fauvism and bold color compositions
    "gustav-klimt",          # Art Nouveau with decorative patterns and gold leaf
    "wassily-kandinsky",     # Abstract art with musical influences
    "jackson-pollock",       # Abstract Expressionism with drip painting technique
    "georgia-okeeffe",       # Magnified flowers and desert landscapes
    "hokusai",               # Japanese woodblock prints
    "roy-lichtenstein",      # Pop Art with comic book style
    "marc-chagall",          # Dreamlike imagery with floating figures
    "rene-magritte",         # Surrealism with conceptual juxtapositions
    "paul-cezanne",          # Post-Impressionism with geometric structure
    "egon-schiele",          # Expressionism with contorted figures
    "piet-mondrian",         # Geometric abstraction with primary colors
    "william-turner"         # Romantic landscapes with atmospheric effects
]

output_directory = "./raw_images/style"
os.makedirs(output_directory, exist_ok=True)

for artist in artists:
    print(f"\nProcessing artist: {artist}")
    download_wikiart_artist(artist, output_directory)
    # Add a delay between artists to be respectful to the website
    time.sleep(3)