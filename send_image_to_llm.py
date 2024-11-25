import openai
import numpy as np
from io import BytesIO
from PIL import Image

# Function to send image and text prompt to OpenAI API using an OpenAI client
def send_image_to_llm(image: np.ndarray, prompt: str, openai_client: openai.Client):
    # Convert the numpy array to a PIL Image
    pil_image = Image.fromarray(image)
    
    # Determine the appropriate size for the image (let's resize to the closest valid size)
    image_size = pil_image.size  # Get the original size (width, height)
    
    # The OpenAI API supports sizes like 256x256, 512x512, or 1024x1024
    valid_sizes = [256, 512, 1024]
    
    # Resize the image to the closest valid size while maintaining aspect ratio
    target_size = min(valid_sizes, key=lambda size: abs(size - max(image_size)))  # Find closest size
    pil_image_resized = pil_image.resize((target_size, target_size))
    
    # Save resized image to a BytesIO object
    img_byte_arr = BytesIO()
    pil_image_resized.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)  # Go to the beginning of the BytesIO object
    
    # Send the resized image and prompt to OpenAI using the provided OpenAI client
    try:
        response = openai_client.images.create(
            prompt=prompt,
            n=1,
            size=f"{target_size}x{target_size}",  # Dynamic size based on the image
            file=img_byte_arr
        )
        return response
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    # Example usage
    # First, create an OpenAI client (make sure it's set up with your API key)
    openai.api_key = 'your-openai-api-key'  # Or set it elsewhere
    openai_client = openai

    # Example image (replace with an actual numpy image)
    image = np.random.rand(600, 800, 3) * 255  # Example numpy image array (600x800)
    response = send_image_to_llm(image, "Describe this image", openai_client)
    print(response)
