import base64
import os
from dotenv import load_dotenv
from transformers import pipeline
from PIL import Image
import io

load_dotenv('../.env')

print(f"HF_TOKEN: {os.getenv('HF_TOKEN')}")

# Your base64 image data (with proper padding fix)
image_b64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBwgHBgkIBwgKCgkLDRYPDQwMDRsUFRAWIB0iIiAdHx8kKDQsJCYxJx8fLT0tMTU3Ojo6Iys/RD84QzQ5OjcBCgoKDQwNGg8PGjclHyU3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3N//AABEIAJQAlAMBIgACEQEDEQH/xAAbAAAABwEAAgEAAAAAAAAAAAAAAAIDBAUGBwgJAP/EADQQAAIBAwMCBAQGAwEAAAAAAAECAwAEEiExBRJQQiQVFhcYGRofEjYnGBkbHR4fEykrHwBVIjMkJz4f/EABkBAAMBAQEAAAAAAAAAAAAAAAABAgMEBf/EACMQAAIBAwMCBAUGAwAAAAAAAAECAwAEEiExQVFhcYGRofEjYnGBkbHR4fEykrHwEjM1FDOSsZHx8fHiMtL/2gAMAwEAAhEDEQA/ALAtLVaMClgUOzYQFLAowKUo6VVksAFN3FzDbLqlcD0HrSp5BBC8hGQo/msXxuadwZTqBY+I+goM5W+k6ejpqeOWzkVxj4+yfccc7i51rIFJ3+npV1wji0PEY8BgJRzTPP3Fc5a5miV1TGl/NnrV92ICRzfiLKCSW+jJ8TEBI19zWpYfbjdkzbkN19Lio/s3mKBFCCW5mGq8igVz1izSyMUNSvwczLFQl0p2NkU2wp4024q7BEZxTLipL0w1SyEdtqaannpl6slCDTbGlmm2NWQak81CkyHxfShV0Q0AFLUUAKWBWrNhAUokICxIAAySaMCs12i4kXnNpC3gQ4kI/UfT6VvHBzlRiculWHxXixnburc/lrzbqff4qkuZJJg3iEmf0kUWrf39KPUAMHf710v8AmxVwBh6jsY+0ZUvoq5bYMNgQfet32QshacHibH5kxLv7+g+gxWZe4DDLKuleYxyrXdmZDJwlM7mN2QfGf/dK7WNQh2JiyObst02pRoloNXODCWppzSzTTmoQbamHp45PIc6ubbsdxq7iEqWqxoRkGWQKT9OdRJvgiMy4pl6u+LcB4jwoar60eNTsH2ZT9RVM4q+Chk001OvTLVZBqTzfShSZPNR1os0wFLFAClgVGzYlzpQtyIB3rH3HCtCzTz3Ko5Oy82Zj/wDOdbC6ufwlrMygd46aFY/pzzx74rE3k5kkbJJUNTeuumN/YKdO7Hbfgd9cwLNa2s8sTNoDRoWyfpypm94VdWqT97azx/hmUTa"

# Fix base64 padding
padding_needed = len(image_b64) % 4
if padding_needed:
    image_b64 += "=" * (4 - padding_needed)

print("Processing base64 image...")

# Convert base64 to PIL Image
try:
    image_data = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_data))
    print(f"Image loaded successfully: {image.size}")
except Exception as e:
    print(f"Error loading image: {e}")
    # Fallback to file path
    image_path = r"C:\Emphasis\FirstProject\bird.jpg"
    image = Image.open(image_path)
    print(f"Using file instead: {image.size}")

print("Loading models...")

# Method 1: Image Classification
print("\n=== Image Classification ===")
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
classification_result = classifier(image)
print("Top 3 classifications:")
for i, result in enumerate(classification_result[:3]):
    print(f"{i+1}. {result['label']}: {result['score']:.2f}")

# Method 2: Image Captioning
print("\n=== Image Captioning ===")
captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
caption_result = captioner(image)
print("Caption:", caption_result[0]['generated_text'])

print("\n=== Summary ===")
print("Image analysis complete!")