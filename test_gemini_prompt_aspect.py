#!/usr/bin/env python3
"""
Prueba: ¿Podemos influir el aspect ratio de Gemini Image mediante el prompt?
"""

import os
import base64
import requests
from dotenv import load_dotenv

load_dotenv()
load_dotenv("config.env")

API_KEY = os.getenv("GEMINI_API_KEY")
OUTPUT_DIR = "test_gemini_prompt_aspect"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL = "gemini-2.5-flash-image"

# Mismo tema, diferentes indicaciones de composición
PROMPTS = {
    "default": "a cat sitting on a fence, coloring book style, black and white line art",
    
    "square_explicit": "a cat sitting on a fence, square composition, coloring book style, black and white line art",
    
    "landscape_explicit": "a cat sitting on a fence, wide horizontal composition, landscape view, panoramic, coloring book style, black and white line art",
    
    "portrait_explicit": "a cat sitting on a fence, tall vertical composition, portrait orientation, coloring book style, black and white line art",
    
    "wide_shot": "a wide shot of a cat sitting on a fence showing the full garden background, horizontal composition, coloring book style",
    
    "close_up": "a close-up portrait of a cat face, vertical composition, filling the frame, coloring book style",
}


def generate(model_id: str, prompt_text: str, filename: str):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={API_KEY}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "responseModalities": ["Text", "Image"],
            "temperature": 0.7,
        }
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=60)
        data = resp.json()
        
        if "candidates" in data and len(data["candidates"]) > 0:
            candidate = data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                for part in candidate["content"]["parts"]:
                    if "inlineData" in part:
                        img_data = base64.b64decode(part["inlineData"]["data"])
                        width = int.from_bytes(img_data[16:20], 'big')
                        height = int.from_bytes(img_data[20:24], 'big')
                        
                        with open(filename, "wb") as f:
                            f.write(img_data)
                        return width, height
        return None, None
    except Exception as exc:
        print(f"Error: {exc}")
        return None, None


print("="*80)
print("🧪 PRUEBA: ¿Influir aspect ratio mediante prompt?")
print(f"Modelo: {MODEL}")
print("="*80)

results = []

for name, prompt in PROMPTS.items():
    print(f"\n📝 {name}:")
    print(f"   Prompt: {prompt[:60]}...")
    
    filename = f"{OUTPUT_DIR}/{name}.png"
    width, height = generate(MODEL, prompt, filename)
    
    if width and height:
        ratio = width / height
        ratio_str = f"{width}x{height}"
        
        if ratio > 1.3:
            orientation = "LANDSCAPE"
        elif ratio < 0.8:
            orientation = "PORTRAIT"
        else:
            orientation = "SQUARE"
        
        print(f"   ✅ {ratio_str} ({orientation}) - ratio: {ratio:.2f}")
        results.append({"name": name, "dims": ratio_str, "orientation": orientation, "ratio": ratio})
    else:
        print(f"   ❌ Falló")

print("\n" + "="*80)
print("📊 RESUMEN")
print("="*80)
print(f"{'Prompt':<20} {'Dimensiones':<12} {'Orientación':<12} {'Ratio':<8}")
print("-"*80)
for r in results:
    print(f"{r['name']:<20} {r['dims']:<12} {r['orientation']:<12} {r['ratio']:.2f}")

print("\n" + "="*80)
print("💡 CONCLUSIÓN:")
print("="*80)
landscapes = sum(1 for r in results if r['orientation'] == 'LANDSCAPE')
portraits = sum(1 for r in results if r['orientation'] == 'PORTRAIT')
squares = sum(1 for r in results if r['orientation'] == 'SQUARE')

print(f"Landscape: {landscapes} | Portrait: {portraits} | Square: {squares}")
print(f"\n¿El prompt influye? {'SÍ' if landscapes > 0 or portraits > 0 else 'NO'}")
