#!/usr/bin/env python3
"""
Prueba de prompts narrativos largos con Gemini Image models.
"""

import os
import time
import base64
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
load_dotenv("config.env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
OUTPUT_DIR = "test_gemini_narrative"

# Prompts narrativos de niños
PROMPTS = {
    "parque_rivadavia": """Yo recuerdo que acá en el parque Rivadavia, donde están los Juegos de Madera, jugamos a la pelota ahí en el costadito de los Juegos de Madera.
Entonces mi papá tiró la pelota y otro día, ahí en los Juegos de Madera al costadito, mi papá tiró la pelota y se atacó en el árbol. ¡Y nada más!""",

    "cumple_pikachu": """Yo recuerdo que el 5 de mayo de 2023 fue mi cumpleaños y en la escuela fui primero en la fila de inglés y fui primero en la fila de salir.
Entonces después fui hasta el parque y hice mi cumpleaños.
Pusieron banderines y fue de Pikachu y Isabela mi cumpleaños.
Entonces todos mis amigos fueron ahí y estaban dos animadores, una chica y un chico, que decían que Disney se había borrado y que necesitábamos una varita mágica para ponerlo en color otra vez.
Entonces también teníamos que encontrar una llave para esa caja de la varita.
Entonces fuimos adivinando pistas y fuimos haciendo cosas, retos, adivinanzas y por fin lo logramos.
Nos encontramos la varita mágica, soplamos las velitas y nada más."""
}

# Probar con estos modelos
MODELS = {
    "gemini-2.5-flash-image": "generateContent",
    "imagen-4.0-fast-generate-001": "predict",
}


def create_coloring_book_prompt(prompt: str) -> str:
    return (
        f"coloring book style, black and white line art outline drawing of: {prompt}, "
        f"white background, clean thick lines suitable for children coloring page, "
        f"no shading, no grayscale, simple shapes, cartoon style, children's illustration"
    )


def generate_with_gemini(model_id: str, prompt_text: str) -> tuple[bool, bytes | None, float, str]:
    """Genera imagen con modelo Gemini Image."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": [{"parts": [{"text": create_coloring_book_prompt(prompt_text)}]}],
        "generationConfig": {
            "responseModalities": ["Text", "Image"],
            "temperature": 0.7,
        }
    }
    
    start = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=120)
        duration = time.time() - start
        
        if resp.status_code != 200:
            return False, None, duration, f"HTTP {resp.status_code}: {resp.text[:200]}"
        
        data = resp.json()
        
        if "candidates" in data and len(data["candidates"]) > 0:
            candidate = data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                for part in candidate["content"]["parts"]:
                    if "inlineData" in part:
                        return True, base64.b64decode(part["inlineData"]["data"]), duration, ""
        
        return False, None, duration, "No image data"
    except Exception as exc:
        return False, None, time.time() - start, str(exc)


def generate_with_imagen(model_id: str, prompt_text: str) -> tuple[bool, bytes | None, float, str]:
    """Genera imagen con modelo Imagen."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:predict?key={GEMINI_API_KEY}"
    
    payload = {
        "instances": [{"prompt": create_coloring_book_prompt(prompt_text)}],
        "parameters": {
            "sampleCount": 1,
            "aspectRatio": "3:4",
        }
    }
    
    start = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=120)
        duration = time.time() - start
        
        if resp.status_code != 200:
            return False, None, duration, f"HTTP {resp.status_code}: {resp.text[:200]}"
        
        data = resp.json()
        predictions = data.get("predictions", [])
        
        if predictions and "bytesBase64Encoded" in predictions[0]:
            return True, base64.b64decode(predictions[0]["bytesBase64Encoded"]), duration, ""
        
        return False, None, duration, "No prediction data"
    except Exception as exc:
        return False, None, time.time() - start, str(exc)


def main():
    print("="*80)
    print("🎨 PRUEBA DE PROMPTS NARRATIVOS LARGOS")
    print("="*80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for story_name, story_prompt in PROMPTS.items():
        print(f"\n{'='*80}")
        print(f"📖 Historia: {story_name}")
        print(f"{'='*80}")
        print(f"Texto ({len(story_prompt)} chars):")
        print(f"\"{story_prompt[:200]}...\"")
        
        for model_name, api_type in MODELS.items():
            print(f"\n  🎨 Generando con: {model_name}")
            
            if api_type == "predict":
                success, image_bytes, duration, error = generate_with_imagen(model_name, story_prompt)
            else:
                success, image_bytes, duration, error = generate_with_gemini(model_name, story_prompt)
            
            if success:
                filename = f"{OUTPUT_DIR}/{story_name}_{model_name.replace('.', '_')}.png"
                with open(filename, "wb") as f:
                    f.write(image_bytes)
                print(f"     ✅ Éxito! ({duration:.2f}s) -> {filename}")
            else:
                print(f"     ❌ Error: {error[:100]}")
            
            time.sleep(2)
    
    print(f"\n{'='*80}")
    print(f"📁 Imágenes guardadas en: {OUTPUT_DIR}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
