#!/usr/bin/env python3
"""
Prueba final con prompts narrativos largos usando el código actualizado.
"""

import os
import base64
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()
load_dotenv("config.env")

API_KEY = os.getenv("GEMINI_API_KEY")
OUTPUT_DIR = "test_narrative_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Prompts narrativos originales
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

# El prompt actualizado del código
def build_prompt(narrative):
    return (
        f"coloring book style, black and white line art outline drawing of {narrative}, "
        f"white background, clean thick lines suitable for children coloring page, "
        f"edge to edge drawing, fills the entire frame, no borders, no margins, "
        f"minimal text only, short labels or signs okay, NO paragraphs, "
        f"NO long text blocks, NO story text in the image, "
        f"no shading, no grayscale"
    )

MODELS = {
    "gemini-2.5-flash-image": "generateContent",
    "imagen-4.0-fast-generate-001": "predict",
}


def generate_gemini(model_id, prompt_text, filename):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={API_KEY}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "responseModalities": ["Text", "Image"],
            "temperature": 0.7,
            "imageConfig": {"aspectRatio": "4:3"}
        }
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=120)
        data = resp.json()
        
        text_response = ""
        for part in data.get("candidates", [{}])[0].get("content", {}).get("parts", []):
            if "text" in part:
                text_response = part["text"]
                break
        
        for part in data.get("candidates", [{}])[0].get("content", {}).get("parts", []):
            if "inlineData" in part:
                img_data = base64.b64decode(part["inlineData"]["data"])
                with open(filename, "wb") as f:
                    f.write(img_data)
                return True, text_response
        return False, text_response
    except Exception as e:
        return False, str(e)


def generate_imagen(model_id, prompt_text, filename):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:predict?key={API_KEY}"
    
    payload = {
        "instances": [{"prompt": prompt_text}],
        "parameters": {"sampleCount": 1, "aspectRatio": "4:3"}
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=120)
        data = resp.json()
        
        predictions = data.get("predictions", [])
        if predictions and "bytesBase64Encoded" in predictions[0]:
            img_data = base64.b64decode(predictions[0]["bytesBase64Encoded"])
            with open(filename, "wb") as f:
                f.write(img_data)
            return True, ""
        return False, "No prediction"
    except Exception as e:
        return False, str(e)


def main():
    print("="*70)
    print("🎨 PRUEBA FINAL - Prompts Narrativos Largos")
    print("="*70)
    print("\nRestricciones aplicadas:")
    print("  • minimal text only")
    print("  • short labels or signs okay")
    print("  • NO paragraphs, NO long text blocks")
    print("="*70)
    
    for story_name, story_text in PROMPTS.items():
        print(f"\n📖 {story_name.upper()}")
        print(f"   ({len(story_text)} caracteres)")
        
        full_prompt = build_prompt(story_text)
        
        for model_name, api_type in MODELS.items():
            print(f"\n   🎨 {model_name}")
            
            filename = f"{OUTPUT_DIR}/{story_name}_{model_name.replace('.', '_')}.png"
            
            if api_type == "predict":
                success, text_resp = generate_imagen(model_name, full_prompt, filename)
            else:
                success, text_resp = generate_gemini(model_name, full_prompt, filename)
            
            if success:
                # Verificar dimensiones
                img = Image.open(filename)
                print(f"      ✅ {img.size}")
                if text_resp:
                    preview = text_resp[:60] + "..." if len(text_resp) > 60 else text_resp
                    print(f"      📝 Texto: \"{preview}\"")
            else:
                print(f"      ❌ Error: {text_resp[:50]}")
            
            import time
            time.sleep(2)
    
    print(f"\n{'='*70}")
    print(f"📁 Imágenes guardadas en: {OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
