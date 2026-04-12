#!/usr/bin/env python3
"""
Script de prueba completo para probar TODOS los modelos de imagen de Gemini/Imagen.
Incluye modelos Gemini Image y modelos Imagen.
"""

import os
import sys
import time
import base64
from pathlib import Path

import requests
from dotenv import load_dotenv

# Cargar configuración
load_dotenv()
load_dotenv("config.env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
OUTPUT_DIR = "test_gemini_outputs"

# ============================================================
# MODELOS GEMINI (usan generateContent)
# ============================================================
GEMINI_IMAGE_MODELS = {
    "gemini-2.5-flash-image": {
        "model_id": "gemini-2.5-flash-image",
        "display_name": "Nano Banana",
        "api_type": "generateContent",
        "description": "Modelo oficial de generación de imágenes - Gemini 2.5 Flash Image",
    },
    "gemini-3-pro-image-preview": {
        "model_id": "gemini-3-pro-image-preview",
        "display_name": "Nano Banana Pro",
        "api_type": "generateContent",
        "description": "Versión Pro del modelo de imagen",
    },
    "nano-banana-pro-preview": {
        "model_id": "nano-banana-pro-preview",
        "display_name": "Nano Banana Pro (Preview)",
        "api_type": "generateContent",
        "description": "Versión preview de Nano Banana Pro",
    },
    "gemini-3.1-flash-image-preview": {
        "model_id": "gemini-3.1-flash-image-preview",
        "display_name": "Nano Banana 2",
        "api_type": "generateContent",
        "description": "Versión 2 del modelo de imagen",
    },
}

# ============================================================
# MODELOS IMAGEN (usan predict)
# ============================================================
IMAGEN_MODELS = {
    "imagen-4.0-generate-001": {
        "model_id": "imagen-4.0-generate-001",
        "display_name": "Imagen 4",
        "api_type": "predict",
        "description": "Modelo Imagen 4 estándar",
    },
    "imagen-4.0-ultra-generate-001": {
        "model_id": "imagen-4.0-ultra-generate-001",
        "display_name": "Imagen 4 Ultra",
        "api_type": "predict",
        "description": "Versión Ultra de Imagen 4",
    },
    "imagen-4.0-fast-generate-001": {
        "model_id": "imagen-4.0-fast-generate-001",
        "display_name": "Imagen 4 Fast",
        "api_type": "predict",
        "description": "Versión rápida de Imagen 4",
    },
}

# Prompts de prueba para niños (estilo coloring book)
TEST_PROMPTS = [
    "un gato jugando con una bola de estambre",
    "un cohete espacial volando hacia la luna",
    "un dragón amigable volando sobre un castillo",
    "un dinosaurio comiendo hojas de un árbol",
    "un robot ayudando a una abuela a cruzar la calle",
    "una mariposa posada en una flor",
    "un tren pasando por un puente",
]


def create_coloring_book_prompt(prompt: str) -> str:
    """Crea un prompt optimizado para páginas de colorear."""
    return (
        f"coloring book style, black and white line art outline drawing of {prompt}, "
        f"white background, clean thick lines suitable for children coloring page, "
        f"no shading, no grayscale, simple shapes, cartoon style"
    )


def test_gemini_generate_content(model_name: str, model_config: dict, prompt: str) -> dict:
    """Prueba un modelo Gemini usando la API generateContent."""
    
    model_id = model_config["model_id"]
    base_url = "https://generativelanguage.googleapis.com/v1beta"
    url = f"{base_url}/models/{model_id}:generateContent?key={GEMINI_API_KEY}"
    
    prompt_text = create_coloring_book_prompt(prompt)
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt_text
            }]
        }],
        "generationConfig": {
            "responseModalities": ["Text", "Image"],
            "temperature": 0.7,
        }
    }
    
    headers = {"Content-Type": "application/json"}
    
    result = {
        "success": False,
        "model": model_name,
        "prompt": prompt,
        "filename": None,
        "error": None,
        "duration_sec": 0,
    }
    
    print(f"\n{'='*70}")
    print(f"🎨 Modelo: {model_config['display_name']} ({model_name})")
    print(f"📝 Prompt: {prompt}")
    print(f"📋 {model_config['description']}")
    print(f"🔌 API: generateContent")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        print(f"⏳ Enviando petición...")
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        result["duration_sec"] = time.time() - start_time
        
        if resp.status_code != 200:
            error_msg = f"Error HTTP {resp.status_code}: {resp.text}"
            print(f"❌ {error_msg[:200]}")
            result["error"] = error_msg
            return result
        
        data = resp.json()
        
        # Extraer imagen de la respuesta
        image_data = None
        text_response = None
        
        if "candidates" in data and len(data["candidates"]) > 0:
            candidate = data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                for part in candidate["content"]["parts"]:
                    if "inlineData" in part:
                        image_data = part["inlineData"]["data"]
                        mime_type = part["inlineData"].get("mimeType", "image/png")
                        print(f"📸 Imagen recibida! (MIME: {mime_type})")
                    elif "text" in part:
                        text_response = part["text"]
        
        if not image_data:
            error_msg = "No image data received"
            print(f"⚠️ {error_msg}")
            if text_response:
                print(f"   Texto: {text_response[:150]}...")
            result["error"] = error_msg
            return result
        
        # Guardar imagen
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        safe_model = model_name.replace(".", "_")
        safe_prompt = prompt.replace(" ", "_").replace("/", "_")[:25]
        filename = f"{OUTPUT_DIR}/{safe_model}_{safe_prompt}_{int(time.time())}.png"
        
        with open(filename, "wb") as fh:
            fh.write(base64.b64decode(image_data))
        
        result["success"] = True
        result["filename"] = filename
        
        file_size = os.path.getsize(filename)
        print(f"✅ Imagen guardada: {filename}")
        print(f"📊 Tamaño: {file_size / 1024:.1f} KB | Tiempo: {result['duration_sec']:.2f}s")
        
    except requests.RequestException as exc:
        result["duration_sec"] = time.time() - start_time
        error_msg = f"Error de conexión: {exc}"
        print(f"❌ {error_msg}")
        result["error"] = error_msg
        
    except Exception as exc:
        result["duration_sec"] = time.time() - start_time
        error_msg = f"Error: {exc}"
        print(f"❌ {error_msg}")
        result["error"] = error_msg
    
    return result


def test_imagen_predict(model_name: str, model_config: dict, prompt: str) -> dict:
    """Prueba un modelo Imagen usando la API predict."""
    
    model_id = model_config["model_id"]
    base_url = "https://generativelanguage.googleapis.com/v1beta"
    url = f"{base_url}/models/{model_id}:predict?key={GEMINI_API_KEY}"
    
    prompt_text = create_coloring_book_prompt(prompt)
    
    # Payload específico para Imagen
    payload = {
        "instances": [
            {"prompt": prompt_text}
        ],
        "parameters": {
            "sampleCount": 1,
            "aspectRatio": "4:3",  # Landscape para páginas de colorear
        }
    }
    
    headers = {"Content-Type": "application/json"}
    
    result = {
        "success": False,
        "model": model_name,
        "prompt": prompt,
        "filename": None,
        "error": None,
        "duration_sec": 0,
    }
    
    print(f"\n{'='*70}")
    print(f"🎨 Modelo: {model_config['display_name']} ({model_name})")
    print(f"📝 Prompt: {prompt}")
    print(f"📋 {model_config['description']}")
    print(f"🔌 API: predict (Imagen)")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        print(f"⏳ Enviando petición...")
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        result["duration_sec"] = time.time() - start_time
        
        if resp.status_code != 200:
            error_msg = f"Error HTTP {resp.status_code}: {resp.text}"
            print(f"❌ {error_msg[:200]}")
            result["error"] = error_msg
            return result
        
        data = resp.json()
        
        # Imagen devuelve las imágenes en predictions
        predictions = data.get("predictions", [])
        if not predictions:
            error_msg = "No predictions in response"
            print(f"⚠️ {error_msg}")
            print(f"   Respuesta: {data}")
            result["error"] = error_msg
            return result
        
        # Imagen devuelve bytesBase64Encoded
        image_data = predictions[0].get("bytesBase64Encoded")
        if not image_data:
            error_msg = "No image data in prediction"
            print(f"⚠️ {error_msg}")
            result["error"] = error_msg
            return result
        
        print(f"📸 Imagen recibida!")
        
        # Guardar imagen
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        safe_model = model_name.replace(".", "_")
        safe_prompt = prompt.replace(" ", "_").replace("/", "_")[:25]
        filename = f"{OUTPUT_DIR}/{safe_model}_{safe_prompt}_{int(time.time())}.png"
        
        with open(filename, "wb") as fh:
            fh.write(base64.b64decode(image_data))
        
        result["success"] = True
        result["filename"] = filename
        
        file_size = os.path.getsize(filename)
        print(f"✅ Imagen guardada: {filename}")
        print(f"📊 Tamaño: {file_size / 1024:.1f} KB | Tiempo: {result['duration_sec']:.2f}s")
        
    except requests.RequestException as exc:
        result["duration_sec"] = time.time() - start_time
        error_msg = f"Error de conexión: {exc}"
        print(f"❌ {error_msg}")
        result["error"] = error_msg
        
    except Exception as exc:
        result["duration_sec"] = time.time() - start_time
        error_msg = f"Error: {exc}"
        print(f"❌ {error_msg}")
        result["error"] = error_msg
    
    return result


def print_summary(results: list):
    """Imprime un resumen de los resultados."""
    print(f"\n{'='*70}")
    print("📊 RESUMEN DE PRUEBAS - TODOS LOS MODELOS DE IMAGEN")
    print(f"{'='*70}")
    
    gemini_results = [r for r in results if "imagen" not in r["model"]]
    imagen_results = [r for r in results if "imagen" in r["model"]]
    
    print(f"\n🤖 Modelos Gemini (generateContent):")
    print(f"   ✅ Exitosos: {sum(1 for r in gemini_results if r['success'])}")
    print(f"   ❌ Fallidos: {sum(1 for r in gemini_results if not r['success'])}")
    
    print(f"\n🎨 Modelos Imagen (predict):")
    print(f"   ✅ Exitosos: {sum(1 for r in imagen_results if r['success'])}")
    print(f"   ❌ Fallidos: {sum(1 for r in imagen_results if not r['success'])}")
    
    print(f"\n{'─'*70}")
    print("Resultados detallados:")
    print(f"{'─'*70}")
    
    for result in results:
        status = "✅ OK" if result["success"] else "❌ FAIL"
        model_short = result['model'][:35]
        print(f"\n{status} | {model_short}")
        if result["success"]:
            print(f"   📝 {result['prompt'][:40]}...")
            print(f"   📁 {result['filename']}")
            print(f"   ⏱️  {result['duration_sec']:.2f}s")
        else:
            error = result.get('error', 'Unknown')
            print(f"   ⚠️  {error[:60]}...")
    
    print(f"\n{'='*70}")
    print(f"📁 Directorio de salida: {OUTPUT_DIR}/")
    print(f"{'='*70}")


def main():
    """Función principal."""
    print("🚀 PRUEBA COMPLETA - Todos los modelos de imagen de Google")
    print("="*70)
    print("Modelos a probar:")
    print("  • Gemini Image Models (Nano Banana series)")
    print("  • Imagen 4 Models (Imagen 4, Ultra, Fast)")
    print("="*70)
    
    if not GEMINI_API_KEY:
        print("\n❌ ERROR: Configura GEMINI_API_KEY o GOOGLE_API_KEY en config.env")
        print("   Obtén tu API key en: https://ai.google.dev/")
        sys.exit(1)
    
    results = []
    
    # ============================================================
    # PROBAR MODELOS GEMINI (generateContent)
    # ============================================================
    print(f"\n{'='*70}")
    print("🤙 PROBANDO MODELOS GEMINI (API: generateContent)")
    print(f"{'='*70}")
    
    for i, (model_name, model_config) in enumerate(GEMINI_IMAGE_MODELS.items()):
        prompt = TEST_PROMPTS[i % len(TEST_PROMPTS)]
        result = test_gemini_generate_content(model_name, model_config, prompt)
        results.append(result)
        
        if i < len(GEMINI_IMAGE_MODELS) - 1:
            print("\n⏳ Pausa de 2s...")
            time.sleep(2)
    
    # ============================================================
    # PROBAR MODELOS IMAGEN (predict)
    # ============================================================
    print(f"\n{'='*70}")
    print("🎨 PROBANDO MODELOS IMAGEN (API: predict)")
    print(f"{'='*70}")
    
    for i, (model_name, model_config) in enumerate(IMAGEN_MODELS.items()):
        prompt = TEST_PROMPTS[i % len(TEST_PROMPTS)]
        result = test_imagen_predict(model_name, model_config, prompt)
        results.append(result)
        
        if i < len(IMAGEN_MODELS) - 1:
            print("\n⏳ Pausa de 2s...")
            time.sleep(2)
    
    # Resumen final
    print_summary(results)


if __name__ == "__main__":
    main()
