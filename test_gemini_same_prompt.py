#!/usr/bin/env python3
"""
Script de prueba para comparar TODOS los modelos de imagen de Gemini/Imagen
usando el MISMO prompt.
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
OUTPUT_DIR = "test_gemini_same_prompt"

# ============================================================
# TODOS LOS MODELOS
# ============================================================
ALL_MODELS = {
    # Modelos Gemini Image (generateContent)
    "gemini-2.5-flash-image": {
        "model_id": "gemini-2.5-flash-image",
        "api_type": "generateContent",
        "category": "Gemini Image",
    },
    "gemini-3-pro-image-preview": {
        "model_id": "gemini-3-pro-image-preview",
        "api_type": "generateContent",
        "category": "Gemini Image",
    },
    "nano-banana-pro-preview": {
        "model_id": "nano-banana-pro-preview",
        "api_type": "generateContent",
        "category": "Gemini Image",
    },
    "gemini-3.1-flash-image-preview": {
        "model_id": "gemini-3.1-flash-image-preview",
        "api_type": "generateContent",
        "category": "Gemini Image",
    },
    # Modelos Imagen (predict)
    "imagen-4.0-generate-001": {
        "model_id": "imagen-4.0-generate-001",
        "api_type": "predict",
        "category": "Imagen",
    },
    "imagen-4.0-ultra-generate-001": {
        "model_id": "imagen-4.0-ultra-generate-001",
        "api_type": "predict",
        "category": "Imagen",
    },
    "imagen-4.0-fast-generate-001": {
        "model_id": "imagen-4.0-fast-generate-001",
        "api_type": "predict",
        "category": "Imagen",
    },
}

# Prompt único para comparación justa
TEST_PROMPT = "un astronauta plantando una bandera en la luna con un cohete en el fondo"


def create_coloring_book_prompt(prompt: str) -> str:
    """Crea un prompt optimizado para páginas de colorear."""
    return (
        f"coloring book style, black and white line art outline drawing of {prompt}, "
        f"white background, clean thick lines suitable for children coloring page, "
        f"no shading, no grayscale, simple shapes, cartoon style"
    )


def test_gemini_generate_content(model_name: str, model_config: dict, prompt: str) -> dict:
    """Prueba un modelo Gemini usando generateContent."""
    model_id = model_config["model_id"]
    base_url = "https://generativelanguage.googleapis.com/v1beta"
    url = f"{base_url}/models/{model_id}:generateContent?key={GEMINI_API_KEY}"
    
    prompt_text = create_coloring_book_prompt(prompt)
    
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "responseModalities": ["Text", "Image"],
            "temperature": 0.7,
        }
    }
    
    headers = {"Content-Type": "application/json"}
    
    result = {
        "success": False,
        "model": model_name,
        "category": model_config["category"],
        "filename": None,
        "error": None,
        "duration_sec": 0,
        "file_size_kb": 0,
        "dimensions": None,
    }
    
    start_time = time.time()
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        result["duration_sec"] = time.time() - start_time
        
        if resp.status_code != 200:
            result["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
            return result
        
        data = resp.json()
        
        image_data = None
        if "candidates" in data and len(data["candidates"]) > 0:
            candidate = data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                for part in candidate["content"]["parts"]:
                    if "inlineData" in part:
                        image_data = part["inlineData"]["data"]
                        break
        
        if not image_data:
            result["error"] = "No image data in response"
            return result
        
        # Guardar imagen
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        safe_model = model_name.replace(".", "_")
        filename = f"{OUTPUT_DIR}/{safe_model}.png"
        
        image_bytes = base64.b64decode(image_data)
        with open(filename, "wb") as fh:
            fh.write(image_bytes)
        
        result["success"] = True
        result["filename"] = filename
        result["file_size_kb"] = len(image_bytes) / 1024
        
        # Intentar obtener dimensiones
        try:
            from PIL import Image
            with Image.open(filename) as img:
                result["dimensions"] = f"{img.width}x{img.height}"
        except:
            pass
        
    except Exception as exc:
        result["duration_sec"] = time.time() - start_time
        result["error"] = str(exc)
    
    return result


def test_imagen_predict(model_name: str, model_config: dict, prompt: str) -> dict:
    """Prueba un modelo Imagen usando predict."""
    model_id = model_config["model_id"]
    base_url = "https://generativelanguage.googleapis.com/v1beta"
    url = f"{base_url}/models/{model_id}:predict?key={GEMINI_API_KEY}"
    
    prompt_text = create_coloring_book_prompt(prompt)
    
    payload = {
        "instances": [{"prompt": prompt_text}],
        "parameters": {
            "sampleCount": 1,
            "aspectRatio": "3:4",
        }
    }
    
    headers = {"Content-Type": "application/json"}
    
    result = {
        "success": False,
        "model": model_name,
        "category": model_config["category"],
        "filename": None,
        "error": None,
        "duration_sec": 0,
        "file_size_kb": 0,
        "dimensions": None,
    }
    
    start_time = time.time()
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        result["duration_sec"] = time.time() - start_time
        
        if resp.status_code != 200:
            result["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
            return result
        
        data = resp.json()
        predictions = data.get("predictions", [])
        
        if not predictions:
            result["error"] = "No predictions in response"
            return result
        
        image_data = predictions[0].get("bytesBase64Encoded")
        if not image_data:
            result["error"] = "No image data in prediction"
            return result
        
        # Guardar imagen
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        safe_model = model_name.replace(".", "_")
        filename = f"{OUTPUT_DIR}/{safe_model}.png"
        
        image_bytes = base64.b64decode(image_data)
        with open(filename, "wb") as fh:
            fh.write(image_bytes)
        
        result["success"] = True
        result["filename"] = filename
        result["file_size_kb"] = len(image_bytes) / 1024
        
        # Intentar obtener dimensiones
        try:
            from PIL import Image
            with Image.open(filename) as img:
                result["dimensions"] = f"{img.width}x{img.height}"
        except:
            pass
        
    except Exception as exc:
        result["duration_sec"] = time.time() - start_time
        result["error"] = str(exc)
    
    return result


def print_summary(results: list, prompt: str):
    """Imprime un resumen comparativo."""
    print(f"\n{'='*80}")
    print("📊 COMPARACIÓN DE MODELOS - MISMO PROMPT")
    print(f"{'='*80}")
    print(f"📝 Prompt: \"{prompt}\"")
    print(f"{'='*80}")
    
    # Separar por categoría
    gemini_results = [r for r in results if r["category"] == "Gemini Image"]
    imagen_results = [r for r in results if r["category"] == "Imagen"]
    
    # Ordenar por tiempo
    gemini_results.sort(key=lambda x: x["duration_sec"] if x["success"] else float('inf'))
    imagen_results.sort(key=lambda x: x["duration_sec"] if x["success"] else float('inf'))
    
    print("\n" + "="*80)
    print("🤖 MODELOS GEMINI IMAGE")
    print("="*80)
    print(f"{'Modelo':<35} {'Tiempo':>10} {'Tamaño':>10} {'Dimensiones':>12} {'Estado':>8}")
    print("-"*80)
    for r in gemini_results:
        status = "✅" if r["success"] else "❌"
        time_str = f"{r['duration_sec']:.2f}s" if r["success"] else "N/A"
        size_str = f"{r['file_size_kb']:.1f}KB" if r["success"] else "N/A"
        dim_str = r.get("dimensions", "N/A") or "N/A"
        print(f"{r['model']:<35} {time_str:>10} {size_str:>10} {dim_str:>12} {status:>8}")
    
    print("\n" + "="*80)
    print("🎨 MODELOS IMAGEN")
    print("="*80)
    print(f"{'Modelo':<35} {'Tiempo':>10} {'Tamaño':>10} {'Dimensiones':>12} {'Estado':>8}")
    print("-"*80)
    for r in imagen_results:
        status = "✅" if r["success"] else "❌"
        time_str = f"{r['duration_sec']:.2f}s" if r["success"] else "N/A"
        size_str = f"{r['file_size_kb']:.1f}KB" if r["success"] else "N/A"
        dim_str = r.get("dimensions", "N/A") or "N/A"
        print(f"{r['model']:<35} {time_str:>10} {size_str:>10} {dim_str:>12} {status:>8}")
    
    # Ranking general
    print("\n" + "="*80)
    print("🏆 RANKING POR VELOCIDAD (todos los modelos)")
    print("="*80)
    all_success = [r for r in results if r["success"]]
    all_success.sort(key=lambda x: x["duration_sec"])
    
    for i, r in enumerate(all_success, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{medal} #{i} {r['model']:<35} - {r['duration_sec']:.2f}s")
    
    print(f"\n{'='*80}")
    print(f"📁 Imágenes guardadas en: {OUTPUT_DIR}/")
    print(f"{'='*80}")


def main():
    """Función principal."""
    print("🚀 PRUEBA COMPARATIVA - MISMO PROMPT EN TODOS LOS MODELOS")
    print("="*80)
    
    if not GEMINI_API_KEY:
        print("\n❌ ERROR: Configura GEMINI_API_KEY o GOOGLE_API_KEY en config.env")
        sys.exit(1)
    
    print(f"\n📝 Prompt de prueba:")
    print(f"   \"{TEST_PROMPT}\"")
    print(f"\n🎯 Modelos a probar: {len(ALL_MODELS)}")
    
    results = []
    
    for i, (model_name, model_config) in enumerate(ALL_MODELS.items()):
        print(f"\n{'='*80}")
        print(f"[{i+1}/{len(ALL_MODELS)}] Probando: {model_name}")
        print(f"Categoría: {model_config['category']}")
        print(f"{'='*80}")
        
        if model_config["api_type"] == "predict":
            result = test_imagen_predict(model_name, model_config, TEST_PROMPT)
        else:
            result = test_gemini_generate_content(model_name, model_config, TEST_PROMPT)
        
        results.append(result)
        
        if result["success"]:
            print(f"✅ ÉXITO - Tiempo: {result['duration_sec']:.2f}s")
            print(f"   Archivo: {result['filename']}")
            print(f"   Tamaño: {result['file_size_kb']:.1f} KB")
            if result.get("dimensions"):
                print(f"   Dimensiones: {result['dimensions']}")
        else:
            print(f"❌ ERROR: {result['error'][:100]}")
        
        # Pausa entre peticiones
        if i < len(ALL_MODELS) - 1:
            print("\n⏳ Pausa de 2s...")
            time.sleep(2)
    
    # Imprimir resumen
    print_summary(results, TEST_PROMPT)
    
    # Guardar resultados en JSON
    import json
    json_path = f"{OUTPUT_DIR}/results.json"
    with open(json_path, "w") as f:
        # Hacer serializable
        json_results = []
        for r in results:
            jr = dict(r)
            jr.pop("filename", None)
            json_results.append(jr)
        json.dump(json_results, f, indent=2)
    print(f"\n📄 Resultados guardados en: {json_path}")


if __name__ == "__main__":
    main()
