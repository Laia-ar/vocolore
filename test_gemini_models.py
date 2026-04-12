#!/usr/bin/env python3
"""
Script de prueba para probar todos los modelos de Gemini disponibles.
Genera imágenes de prueba con diferentes prompts usando cada modelo.
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

# Modelos Gemini disponibles ACTUALIZADOS según la API
GEMINI_MODELS = {
    "gemini-2.5-flash-image": {
        "model_id": "gemini-2.5-flash-image",  # Nano Banana - modelo de imagen oficial
        "description": "Nano Banana - Modelo oficial de generación de imágenes",
        "supports_image": True,
    },
    "gemini-2.5-flash": {
        "model_id": "gemini-2.5-flash",
        "description": "Gemini 2.5 Flash - Modelo más reciente",
        "supports_image": True,  # Probar si soporta imágenes
    },
    "gemini-2.5-pro": {
        "model_id": "gemini-2.5-pro",
        "description": "Gemini 2.5 Pro - Modelo más capaz",
        "supports_image": True,  # Probar si soporta imágenes
    },
    "gemini-2.0-flash": {
        "model_id": "gemini-2.0-flash",
        "description": "Gemini 2.0 Flash - Versión estable",
        "supports_image": True,  # Probar si soporta imágenes
    },
    "gemini-2.0-flash-001": {
        "model_id": "gemini-2.0-flash-001",
        "description": "Gemini 2.0 Flash 001 - Versión específica",
        "supports_image": True,
    },
}

# Prompts de prueba para niños (estilo coloring book)
TEST_PROMPTS = [
    "un gato jugando con una bola de estambre",
    "un cohete espacial volando hacia la luna",
    "un dragón amigable volando sobre un castillo",
    "un dinosaurio comiendo hojas de un árbol",
    "un robot ayudando a una abuela a cruzar la calle",
]


def create_coloring_book_prompt(prompt: str) -> str:
    """Crea un prompt optimizado para páginas de colorear."""
    return (
        f"coloring book style, black and white line art outline drawing of {prompt}, "
        f"white background, clean thick lines suitable for children coloring page, "
        f"no shading, no grayscale, simple shapes, cartoon style"
    )


def test_gemini_model(model_name: str, model_config: dict, prompt: str) -> dict:
    """
    Prueba un modelo de Gemini con un prompt específico.
    
    Returns:
        dict con información del resultado (success, filename, error, etc.)
    """
    if not GEMINI_API_KEY:
        return {"success": False, "error": "GEMINI_API_KEY no está configurado"}
    
    model_id = model_config["model_id"]
    base_url = "https://generativelanguage.googleapis.com/v1beta"
    url = f"{base_url}/models/{model_id}:generateContent?key={GEMINI_API_KEY}"
    
    prompt_text = create_coloring_book_prompt(prompt)
    
    # Configurar payload según el modelo
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
    
    headers = {
        "Content-Type": "application/json",
    }
    
    result = {
        "success": False,
        "model": model_name,
        "prompt": prompt,
        "filename": None,
        "error": None,
        "duration_sec": 0,
    }
    
    print(f"\n{'='*60}")
    print(f"🎨 Probando modelo: {model_name}")
    print(f"📝 Prompt: {prompt}")
    print(f"📋 Descripción: {model_config['description']}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        print(f"⏳ Enviando petición a Gemini API...")
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        result["duration_sec"] = time.time() - start_time
        
        if resp.status_code != 200:
            error_msg = f"Error HTTP {resp.status_code}: {resp.text}"
            print(f"❌ {error_msg}")
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
                        print(f"📸 Imagen encontrada en la respuesta!")
                    elif "text" in part:
                        text_response = part["text"]
                        print(f"📝 Texto en respuesta: {text_response[:100]}...")
        
        if not image_data:
            error_msg = "No se recibieron datos de imagen en la respuesta"
            print(f"⚠️ {error_msg}")
            if text_response:
                print(f"   El modelo respondió con texto: {text_response[:200]}")
            result["error"] = error_msg + (f" (texto: {text_response[:100]})" if text_response else "")
            # No consideramos esto un error total si el modelo respondió
            result["text_response"] = text_response
            return result
        
        # Guardar imagen
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        safe_model = model_name.replace(".", "_")
        safe_prompt = prompt.replace(" ", "_").replace("/", "_")[:30]
        filename = f"{OUTPUT_DIR}/{safe_model}_{safe_prompt}_{int(time.time())}.png"
        
        with open(filename, "wb") as fh:
            fh.write(base64.b64decode(image_data))
        
        result["success"] = True
        result["filename"] = filename
        
        print(f"✅ ¡Éxito! Imagen guardada en: {filename}")
        print(f"⏱️  Tiempo de generación: {result['duration_sec']:.2f}s")
        
        # Verificar tamaño del archivo
        file_size = os.path.getsize(filename)
        print(f"📊 Tamaño del archivo: {file_size / 1024:.1f} KB")
        
    except requests.RequestException as exc:
        result["duration_sec"] = time.time() - start_time
        error_msg = f"Error de conexión: {exc}"
        print(f"❌ {error_msg}")
        result["error"] = error_msg
        
    except Exception as exc:
        result["duration_sec"] = time.time() - start_time
        error_msg = f"Error inesperado: {exc}"
        print(f"❌ {error_msg}")
        result["error"] = error_msg
    
    return result


def print_summary(results: list):
    """Imprime un resumen de los resultados de las pruebas."""
    print(f"\n{'='*70}")
    print("📊 RESUMEN DE PRUEBAS DE MODELOS GEMINI")
    print(f"{'='*70}")
    
    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count
    
    print(f"\n✅ Exitosos: {success_count}")
    print(f"❌ Fallidos: {fail_count}")
    print(f"📈 Total: {len(results)}")
    
    print(f"\n{'─'*70}")
    print("Detalles por modelo:")
    print(f"{'─'*70}")
    
    for result in results:
        status = "✅ OK" if result["success"] else "❌ FAIL"
        print(f"\n{status} | {result['model']}")
        print(f"   Prompt: {result['prompt'][:50]}...")
        if result["success"]:
            print(f"   Archivo: {result['filename']}")
            print(f"   Tiempo: {result['duration_sec']:.2f}s")
        else:
            error = result.get('error', 'Unknown error')
            print(f"   Error: {error[:100]}")
    
    print(f"\n{'='*70}")
    print(f"📁 Imágenes guardadas en: {OUTPUT_DIR}/")
    print(f"{'='*70}")


def main():
    """Función principal de prueba."""
    print("🚀 Iniciando pruebas de modelos Gemini para generación de imágenes")
    print(f"🔑 API Key configurada: {'Sí' if GEMINI_API_KEY else 'No'}")
    
    if not GEMINI_API_KEY:
        print("\n❌ ERROR: Debes configurar GEMINI_API_KEY o GOOGLE_API_KEY en config.env")
        print("   Obtén tu API key en: https://ai.google.dev/")
        sys.exit(1)
    
    results = []
    
    # Probar cada modelo con un prompt
    models = list(GEMINI_MODELS.items())
    prompts = TEST_PROMPTS[:len(models)]  # Usar mismos prompts que modelos
    
    for i, (model_name, model_config) in enumerate(models):
        prompt = prompts[i] if i < len(prompts) else prompts[0]
        result = test_gemini_model(model_name, model_config, prompt)
        results.append(result)
        
        # Pausa breve entre peticiones para no sobrecargar la API
        if i < len(models) - 1:
            print("\n⏳ Pausa de 3 segundos antes de la siguiente prueba...")
            time.sleep(3)
    
    # Imprimir resumen final
    print_summary(results)


if __name__ == "__main__":
    main()
