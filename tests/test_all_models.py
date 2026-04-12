#!/usr/bin/env python3
"""
Script de prueba para generar imágenes con TODOS los modelos disponibles
usando las narrativas de prueba del niño.

Uso: cd /home/didi/code/vocolore && uv run python test_data/test_all_models.py
"""

import os
import sys
import time
import requests
from datetime import datetime

# Agregar directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wifi_transcribe import (
    FREEPIK_MODELS,
    GEMINI_MODELS,
    get_freepik_model_config,
    build_freepik_payload,
    _call_gemini_generate_content,
    _call_imagen_predict,
    GEMINI_API_KEY,
)
from dotenv import load_dotenv

load_dotenv()

FREEPIK_API_KEY = os.getenv("FREEPIK_API_KEY")

# Narrativas de prueba - prompts simplificados para evitar texto
NARRATIVAS = {
    "parque_rivadavia": {
        "titulo": "Parque Rivadavia - Juegos de Madera",
        "prompt": "children playing ball near wooden playground structure in a park, ball stuck in a tree, father and child, outdoor fun, family moment"
    },
    "cumpleanos_pikachu": {
        "titulo": "Cumpleaños Pikachu - 5 de mayo 2023", 
        "prompt": "children birthday party in a park with yellow electric mouse character decorations, colorful pennant banners, kids celebrating, animators with magic wand, mystery box with key, birthday cake with candles, games and challenges"
    }
}


def guardar_imagen(image_data: bytes, modelo: str, narrativa: str, proveedor: str) -> str:
    """Guarda la imagen generada con nombre descriptivo."""
    timestamp = int(time.time())
    safe_modelo = modelo.replace("-", "_").replace(".", "_")
    safe_narrativa = narrativa.replace(" ", "_").lower()
    filename = f"test_data/img_{safe_narrativa}_{proveedor}_{safe_modelo}_{timestamp}.png"
    
    with open(filename, "wb") as f:
        f.write(image_data)
    
    return filename


def send_freepik_request(prompt: str, model_name: str) -> tuple[bytes | None, str | None]:
    """Llama a la API de Freepik y retorna (image_data, error)."""
    if not FREEPIK_API_KEY:
        return None, "FREEPIK_API_KEY not set"
    
    config = get_freepik_model_config(model_name)
    base_url = "https://api.freepik.com"
    url = f"{base_url}{config['endpoint']}"
    
    payload = build_freepik_payload(model_name, prompt)
    headers = {
        "Content-Type": "application/json",
        "x-freepik-api-key": FREEPIK_API_KEY,
    }
    
    try:
        # Initial request
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        # Get task_id
        task_id = data.get("data", {}).get("task_id") or data.get("task_id") or data.get("id")
        if not task_id:
            return None, "No task_id returned from Freepik"
        
        # Poll for completion
        poll_deadline = time.time() + 120
        image_url = None
        while time.time() < poll_deadline:
            poll_resp = requests.get(f"{url}/{task_id}", headers=headers, timeout=15)
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()
            
            status = poll_data.get("data", {}).get("status") or poll_data.get("status")
            
            if status == "COMPLETED":
                # Try different locations for image URL
                image_url = (poll_data.get("data", {}).get("image", {}).get("url") or 
                            poll_data.get("data", {}).get("images", [{}])[0].get("url") or
                            poll_data.get("image", {}).get("url"))
                break
            elif status in ["FAILED", "ERROR"]:
                return None, f"Freepik task failed with status: {status}"
            
            time.sleep(2)
        
        if not image_url:
            return None, "Timeout waiting for Freepik image"
        
        # Download image
        img_resp = requests.get(image_url, timeout=30)
        img_resp.raise_for_status()
        return img_resp.content, None
        
    except requests.RequestException as e:
        return None, f"Request error: {e}"
    except Exception as e:
        return None, f"Error: {e}"


def probar_modelo_freepik(modelo: str, prompt: str, narrativa_key: str) -> dict:
    """Prueba un modelo de Freepik."""
    print(f"  → Freepik/{modelo}...", end=" ", flush=True)
    
    try:
        image_data, error = send_freepik_request(prompt, model_name=modelo)
        
        if error:
            error_short = error[:55] if len(error) > 55 else error
            print(f"❌ {error_short}")
            return {"success": False, "error": error}
        
        if image_data:
            filename = guardar_imagen(image_data, modelo, narrativa_key, "freepik")
            size_kb = len(image_data) / 1024
            print(f"✅ ({size_kb:.1f} KB)")
            return {"success": True, "filename": filename, "size_kb": size_kb}
        else:
            print("❌ No image data")
            return {"success": False, "error": "No image data"}
            
    except Exception as e:
        error_short = str(e)[:55] if len(str(e)) > 55 else str(e)
        print(f"❌ {error_short}")
        return {"success": False, "error": str(e)}


def probar_modelo_gemini(modelo: str, prompt: str, narrativa_key: str) -> dict:
    """Prueba un modelo de Gemini."""
    print(f"  → Gemini/{modelo}...", end=" ", flush=True)
    
    try:
        if not GEMINI_API_KEY:
            return {"success": False, "error": "GEMINI_API_KEY not set"}
        
        if modelo not in GEMINI_MODELS:
            return {"success": False, "error": f"Model {modelo} not found"}
        
        config = GEMINI_MODELS[modelo]
        model_id = config["model_id"]
        api_type = config.get("api_type", "generateContent")
        
        if api_type == "predict":
            success, image_data, error = _call_imagen_predict(GEMINI_API_KEY, model_id, prompt, modelo)
        else:
            success, image_data, error = _call_gemini_generate_content(GEMINI_API_KEY, model_id, prompt, modelo)
        
        if not success:
            error_short = error[:55] if error and len(error) > 55 else (error or "Unknown error")
            print(f"❌ {error_short}")
            return {"success": False, "error": error}
        
        if image_data:
            filename = guardar_imagen(image_data, modelo, narrativa_key, "gemini")
            size_kb = len(image_data) / 1024
            print(f"✅ ({size_kb:.1f} KB)")
            return {"success": True, "filename": filename, "size_kb": size_kb}
        else:
            print("❌ No image data")
            return {"success": False, "error": "No image data"}
            
    except Exception as e:
        error_short = str(e)[:55] if len(str(e)) > 55 else str(e)
        print(f"❌ {error_short}")
        return {"success": False, "error": str(e)}


def main():
    print("=" * 70)
    print("PRUEBA DE TODOS LOS MODELOS DE GENERACIÓN DE IMÁGENES")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Modelos Freepik: {len(FREEPIK_MODELS)}")
    print(f"Modelos Gemini: {len(GEMINI_MODELS)}")
    print()
    
    resultados = []
    
    for narrativa_key, narrativa_data in NARRATIVAS.items():
        titulo = narrativa_data["titulo"]
        prompt = narrativa_data["prompt"]
        
        print(f"\n{'='*70}")
        print(f"NARRATIVA: {titulo}")
        print(f"{'='*70}")
        print(f"Prompt: {prompt[:70]}...")
        print()
        
        # Probar modelos Freepik
        print(f"--- FREEPIK ({len(FREEPIK_MODELS)} modelos) ---")
        for modelo in FREEPIK_MODELS.keys():
            resultado = probar_modelo_freepik(modelo, prompt, narrativa_key)
            resultados.append({
                "narrativa": narrativa_key,
                "proveedor": "freepik",
                "modelo": modelo,
                **resultado
            })
            time.sleep(3)  # Rate limiting
        
        # Probar modelos Gemini
        print(f"\n--- GEMINI ({len(GEMINI_MODELS)} modelos) ---")
        for modelo in GEMINI_MODELS.keys():
            resultado = probar_modelo_gemini(modelo, prompt, narrativa_key)
            resultados.append({
                "narrativa": narrativa_key,
                "proveedor": "gemini",
                "modelo": modelo,
                **resultado
            })
            time.sleep(3)  # Rate limiting
    
    # Resumen final
    print(f"\n{'='*70}")
    print("RESUMEN")
    print(f"{'='*70}")
    
    exitosos = [r for r in resultados if r["success"]]
    fallidos = [r for r in resultados if not r["success"]]
    
    print(f"\nTotal: {len(resultados)} | Exitosos: {len(exitosos)} ✅ | Fallidos: {len(fallidos)} ❌")
    
    if exitosos:
        print(f"\n--- ÉXITOS ---")
        for r in exitosos:
            print(f"  ✅ {r['proveedor']}/{r['modelo']} ({r['narrativa']}) - {r.get('size_kb', 0):.1f} KB")
    
    if fallidos:
        print(f"\n--- FALLOS ---")
        for r in fallidos:
            error = r.get('error', 'Unknown')
            error_short = error[:50] if len(error) > 50 else error
            print(f"  ❌ {r['proveedor']}/{r['modelo']} ({r['narrativa']}): {error_short}")
    
    print(f"\n{'='*70}")
    print("Completado")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
