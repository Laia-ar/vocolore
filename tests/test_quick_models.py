#!/usr/bin/env python3
"""
Prueba rápida de algunos modelos de generación de imágenes.
Usa prompts cortos y timeouts reducidos.

Uso: cd /home/didi/code/vocolore && uv run python test_data/test_quick_models.py
"""

import os
import sys
import time
import requests
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wifi_transcribe import (
    GEMINI_MODELS,
    _call_gemini_generate_content,
    _call_imagen_predict,
    GEMINI_API_KEY,
)
from dotenv import load_dotenv

load_dotenv()

# Prompts de prueba - escenas simples de las narrativas
PROMPTS = {
    "parque_rivadavia": "children playing ball near wooden playground in park, ball stuck in tree",
    "cumpleanos_pikachu": "children birthday party with yellow electric mouse decorations, magic wand, birthday cake",
}


def test_gemini_flash(prompt: str, key: str) -> dict:
    """Prueba Gemini 2.5 Flash (más rápido)."""
    print(f"  → Gemini 2.5 Flash ({key})...", end=" ", flush=True)
    
    try:
        model_name = "gemini-2.5-flash-image"
        model_id = GEMINI_MODELS[model_name]["model_id"]
        
        success, image_data, error = _call_gemini_generate_content(
            GEMINI_API_KEY, model_id, prompt, model_name
        )
        
        if not success:
            print(f"❌ {error[:50] if error else 'Unknown'}")
            return {"success": False, "error": error}
        
        filename = f"test_data/img_{key}_gemini_flash_{int(time.time())}.png"
        with open(filename, "wb") as f:
            f.write(image_data)
        
        size_kb = len(image_data) / 1024
        print(f"✅ ({size_kb:.1f} KB) → {filename}")
        return {"success": True, "filename": filename, "size_kb": size_kb}
        
    except Exception as e:
        print(f"❌ {str(e)[:50]}")
        return {"success": False, "error": str(e)}


def test_imagen4(prompt: str, key: str) -> dict:
    """Prueba Imagen 4."""
    print(f"  → Imagen 4 ({key})...", end=" ", flush=True)
    
    try:
        model_name = "imagen-4"
        model_id = GEMINI_MODELS[model_name]["model_id"]
        
        success, image_data, error = _call_imagen_predict(
            GEMINI_API_KEY, model_id, prompt, model_name
        )
        
        if not success:
            print(f"❌ {error[:50] if error else 'Unknown'}")
            return {"success": False, "error": error}
        
        filename = f"test_data/img_{key}_imagen4_{int(time.time())}.png"
        with open(filename, "wb") as f:
            f.write(image_data)
        
        size_kb = len(image_data) / 1024
        print(f"✅ ({size_kb:.1f} KB) → {filename}")
        return {"success": True, "filename": filename, "size_kb": size_kb}
        
    except Exception as e:
        print(f"❌ {str(e)[:50]}")
        return {"success": False, "error": str(e)}


def main():
    print("=" * 60)
    print("PRUEBA RÁPIDA DE MODELOS")
    print("=" * 60)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    resultados = []
    
    for key, prompt in PROMPTS.items():
        print(f"\n--- Prompt: {key} ---")
        print(f"    '{prompt}'")
        print()
        
        # Probar Gemini Flash
        r1 = test_gemini_flash(prompt, key)
        resultados.append({"modelo": "gemini-2.5-flash", "prompt": key, **r1})
        time.sleep(2)
        
        # Probar Imagen 4
        r2 = test_imagen4(prompt, key)
        resultados.append({"modelo": "imagen-4", "prompt": key, **r2})
        time.sleep(2)
    
    # Resumen
    print(f"\n{'='*60}")
    print("RESUMEN")
    print(f"{'='*60}")
    
    exitosos = [r for r in resultados if r["success"]]
    fallidos = [r for r in resultados if not r["success"]]
    
    print(f"Total: {len(resultados)} | Exitosos: {len(exitosos)} ✅ | Fallidos: {len(fallidos)} ❌")
    
    if exitosos:
        print(f"\nImágenes generadas:")
        for r in exitosos:
            print(f"  ✅ {r['modelo']} ({r['prompt']}): {r['filename']} ({r['size_kb']:.1f} KB)")
    
    if fallidos:
        print(f"\nErrores:")
        for r in fallidos:
            error = r.get('error', 'Unknown')[:50]
            print(f"  ❌ {r['modelo']} ({r['prompt']}): {error}")


if __name__ == "__main__":
    main()
