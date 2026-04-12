#!/usr/bin/env python3
"""
Prueba de Gemini 3.1-flash y 3-pro con las narrativas.

Uso: cd /home/didi/code/vocolore && uv run python test_data/test_gemini_new.py
"""

import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wifi_transcribe import (
    GEMINI_MODELS,
    _call_gemini_generate_content,
    GEMINI_API_KEY,
)
from dotenv import load_dotenv

load_dotenv()

# Prompts de prueba
PROMPTS = {
    "parque_rivadavia": "children playing ball near wooden playground in park, ball stuck in tree",
    "cumpleanos_pikachu": "children birthday party with yellow electric mouse decorations, magic wand, birthday cake",
}

MODELS_TO_TEST = [
    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview",
]


def test_model(model_name: str, prompt: str, key: str) -> dict:
    """Prueba un modelo Gemini."""
    print(f"  → {model_name} ({key})...", end=" ", flush=True)
    
    try:
        if model_name not in GEMINI_MODELS:
            print(f"❌ Modelo no encontrado")
            return {"success": False, "error": "Model not found"}
        
        model_id = GEMINI_MODELS[model_name]["model_id"]
        
        success, image_data, error = _call_gemini_generate_content(
            GEMINI_API_KEY, model_id, prompt, model_name
        )
        
        if not success:
            error_short = error[:55] if error and len(error) > 55 else error
            print(f"❌ {error_short}")
            return {"success": False, "error": error}
        
        filename = f"test_data/img_{key}_{model_name.replace('-', '_')}_{int(time.time())}.png"
        with open(filename, "wb") as f:
            f.write(image_data)
        
        size_kb = len(image_data) / 1024
        print(f"✅ ({size_kb:.1f} KB)")
        return {"success": True, "filename": filename, "size_kb": size_kb}
        
    except Exception as e:
        error_short = str(e)[:55] if len(str(e)) > 55 else str(e)
        print(f"❌ {error_short}")
        return {"success": False, "error": str(e)}


def main():
    print("=" * 60)
    print("PRUEBA: Gemini 3.1-flash y 3-pro")
    print("=" * 60)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    resultados = []
    
    for key, prompt in PROMPTS.items():
        print(f"\n--- Prompt: {key} ---")
        print(f"    '{prompt}'")
        print()
        
        for model_name in MODELS_TO_TEST:
            r = test_model(model_name, prompt, key)
            resultados.append({"modelo": model_name, "prompt": key, **r})
            time.sleep(2)
    
    # Resumen
    print(f"\n{'='*60}")
    print("RESUMEN")
    print(f"{'='*60}")
    
    exitosos = [r for r in resultados if r["success"]]
    fallidos = [r for r in resultados if not r["success"]]
    
    print(f"\nTotal: {len(resultados)} | Exitosos: {len(exitosos)} ✅ | Fallidos: {len(fallidos)} ❌")
    
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
