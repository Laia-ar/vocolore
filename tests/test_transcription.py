#!/usr/bin/env python3
"""
Script de prueba para procesar narrativas de niños y generar prompts de imágenes.
Uso: python test_data/test_transcription.py
"""

import sys
sys.path.insert(0, '.')

from wifi_transcribe import extract_subject, enhance_prompt_for_coloring_book

# Narrativas de prueba
NARRATIVAS = {
    "parque_rivadavia": """
        Yo recuerdo que acá en el parque Rivadavia, donde están los Juegos de Madera, 
        jugamos a la pelota ahí en el costadito de los Juegos de Madera.
        Entonces mi papá tiró la pelota y otro día, ahí en los Juegos de Madera al costadito, 
        mi papá tiró la pelota y se atacó en el árbol. ¡Y nada más!
    """,
    
    "cumpleanos_pikachu": """
        Yo recuerdo que el 5 de mayo de 2023 fue mi cumpleaños y en la escuela fui primero 
        en la fila de inglés y fui primero en la fila de salir.
        Entonces después fui hasta el parque y hice mi cumpleaños.
        Pusieron banderines y fue de Pikachu y Isabela mi cumpleaños.
        Entonces todos mis amigos fueron ahí y estaban dos animadores, una chica y un chico, 
        que decían que Disney se había borrado y que necesitábamos una varita mágica 
        para ponerlo en color otra vez.
        Entonces también teníamos que encontrar una llave para esa caja de la varita.
        Entonces fuimos adivinando pistas y fuimos haciendo cosas, retos, adivinanzas y por fin lo logramos.
        Nos encontramos la varita mágica, soplamos las velitas y nada más.
    """
}


def test_subject_extraction():
    """Prueba la extracción de sujetos de las narrativas."""
    print("=" * 60)
    print("PRUEBA: Extracción de sujetos")
    print("=" * 60)
    
    for nombre, texto in NARRATIVAS.items():
        print(f"\n--- {nombre} ---")
        print(f"Texto: {texto[:100]}...")
        sujeto = extract_subject(texto)
        print(f"Sujeto extraído: {sujeto}")
    
    print("\n")


def test_prompt_enhancement():
    """Prueba la mejora de prompts para coloring book."""
    print("=" * 60)
    print("PRUEBA: Mejora de prompts")
    print("=" * 60)
    
    prompts_basicos = [
        "niño jugando a la pelota en el parque",
        "cumpleaños de Pikachu con banderines",
        "animadores con varita mágica",
        "niños buscando una llave",
    ]
    
    for prompt in prompts_basicos:
        print(f"\nOriginal: {prompt}")
        mejorado = enhance_prompt_for_coloring_book(prompt)
        print(f"Mejorado: {mejorado}")
    
    print("\n")


def test_full_pipeline():
    """Prueba el pipeline completo: narrativa -> prompt final."""
    print("=" * 60)
    print("PRUEBA: Pipeline completo")
    print("=" * 60)
    
    for nombre, texto in NARRATIVAS.items():
        print(f"\n--- {nombre} ---")
        print(f"Narrativa: {texto[:80]}...")
        
        # Extraer sujeto
        sujeto = extract_subject(texto)
        print(f"Sujeto: {sujeto}")
        
        # Mejorar prompt
        prompt_final = enhance_prompt_for_coloring_book(sujeto)
        print(f"Prompt final: {prompt_final[:100]}...")
    
    print("\n")


if __name__ == "__main__":
    test_subject_extraction()
    test_prompt_enhancement()
    test_full_pipeline()
    print("✅ Pruebas completadas")
