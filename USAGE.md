# 🚀 Guía de uso - Vocolore

## ✅ Pre-requisitos (configurar una vez)

1. **API Keys** en `config.env`:
   ```bash
   FREEPIK_API_KEY=tu_key_aqui
   GEMINI_API_KEY=tu_key_aqui
   ```

2. **Impresora** configurada en CUPS:
   ```bash
   lpstat -p  # Verificar que aparece la impresora
   ```

---

## 📋 Flujo de uso diario

| # | Paso | Qué ver |
|---|------|---------|
| 1 | Conectar **impresora** por USB | Led impresora encendido |
| 2 | Conectar **dongle WiFi** (para internet) | Segunda interfaz de red activa |
| 3 | Encender el **M5Atom Echo** | Led rojo → azul |
| 4 | Conectarse a red **"Vocolore"** | Contraseña: `12345678` |
| 5 | Abrir terminal | — |
| 6 | `cd ~/code/vocolore` | — |
| 7 | `uv run python run_wifi_and_ui.py` | Aparecen 2 ventanas (debug + usuario) |
| 8 | En UI debug: seleccionar **Provider: gemini** | Dropdown cambia a modelos Gemini |
| 9 | Ver **"Connected to 192.168.4.1:3333"** | Led Atom pasa a **verde** 🟢 |
| 10 | **¡Listo!** Presionar botón del Atom y hablar | — |

---

## 🎮 Cómo usar

1. **Mantener presionado** el botón del M5Atom (led naranja = grabando)
2. **Hablar** la narrativa del niño
3. **Soltar** el botón (led azul = procesando)
4. Esperar la imagen generada (aparece en pantalla y se imprime si está habilitado)

---

## 🔧 Si algo falla

| Problema | Solución |
|----------|----------|
| No conecta al Atom | Verificar que la laptop está en red "Vocolore" |
| No genera imágenes | Revisar API keys en `config.env` |
| No imprime | Verificar `lpstat -p` y que `PRINT_COMMAND=lp` |
| Atom led rojo | Reconectar WiFi o reiniciar Atom |
