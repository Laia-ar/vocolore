#include <M5Atom.h>
#include <WiFi.h>
#include <driver/i2s.h>
#include "sounds.h"

// ======================
// Wi-Fi AP config
// ======================
const char* AP_SSID = "AtomEchoAP";
const char* AP_PASSWORD = "12345678";

WiFiServer server(5005);
WiFiClient client;

// State for LED logic
bool wifiReady       = false;
bool clientConnected = false;

// ======================
// I2S config
// ======================
#define CONFIG_I2S_BCK_PIN      19   // SCK
#define CONFIG_I2S_LRCK_PIN     33   // WS/LRCLK
#define CONFIG_I2S_DATA_PIN     22   // DATA OUT (speaker)
#define CONFIG_I2S_DATA_IN_PIN  23   // SD (microphone)

#define I2S_NUM                 I2S_NUM_0
#define SAMPLE_RATE             16000
#define RAW_DATA_SIZE           8192

// Sound playback state
static volatile bool soundPlaying = false;
static volatile size_t soundPos = 0;
static volatile const int16_t* soundData = nullptr;
static volatile size_t soundLen = 0;
static volatile uint32_t soundSampleRate = 8000;

static bool i2s_initialized = false;

// ======================
// Sound playback
// ======================
void playSound(uint8_t soundId) {
  if (soundId >= NUM_SOUNDS) return;
  if (soundPlaying) return;  // Don't interrupt current sound
  
  soundData = SOUNDS[soundId].data;
  soundLen = SOUNDS[soundId].len;
  soundSampleRate = SOUNDS[soundId].sample_rate;
  soundPos = 0;
  soundPlaying = true;
  
  Serial.printf("Playing sound %d, len=%d samples\n", soundId, (int)soundLen);
}

// Process sound playback (call from loop)
void processSoundPlayback() {
  if (!soundPlaying || soundData == nullptr) return;
  
  // Calculate how many samples to output based on sample rate ratio
  // We output at 16kHz, but sounds may be at 8kHz
  static uint32_t acc = 0;
  const uint32_t step = (soundSampleRate << 16) / SAMPLE_RATE;  // Fixed point ratio
  
  // Prepare output buffer (stereo interleaved)
  int16_t buffer[64];  // 32 stereo samples
  size_t samplesToWrite = 0;
  
  for (int i = 0; i < 32; i++) {
    if (soundPos >= soundLen) {
      soundPlaying = false;
      soundData = nullptr;
      break;
    }
    
    int32_t sample = soundData[soundPos];
    // Scale up a bit for better volume
    sample = (sample * 3) / 2;
    // Clamp
    if (sample > 32767) sample = 32767;
    if (sample < -32768) sample = -32768;
    
    buffer[i * 2] = (int16_t)sample;      // Left
    buffer[i * 2 + 1] = (int16_t)sample;  // Right
    samplesToWrite++;
    
    // Advance position based on sample rate ratio
    acc += step;
    while (acc >= (1 << 16)) {
      soundPos++;
      acc -= (1 << 16);
    }
  }
  
  if (samplesToWrite > 0) {
    size_t bytesWritten = 0;
    i2s_write(I2S_NUM, buffer, samplesToWrite * 4, &bytesWritten, 0);
  }
}

// ======================
// I2S init for MIC (RX mode)
// ======================
void InitI2SMic() {
  esp_err_t err;

  if (i2s_initialized) {
    i2s_driver_uninstall(I2S_NUM);
    i2s_initialized = false;
  }

  i2s_config_t i2s_config = {
    .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate          = SAMPLE_RATE,
    .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format       = I2S_CHANNEL_FMT_RIGHT_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count        = 10,
    .dma_buf_len          = 1024,
    .use_apll             = true,
  };

  i2s_config.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_PDM);

  err = i2s_driver_install(I2S_NUM, &i2s_config, 0, NULL);
  if (err != ESP_OK) {
    Serial.printf("i2s_driver_install failed: %d\n", err);
    return;
  }

  i2s_pin_config_t pin_config;
#if (ESP_IDF_VERSION > ESP_IDF_VERSION_VAL(4, 3, 0))
  pin_config.mck_io_num = I2S_PIN_NO_CHANGE;
#endif
  pin_config.bck_io_num   = CONFIG_I2S_BCK_PIN;
  pin_config.ws_io_num    = CONFIG_I2S_LRCK_PIN;
  pin_config.data_out_num = I2S_PIN_NO_CHANGE;
  pin_config.data_in_num  = CONFIG_I2S_DATA_IN_PIN;

  err = i2s_set_pin(I2S_NUM, &pin_config);
  if (err != ESP_OK) {
    Serial.printf("i2s_set_pin failed: %d\n", err);
    return;
  }

  err = i2s_set_clk(I2S_NUM, SAMPLE_RATE, I2S_BITS_PER_SAMPLE_32BIT, I2S_CHANNEL_STEREO);
  if (err != ESP_OK) {
    Serial.printf("i2s_set_clk failed: %d\n", err);
    return;
  }

  i2s_initialized = true;
}

// ======================
// I2S init for Speaker (TX mode)
// ======================
void InitI2SSpeaker() {
  esp_err_t err;

  if (i2s_initialized) {
    i2s_driver_uninstall(I2S_NUM);
  }

  i2s_config_t i2s_config = {
    .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
    .sample_rate          = SAMPLE_RATE,
    .bits_per_sample      = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format       = I2S_CHANNEL_FMT_RIGHT_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count        = 4,
    .dma_buf_len          = 256,
    .use_apll             = false,
  };

  err = i2s_driver_install(I2S_NUM, &i2s_config, 0, NULL);
  if (err != ESP_OK) {
    Serial.printf("i2s_driver_install (speaker) failed: %d\n", err);
    return;
  }

  i2s_pin_config_t pin_config;
#if (ESP_IDF_VERSION > ESP_IDF_VERSION_VAL(4, 3, 0))
  pin_config.mck_io_num = I2S_PIN_NO_CHANGE;
#endif
  pin_config.bck_io_num   = CONFIG_I2S_BCK_PIN;
  pin_config.ws_io_num    = CONFIG_I2S_LRCK_PIN;
  pin_config.data_out_num = CONFIG_I2S_DATA_PIN;
  pin_config.data_in_num  = I2S_PIN_NO_CHANGE;

  err = i2s_set_pin(I2S_NUM, &pin_config);
  if (err != ESP_OK) {
    Serial.printf("i2s_set_pin (speaker) failed: %d\n", err);
    return;
  }

  i2s_initialized = true;
}

// ======================
// LED state helper
// ======================
void updateLed(bool wifiReady, bool clientConnected, bool buttonPressed) {
  if (!wifiReady) {
    M5.dis.drawpix(0, CRGB(255, 0, 0));        // red
  } else if (buttonPressed) {
    M5.dis.drawpix(0, CRGB(255, 165, 0));      // orange
  } else if (clientConnected) {
    M5.dis.drawpix(0, CRGB(0, 255, 0));        // green
  } else {
    M5.dis.drawpix(0, CRGB(0, 0, 255));        // blue
  }
}

// ======================
// Battery monitoring
// ======================
#define IP5306_ADDR 0x75

int getBatteryLevel() {
  #ifdef M5STACK_POWER
    return M5.Power.getBatteryLevel();
  #else
    Wire.beginTransmission(IP5306_ADDR);
    Wire.write(0x78);
    if (Wire.endTransmission(false) != 0) return -1;
    
    Wire.requestFrom(IP5306_ADDR, 1);
    if (Wire.available()) {
      uint8_t data = Wire.read();
      switch (data & 0x0F) {
        case 0: return 100;
        case 1: return 75;
        case 2: return 50;
        case 3: return 25;
        default: return 0;
      }
    }
    return -1;
  #endif
}

// ======================
// Framed protocol
// [type:1][len:2][payload:len]
// type = 'C' control (DOWN/UP)
//      = 'A' audio (16-bit mono PCM)
//      = 'B' battery level
//      = 'S' sound command (from PC)
// ======================
void sendPacket(char type, const uint8_t* data, uint16_t len) {
  if (!client || !client.connected()) return;

  uint8_t header[3];
  header[0] = (uint8_t)type;
  header[1] = (uint8_t)(len >> 8);
  header[2] = (uint8_t)(len & 0xFF);

  client.write(header, 3);
  if (len > 0) {
    client.write(data, len);
  }
}

void sendButtonEvent(const char* state) {
  sendPacket('C', (const uint8_t*)state, strlen(state));
}

void sendBatteryLevel() {
  static unsigned long lastBatteryMs = 0;
  unsigned long now = millis();
  
  if (now - lastBatteryMs < 30000) return;
  lastBatteryMs = now;
  
  int level = getBatteryLevel();
  if (level >= 0) {
    char buf[8];
    snprintf(buf, sizeof(buf), "%d", level);
    sendPacket('B', (const uint8_t*)buf, strlen(buf));
    Serial.printf("Battery: %d%%\n", level);
  }
}

// ======================
// Wi-Fi AP setup
// ======================
void setupWiFiAP() {
  WiFi.mode(WIFI_AP);
  bool ok = WiFi.softAP(AP_SSID, AP_PASSWORD);
  if (!ok) {
    Serial.println("Failed to start softAP");
    wifiReady = false;
  } else {
    wifiReady = true;
    Serial.print("AP started: ");
    Serial.println(AP_SSID);
    Serial.print("AP IP: ");
    Serial.println(WiFi.softAPIP());
  }
  server.begin();
  server.setNoDelay(true);
}

void acceptClientIfAny() {
  if (client && client.connected()) {
    clientConnected = true;
    return;
  }

  WiFiClient newClient = server.available();
  if (newClient) {
    if (client) client.stop();
    client = newClient;
    clientConnected = true;
    Serial.print("Client connected: ");
    Serial.println(client.remoteIP());
    playSound(SOUND_PICKUP_COIN);  // Play sound on connection
  } else {
    clientConnected = false;
  }
}

// ======================
// Sound command handling from PC
// ======================
void handleSoundCommand(const uint8_t* data, uint16_t len) {
  if (len < 1) return;
  uint8_t soundId = data[0];
  Serial.printf("Received sound command: %d\n", soundId);
  playSound(soundId);
}

void setup() {
  Serial.begin(115200);
  M5.begin(true, false, true);

  updateLed(false, false, false);

  setupWiFiAP();
  InitI2SSpeaker();  // Start in speaker mode (default)
}

void loop() {
  static bool lastPressed = false;
  static bool wasRecording = false;
  static unsigned long lastDebugMs = 0;

  acceptClientIfAny();
  M5.update();

  bool pressed = M5.Btn.isPressed();

  // Button edge detection
  if (pressed && !lastPressed) {
    Serial.println("BTN DOWN");
    sendButtonEvent("DOWN");
    playSound(SOUND_PICKUP_COIN);  // Sound on start recording
    InitI2SMic();  // Switch to mic mode
    wasRecording = true;
  } else if (!pressed && lastPressed) {
    Serial.println("BTN UP");
    sendButtonEvent("UP");
    // Recording finished - handled below
  }
  lastPressed = pressed;

  // Process sound playback when not recording
  if (!pressed && !wasRecording) {
    processSoundPlayback();
  }

  // Handle recording mode
  if (pressed) {
    // Read mic data and send to client
    size_t byte_read = 0;
    uint8_t rawBuffer[RAW_DATA_SIZE];
    int16_t left16[RAW_DATA_SIZE / 8];
    
    esp_err_t ret = i2s_read(I2S_NUM, (void*)rawBuffer, RAW_DATA_SIZE, &byte_read, 50 / portTICK_PERIOD_MS);
    if (ret == ESP_OK && byte_read > 0) {
      byte_read = (byte_read / 4) * 4;
      size_t word_count = byte_read / 4;
      size_t samples = word_count / 2;
      if (samples > (RAW_DATA_SIZE / 8)) samples = RAW_DATA_SIZE / 8;

      const int shift = 11;
      int64_t sumL = 0, sumR = 0;
      int32_t* words = (int32_t*)rawBuffer;

      for (size_t i = 0; i < samples; ++i) {
        sumL += words[2 * i] >> shift;
        sumR += words[2 * i + 1] >> shift;
      }
      int32_t meanL = samples ? (int32_t)(sumL / (int64_t)samples) : 0;
      int32_t meanR = samples ? (int32_t)(sumR / (int64_t)samples) : 0;

      auto clamp16 = [](int32_t v) -> int16_t {
        if (v > 32767) return 32767;
        if (v < -32768) return -32768;
        return (int16_t)v;
      };

      int64_t accL = 0, accR = 0;
      for (size_t i = 0; i < samples; ++i) {
        int32_t sL = words[2 * i] >> shift;
        int32_t sR = words[2 * i + 1] >> shift;
        int16_t vL = clamp16(sL - meanL);
        int16_t vR = clamp16(sR - meanR);
        left16[i] = vL;
        accL += (int32_t)vL * (int32_t)vL;
        accR += (int32_t)vR * (int32_t)vR;
      }

      float rmsL = samples ? sqrtf((float)accL / samples) / 32768.0f : 0.0f;
      float rmsR = samples ? sqrtf((float)accR / samples) / 32768.0f : 0.0f;
      bool useLeft = rmsL > rmsR * 1.5f && rmsL > 0.0f;
      int16_t* chosen = useLeft ? left16 : left16;

      if (samples > 0 && client && client.connected()) {
        sendPacket('A', (uint8_t*)chosen, (uint16_t)(samples * 2));
      }
    }
  } else if (wasRecording) {
    // Just stopped recording
    wasRecording = false;
    playSound(SOUND_PICKUP_COIN_REV);  // Sound on stop recording
    // Small delay for sound to start before switching back
    delay(100);
    InitI2SSpeaker();  // Switch back to speaker mode
  }

  // Check for incoming commands from PC
  if (client && client.connected() && client.available() >= 3) {
    uint8_t header[3];
    if (client.read(header, 3) == 3) {
      char type = header[0];
      uint16_t len = (header[1] << 8) | header[2];
      if (len > 0 && len < 256) {
        uint8_t payload[256];
        size_t read = 0;
        while (read < len && client.available()) {
          payload[read++] = client.read();
        }
        if (type == 'S') {
          handleSoundCommand(payload, read);
        }
      }
    }
  }

  updateLed(wifiReady, clientConnected, pressed);
  sendBatteryLevel();

  delay(1);
}
