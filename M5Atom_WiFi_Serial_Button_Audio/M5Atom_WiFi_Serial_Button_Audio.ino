#include <M5Atom.h>
#include <WiFi.h>
#include <driver/i2s.h>

// ======================
// Wi-Fi AP config
// ======================
const char* AP_SSID = "AtomEchoAP";
const char* AP_PASSWORD = "12345678";   // change if you want

WiFiServer server(5005);
WiFiClient client;

// State for LED logic
bool wifiReady       = false;
bool clientConnected = false;

// ======================
// Atom Echo I2S config
// ======================
#define CONFIG_I2S_BCK_PIN      19   // SCK
#define CONFIG_I2S_LRCK_PIN     33   // WS/LRCLK
#define CONFIG_I2S_DATA_PIN     22   // unused for mic
#define CONFIG_I2S_DATA_IN_PIN  23   // SD

#define SPEAKER_I2S_NUMBER      I2S_NUM_0
#define SAMPLE_RATE             16000
#define RAW_DATA_SIZE           8192   // bytes from i2s_read (larger chunk for more headroom)

static bool i2s_initialized = false;

// ======================
// I2S init for MIC only
// (your original MIC init)
// ======================
void InitI2SMic() {
  esp_err_t err;

  if (i2s_initialized) {
    i2s_driver_uninstall(SPEAKER_I2S_NUMBER);
    i2s_initialized = false;
  }

  // Standard I2S RX (Atom Echo mic is an I2S mic)
  i2s_config_t i2s_config = {
    .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate          = SAMPLE_RATE,
    .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format       = I2S_CHANNEL_FMT_RIGHT_LEFT, // capture both channels
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count        = 10,
    .dma_buf_len          = 1024,
    .use_apll             = true,
  };

  // MIC mode: master + RX + PDM
  i2s_config.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_PDM);

  err = i2s_driver_install(SPEAKER_I2S_NUMBER, &i2s_config, 0, NULL);
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
  pin_config.data_out_num = I2S_PIN_NO_CHANGE;        // not used in mic mode
  pin_config.data_in_num  = CONFIG_I2S_DATA_IN_PIN;   // mic input

  err = i2s_set_pin(SPEAKER_I2S_NUMBER, &pin_config);
  if (err != ESP_OK) {
    Serial.printf("i2s_set_pin failed: %d\n", err);
    return;
  }

  // Explicitly set clock to 16 kHz, stereo, 32-bit source (we'll downconvert to 16-bit mono)
  err = i2s_set_clk(SPEAKER_I2S_NUMBER, SAMPLE_RATE, I2S_BITS_PER_SAMPLE_32BIT, I2S_CHANNEL_STEREO);
  if (err != ESP_OK) {
    Serial.printf("i2s_set_clk failed: %d\n", err);
    return;
  }

  i2s_initialized = true;
}

// ======================
// LED state helper
// ======================
// offline (wifiReady == false)          -> RED
// buttonPressed == true                 -> ORANGE (overrides client state)
// wifiReady && clientConnected          -> GREEN
// wifiReady && !clientConnected         -> BLUE
void updateLed(bool wifiReady, bool clientConnected, bool buttonPressed) {
  if (!wifiReady) {
    // AP not started or failed
    M5.dis.drawpix(0, CRGB(255, 0, 0));        // red
  } else if (buttonPressed) {
    // Recording / button pressed
    M5.dis.drawpix(0, CRGB(255, 165, 0));      // orange
  } else if (clientConnected) {
    // Client connected, idle
    M5.dis.drawpix(0, CRGB(0, 255, 0));        // green
  } else {
    // AP up, no client
    M5.dis.drawpix(0, CRGB(0, 0, 255));        // blue
  }
}

// ======================
// Simple framed protocol:
// [type:1][len:2][payload:len]
// type = 'C' (control, e.g. "DOWN"/"UP")
//      = 'A' (audio, 16-bit mono PCM)
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
  sendPacket('C', (const uint8_t*)state, strlen(state)); // "DOWN" or "UP"
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
    Serial.println(WiFi.softAPIP());   // typically 192.168.4.1
  }
  server.begin();
  server.setNoDelay(true);
}

void acceptClientIfAny() {
  // If we already have a connected client, mark state and return
  if (client && client.connected()) {
    clientConnected = true;
    return;
  }

  // Otherwise, see if a new client appears
  WiFiClient newClient = server.available();
  if (newClient) {
    if (client) client.stop();
    client = newClient;
    clientConnected = true;
    Serial.print("Client connected: ");
    Serial.println(client.remoteIP());
  } else {
    clientConnected = false;
  }
}

void setup() {
  Serial.begin(115200);
  M5.begin(true, false, true);

  // Start with "offline" LED until AP is ready
  updateLed(false, false, false);

  setupWiFiAP();
  InitI2SMic();
}

void loop() {
  static bool lastPressed = false;
  static uint8_t rawBuffer[RAW_DATA_SIZE];
  static int16_t left16[RAW_DATA_SIZE / 8];   // max 256 samples @ RAW_DATA_SIZE=2048
  static int16_t right16[RAW_DATA_SIZE / 8];
  static unsigned long lastDebugMs = 0;

  acceptClientIfAny();
  M5.update();

  bool pressed = M5.Btn.isPressed();

  // Button edge detection
  if (pressed && !lastPressed) {
    Serial.println("BTN DOWN");
    sendButtonEvent("DOWN");
  } else if (!pressed && lastPressed) {
    Serial.println("BTN UP");
    sendButtonEvent("UP");
  }
  lastPressed = pressed;

  // While pressed, read mic; send to client if connected
  if (pressed && i2s_initialized) {
    size_t byte_read = 0;
    esp_err_t ret = i2s_read(SPEAKER_I2S_NUMBER,
                             (void*)rawBuffer,
                             RAW_DATA_SIZE,
                             &byte_read,
                             150 / portTICK_PERIOD_MS); // longer timeout to fill bigger buffer
    if (ret == ESP_OK && byte_read > 0) {
      // Align to complete 32-bit frames
      byte_read = (byte_read / 4) * 4;
      size_t word_count = byte_read / 4;          // 32-bit words
      size_t samples = word_count / 2;            // stereo frames (L,R)
      if (samples > (RAW_DATA_SIZE / 8)) {
        samples = RAW_DATA_SIZE / 8;
      }

      const int shift = 11; // start with upper bits, adjust if clipping
      int16_t minL = 32767, maxL = -32768;
      int16_t minR = 32767, maxR = -32768;
      int64_t accL = 0, accR = 0;
      int64_t sumL = 0, sumR = 0;
      int32_t* words = (int32_t*)rawBuffer;

      // First pass: shift and accumulate for DC offset
      for (size_t i = 0; i < samples; ++i) {
        int32_t sL = words[2 * i] >> shift;
        int32_t sR = words[2 * i + 1] >> shift;
        sumL += sL;
        sumR += sR;
      }
      int32_t meanL = samples ? (int32_t)(sumL / (int64_t)samples) : 0;
      int32_t meanR = samples ? (int32_t)(sumR / (int64_t)samples) : 0;

      auto clamp16 = [](int32_t v) -> int16_t {
        if (v > 32767) return 32767;
        if (v < -32768) return -32768;
        return (int16_t)v;
      };

      for (size_t i = 0; i < samples; ++i) {
        int32_t sL = words[2 * i] >> shift;
        int32_t sR = words[2 * i + 1] >> shift;
        int16_t vL = clamp16(sL - meanL);
        int16_t vR = clamp16(sR - meanR);
        left16[i] = vL;
        right16[i] = vR;
        if (vL < minL) minL = vL;
        if (vL > maxL) maxL = vL;
        if (vR < minR) minR = vR;
        if (vR > maxR) maxR = vR;
        accL += (int32_t)vL * (int32_t)vL;
        accR += (int32_t)vR * (int32_t)vR;
      }

      float rmsL = samples ? sqrtf((float)accL / samples) / 32768.0f : 0.0f;
      float rmsR = samples ? sqrtf((float)accR / samples) / 32768.0f : 0.0f;
      bool useLeft = rmsL > rmsR * 1.5f && rmsL > 0.0f;
      int16_t* chosen = useLeft ? left16 : right16;

      if (samples > 0 && client && client.connected()) {
        sendPacket('A', (uint8_t*)chosen, (uint16_t)(samples * 2)); // mono 16-bit
      }

      unsigned long now = millis();
      if (now - lastDebugMs > 1000) {
        lastDebugMs = now;
        Serial.printf("I2S32 bytes=%u frames=%u L[min=%d max=%d rms=%.4f] R[min=%d max=%d rms=%.4f] first4=%02X %02X %02X %02X\n",
                      (unsigned)byte_read, (unsigned)samples,
                      minL, maxL, rmsL, minR, maxR, rmsR,
                      rawBuffer[0], rawBuffer[1], rawBuffer[2], rawBuffer[3]);
      }
    } else {
      Serial.printf("i2s_read err=%d bytes=%u\n", ret, (unsigned)byte_read);
    }
  }

  // Update LED according to current state
  updateLed(wifiReady, clientConnected, pressed);

  delay(1);
}
