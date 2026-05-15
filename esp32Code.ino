#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <ArduinoJson.h>
#include "driver/i2s.h"

#include <Preferences.h>
#include <WebServer.h>
#include <DNSServer.h>

static const char* SERVER_HOST = ""; //put in users host i removed mine for privacy 
static const uint16_t SERVER_PORT = 443;

static const char* SETUP_AP_SSID = "Glasses-Setup";
static const char* SETUP_AP_PASS = "setup1234";

static const int STATUS_LED_PIN = 14;

static const int SETUP_BTN_PIN = -1;
static const bool SETUP_BTN_ACTIVE_LOW = true;

// I2S MIC
#define MIC_BCLK  4
#define MIC_WS    5
#define MIC_SD    6

#define SPK_BCLK  19
#define SPK_WS    18
#define SPK_DOUT  20
#define SPK_EN_PIN (-1)  // wire EN to a GPIO to fully mute idle hiss

static const int MIC_SAMPLE_RATE = 16000;
static const int CHUNK_SAMPLES   = 640;  // ~40ms @ 16k

static int32_t i2s_raw[CHUNK_SAMPLES];
static int16_t pcm16[CHUNK_SAMPLES];

static const i2s_port_t I2S_PORT = I2S_NUM_0;

WiFiClientSecure client;
static volatile bool receivingAudio = false;

uint32_t finalCount = 0;

enum I2SModeState { MODE_NONE, MODE_MIC_RX, MODE_SPK_TX };
static I2SModeState i2sMode = MODE_NONE;

Preferences prefs;
String savedSsid;
String savedPass;

WebServer web(80);
DNSServer dns;

enum LedMode {
  LED_BOOT,
  LED_WIFI_CONNECTING,
  LED_PORTAL,
  LED_TCP_CONNECTING,
  LED_STREAMING,
  LED_ERROR
};

static LedMode ledMode = LED_BOOT;
static uint32_t ledLastMs = 0;
static bool ledOn = false;
static uint8_t portalStep = 0;

static void setLed(bool on) {
  ledOn = on;
  digitalWrite(STATUS_LED_PIN, on ? HIGH : LOW);
}

static void setLedMode(LedMode m) {
  ledMode = m;
  ledLastMs = 0;
  portalStep = 0;
  if (m == LED_STREAMING) setLed(true);
  else setLed(false);
}

static void updateLed() {
  uint32_t now = millis();

  switch (ledMode) {
    case LED_BOOT:
      if (now - ledLastMs >= 100) { ledLastMs = now; setLed(!ledOn); }
      break;

    case LED_WIFI_CONNECTING:
      if (now - ledLastMs >= 250) { ledLastMs = now; setLed(!ledOn); }
      break;

    case LED_TCP_CONNECTING:
      if (!ledOn) {
        if (now - ledLastMs >= 240) { ledLastMs = now; setLed(true); }
      } else {
        if (now - ledLastMs >= 60) { ledLastMs = now; setLed(false); }
      }
      break;

    case LED_PORTAL: {
      if (ledLastMs == 0) { ledLastMs = now; portalStep = 0; setLed(false); }
      uint32_t dt = now - ledLastMs;
      if (portalStep == 0 && dt >= 0)         { setLed(true);  portalStep = 1; ledLastMs = now; }
      else if (portalStep == 1 && dt >= 80)   { setLed(false); portalStep = 2; ledLastMs = now; }
      else if (portalStep == 2 && dt >= 120)  { setLed(true);  portalStep = 3; ledLastMs = now; }
      else if (portalStep == 3 && dt >= 80)   { setLed(false); portalStep = 4; ledLastMs = now; }
      else if (portalStep == 4 && dt >= 1220) { portalStep = 0; }
    } break;

    case LED_STREAMING:
      // solid on
      break;

    case LED_ERROR:
      if (ledLastMs == 0) { ledLastMs = now; portalStep = 0; setLed(false); }
      if (now - ledLastMs >= 120) {
        ledLastMs = now;
        setLed(!ledOn);
      }
      break;
  }
}

static void loadWifiFromNVS() {
  prefs.begin("glasses", true);
  savedSsid = prefs.getString("ssid", "");
  savedPass = prefs.getString("pass", "");
  prefs.end();
}

static void saveWifiToNVS(const String& ssid, const String& pass) {
  prefs.begin("glasses", false);
  prefs.putString("ssid", ssid);
  prefs.putString("pass", pass);
  prefs.end();
}

static void clearWifiNVS() {
  prefs.begin("glasses", false);
  prefs.remove("ssid");
  prefs.remove("pass");
  prefs.end();
}


static String portalHtml() {
  String h;
  h += "<!doctype html><html><head>";
  h += "<meta name='viewport' content='width=device-width,initial-scale=1'>";
  h += "<title>Glasses Setup</title></head><body style='font-family:system-ui;padding:16px;'>";
  h += "<h2>Glasses Wi-Fi Setup</h2>";
  h += "<form method='POST' action='/save'>";
  h += "<label>Wi-Fi SSID</label><br><input name='ssid' style='width:100%;padding:10px' required><br><br>";
  h += "<label>Wi-Fi Password</label><br><input name='pass' type='password' style='width:100%;padding:10px'><br><br>";
  h += "<button style='padding:12px 16px;font-size:16px;'>Save & Reboot</button>";
  h += "</form><hr style='margin:18px 0;'>";
  h += "<form method='POST' action='/clear' onsubmit='return confirm(\"Clear saved Wi-Fi and reboot?\")'>";
  h += "<button style='padding:10px 12px;'>Clear Saved Wi-Fi</button></form>";
  h += "</body></html>";
  return h;
}

static void startCaptivePortal() {
  setLedMode(LED_PORTAL);

  WiFi.mode(WIFI_AP);
  if (strlen(SETUP_AP_PASS) == 0) WiFi.softAP(SETUP_AP_SSID);
  else WiFi.softAP(SETUP_AP_SSID, SETUP_AP_PASS);

  delay(100);
  IPAddress apIP = WiFi.softAPIP();
  dns.start(53, "*", apIP);

  web.on("/", HTTP_GET, []() { web.send(200, "text/html", portalHtml()); });

  web.on("/save", HTTP_POST, []() {
    String ssid = web.arg("ssid");
    String pass = web.arg("pass");
    ssid.trim();
    if (ssid.length() < 1) { web.send(400, "text/plain", "Missing SSID"); return; }
    saveWifiToNVS(ssid, pass);
    web.send(200, "text/html", "<html><body>Saved. Rebooting…</body></html>");
    delay(300);
    ESP.restart();
  });

  web.on("/clear", HTTP_POST, []() {
    clearWifiNVS();
    web.send(200, "text/html", "<html><body>Cleared. Rebooting…</body></html>");
    delay(300);
    ESP.restart();
  });

  web.onNotFound([&]() {
    web.sendHeader("Location", String("http://") + WiFi.softAPIP().toString() + "/", true);
    web.send(302, "text/plain", "");
  });

  web.begin();

  Serial.println("=== CAPTIVE PORTAL MODE ===");
  Serial.print("AP SSID: "); Serial.println(SETUP_AP_SSID);
  Serial.print("AP IP:   "); Serial.println(apIP);
  Serial.println("===========================");

  while (true) {
    dns.processNextRequest();
    web.handleClient();
    updateLed();
    delay(2);
  }
}

static bool tryConnectSavedWiFi(uint32_t timeoutMs = 15000) {
  if (savedSsid.length() < 1) return false;

  setLedMode(LED_WIFI_CONNECTING);
  WiFi.mode(WIFI_STA);
  WiFi.begin(savedSsid.c_str(), savedPass.c_str());

  uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - start) < timeoutMs) {
    updateLed();
    delay(200);
  }
  return WiFi.status() == WL_CONNECTED;
}

static bool isSetupButtonHeldAtBoot() {
  if (SETUP_BTN_PIN < 0) return false;
  pinMode(SETUP_BTN_PIN, SETUP_BTN_ACTIVE_LOW ? INPUT_PULLUP : INPUT_PULLDOWN);

  uint32_t t0 = millis();
  bool held = true;
  while (millis() - t0 < 250) {
    bool v = digitalRead(SETUP_BTN_PIN);
    bool pressed = SETUP_BTN_ACTIVE_LOW ? (v == LOW) : (v == HIGH);
    held = held && pressed;
    delay(10);
  }
  return held;
}

static void connectWiFiOrPortal() {
  loadWifiFromNVS();

  bool forcePortal = isSetupButtonHeldAtBoot();
  if (!forcePortal) {
    Serial.printf("Trying saved Wi-Fi SSID: '%s'\n", savedSsid.c_str());
    if (tryConnectSavedWiFi(15000)) {
      Serial.print("WiFi OK, IP: ");
      Serial.println(WiFi.localIP());
      return;
    }
    Serial.println("WiFi connect failed.");
  } else {
    Serial.println("Setup button held: forcing portal.");
  }

  startCaptivePortal();
}

bool connectTCP() {
  setLedMode(LED_TCP_CONNECTING);
  Serial.printf("Connecting TCP (TLS) to %s:%u...\n", SERVER_HOST, SERVER_PORT);
  client.stop();

  client.setInsecure();

  if (client.connect(SERVER_HOST, SERVER_PORT)) {
    client.setNoDelay(true);

    // Big enough so readBytes during long TTS doesn't time out early
    client.setTimeout(15000); // ms

    Serial.println("TCP connected!");
    return true;
  }
  Serial.println("TCP connect failed.");
  return false;
}

static void uninstallI2SIfNeeded() {
  if (i2sMode != MODE_NONE) {
    i2s_driver_uninstall(I2S_PORT);
    i2sMode = MODE_NONE;
  }
}

static void installMicI2S() {
  if (i2sMode == MODE_MIC_RX) return;
  uninstallI2SIfNeeded();

  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = MIC_SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 256,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  i2s_pin_config_t pins = {
    .bck_io_num = MIC_BCLK,
    .ws_io_num = MIC_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = MIC_SD
  };

  esp_err_t err = i2s_driver_install(I2S_PORT, &cfg, 0, NULL);
  if (err != ESP_OK) Serial.printf("MIC i2s_driver_install failed: %d\n", (int)err);

  err = i2s_set_pin(I2S_PORT, &pins);
  if (err != ESP_OK) Serial.printf("MIC i2s_set_pin failed: %d\n", (int)err);

  err = i2s_set_clk(I2S_PORT, MIC_SAMPLE_RATE, I2S_BITS_PER_SAMPLE_32BIT, I2S_CHANNEL_MONO);
  if (err != ESP_OK) Serial.printf("MIC i2s_set_clk failed: %d\n", (int)err);

  i2s_zero_dma_buffer(I2S_PORT);
  i2sMode = MODE_MIC_RX;
}


static void installSpkI2S() {
  if (i2sMode == MODE_SPK_TX) return;
  uninstallI2SIfNeeded();

  if (SPK_EN_PIN >= 0) {
    pinMode(SPK_EN_PIN, OUTPUT);
    digitalWrite(SPK_EN_PIN, HIGH);
  }

  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
    .sample_rate = 22050, // placeholder, overridden per packet
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 16,
    .dma_buf_len = 512,
    .use_apll = false,
    .tx_desc_auto_clear = true,
    .fixed_mclk = 0
  };

  i2s_pin_config_t pins = {
    .bck_io_num = SPK_BCLK,
    .ws_io_num = SPK_WS,
    .data_out_num = SPK_DOUT,
    .data_in_num = I2S_PIN_NO_CHANGE
  };

  esp_err_t err = i2s_driver_install(I2S_PORT, &cfg, 0, NULL);
  if (err != ESP_OK) Serial.printf("SPK i2s_driver_install failed: %d\n", (int)err);

  err = i2s_set_pin(I2S_PORT, &pins);
  if (err != ESP_OK) Serial.printf("SPK i2s_set_pin failed: %d\n", (int)err);

  // will be set to actual SR by i2s_set_clk before playback
  i2s_zero_dma_buffer(I2S_PORT);
  i2sMode = MODE_SPK_TX;
}

// Convert 32-bit I2S mic sample to 16-bit PCM
static inline int16_t convert_sample(int32_t s) {
  const int SHIFT = 14;
  s = s >> SHIFT;
  if (s > 32767) s = 32767;
  if (s < -32768) s = -32768;
  return (int16_t)s;
}

void playPcmFromServer(size_t totalBytes, int sr) {
  receivingAudio = true;

  installSpkI2S();
  i2s_set_clk(I2S_PORT, sr, I2S_BITS_PER_SAMPLE_16BIT, I2S_CHANNEL_STEREO);
  i2s_zero_dma_buffer(I2S_PORT);

  static uint8_t inbuf[1024];
  static int16_t outStereo[1024]; // 512 stereo frames

  const float VOL = 0.23f;
  size_t remaining = totalBytes;

    uint32_t expectedMs = (uint32_t)((totalBytes / 2.0f) * 1000.0f / (float)sr); // mono samples
  uint32_t deadline = millis() + expectedMs + 4000;

  while (remaining > 0 && client.connected()) {
    updateLed();

    // If nothing available yet, wait (until deadline)
    if (client.available() <= 0) {
      if ((int32_t)(millis() - deadline) > 0) break;
      delay(1);
      continue;
    }

    size_t toRead = remaining > sizeof(inbuf) ? sizeof(inbuf) : remaining;
    int n = client.read((uint8_t*)inbuf, toRead); // non-blocking-ish
    if (n <= 0) {
      if ((int32_t)(millis() - deadline) > 0) break;
      delay(1);
      continue;
    }

    int monoSamples = n / 2;
    int16_t* mono = (int16_t*)inbuf;

    int outIdx = 0;
    for (int i = 0; i < monoSamples; i++) {
      int32_t s = mono[i];
      s = (int32_t)(s * VOL);
      if (s > 32767) s = 32767;
      if (s < -32768) s = -32768;
      outStereo[outIdx++] = (int16_t)s;
      outStereo[outIdx++] = (int16_t)s;
    }

    size_t bytesToWrite = (size_t)outIdx * sizeof(int16_t);
    size_t written = 0;
    i2s_write(I2S_PORT, outStereo, bytesToWrite, &written, portMAX_DELAY);

    remaining -= (size_t)n;
  }

  // Back to mic
  installMicI2S();
  receivingAudio = false;
}

void handleServerLine(const String& line) {
  StaticJsonDocument<256> doc;
  DeserializationError err = deserializeJson(doc, line);
  if (err) {
    Serial.print("[SERVER RAW] ");
    Serial.println(line);
    return;
  }

  const char* type = doc["type"] | "";

  if (!strcmp(type, "info")) {
    Serial.print("[INFO] ");
    Serial.println((const char*)(doc["text"] | ""));
    return;
  }

  if (!strcmp(type, "partial")) {
    return;
  }

  if (!strcmp(type, "final")) {
    const char* text = doc["text"] | "";
    if (text[0]) {
      finalCount++;
      Serial.print("[FINAL] ");
      Serial.println(text);
      Serial.print("final_len ");
      Serial.println((int)strlen(text));
      Serial.print("final_count ");
      Serial.println(finalCount);
    }
    return;
  }

  if (!strcmp(type, "tts_audio")) {
    size_t bytes = doc["bytes"] | 0;
    int sr = doc["sr"] | 22050;
    Serial.printf("[TTS_AUDIO] bytes=%u sr=%d\n", (unsigned)bytes, sr);
    if (bytes > 0) playPcmFromServer(bytes, sr);
    return;
  }

  Serial.print("[SERVER] ");
  Serial.println(line);
}

void pollServer() {
  while (client.connected() && client.available()) {
    if (receivingAudio) return;

    String line = client.readStringUntil('\n');
    line.trim();
    if (line.length()) handleServerLine(line);

    // If tts_audio started playback, stop immediately (don't eat PCM as text)
    if (receivingAudio) return;
  }
}


void setup() {
  Serial.begin(115200);
  delay(150);

  pinMode(STATUS_LED_PIN, OUTPUT);
  setLedMode(LED_BOOT);

  connectWiFiOrPortal();

  installMicI2S();

  while (!connectTCP()) {
    updateLed();
    delay(400);
  }

  setLedMode(LED_STREAMING);
}

void loop() {
  updateLed();

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi dropped. Reconnecting...");
    loadWifiFromNVS();
    if (!tryConnectSavedWiFi(12000)) {
      Serial.println("WiFi reconnect failed -> portal");
      startCaptivePortal();
    }
    client.stop();
  }

  if (!client.connected()) {
    Serial.println("TCP disconnected. Reconnecting...");
    setLedMode(LED_TCP_CONNECTING);
    delay(100);
    if (connectTCP()) setLedMode(LED_STREAMING);
    else { delay(300); return; }
  }

  pollServer();
  installMicI2S();

  size_t bytes_read = 0;
  esp_err_t res = i2s_read(I2S_PORT, (void*)i2s_raw, sizeof(i2s_raw), &bytes_read, portMAX_DELAY);
  if (res != ESP_OK || bytes_read == 0) return;

  int samples = bytes_read / sizeof(int32_t);
  if (samples <= 0) return;

  for (int i = 0; i < samples; i++) {
    pcm16[i] = convert_sample(i2s_raw[i]);
  }

  const uint8_t* out_bytes = (const uint8_t*)pcm16;
  const size_t out_len = (size_t)samples * sizeof(int16_t);
  client.write(out_bytes, out_len);

  pollServer();
  delay(0);
}
