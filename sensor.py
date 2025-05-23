import time
import csv
import os
from datetime import datetime
from gpiozero import LED
import qwiic_as7265x
import matplotlib.pyplot as plt

# === LED Setup ===
green_led = LED(17)
red_led = LED(27)

# === Sensor Setup ===
sensor = qwiic_as7265x.QwiicAS7265x()
if not sensor.begin():
    print("Sensor not connected. Check wiring.")
    exit()

sensor.set_gain(3)  # 0=1x, 1=3.7x, 2=16x, 3=64x

# === Get Calibrated Values ===
def get_all_calibrated_values(sensor):
    try:
        return [
            sensor.get_calibrated_a(), sensor.get_calibrated_b(), sensor.get_calibrated_c(),
            sensor.get_calibrated_d(), sensor.get_calibrated_e(), sensor.get_calibrated_f(),
            sensor.get_calibrated_g(), sensor.get_calibrated_h(), sensor.get_calibrated_i(),
            sensor.get_calibrated_j(), sensor.get_calibrated_k(), sensor.get_calibrated_l(),
            sensor.get_calibrated_r(), sensor.get_calibrated_s(), sensor.get_calibrated_t(),
            sensor.get_calibrated_u(), sensor.get_calibrated_v(), sensor.get_calibrated_w()
        ]
    except Exception as e:
        print(f"Error reading calibrated values: {e}")
        return None

# === Calibration Functions ===
def capture_calibration(sensor):
    # Turn on white LEDs
    sensor.enable_bulb(0)
    sensor.enable_bulb(1)
    sensor.enable_bulb(2)

    print("🔴 Starting DARK reference calibration...")
    red_led.on()
    green_led.off()
    input("👉 Cover the sensor completely, then press Enter.")
    sensor.take_measurements()
    dark = get_all_calibrated_values(sensor)
    print("✅ Dark reference captured.")
    red_led.off()

    print("🟢 Starting WHITE reference calibration...")
    green_led.on()
    input("👉 Place white reference, then press Enter.")
    sensor.take_measurements()
    white = get_all_calibrated_values(sensor)
    print("✅ White reference captured.")
    green_led.off()

    return dark, white

def normalize_spectral(reading, dark, white):
    return [
        max(0.0, min(1.0, (r - d) / (w - d) if (w - d) != 0 else 0))
        for r, d, w in zip(reading, dark, white)
    ]

# === Save Graph ===
def save_graph(x_labels, y_data, title, folder, timestamp):
    plt.figure()
    plt.plot(x_labels, y_data, marker='o')
    plt.title(title)
    plt.xlabel("Spectral Band")
    plt.ylabel("Normalized Value")
    plt.grid(True)
    filename = os.path.join(folder, f"{title}_{timestamp}.png")
    plt.savefig(filename)
    plt.close()

# === Set up folders ===
desktop_path = os.path.expanduser("~/Desktop")
base_folder = os.path.join(desktop_path, "BTech Project")
healthy_folder = os.path.join(base_folder, "Healthy Leaf")
unhealthy_folder = os.path.join(base_folder, "Unhealthy Leaf")
os.makedirs(healthy_folder, exist_ok=True)
os.makedirs(unhealthy_folder, exist_ok=True)

# === CSV Logging Setup ===
header = ["Timestamp"] + [f"Band{i+1}" for i in range(18)] + [
    "Chlorophyll Status", "Vitamin A Status", "Vitamin C Status", "Moisture Status"
]
csv_path_healthy = os.path.join(healthy_folder, "leaf_analysis_healthy.csv")
csv_path_unhealthy = os.path.join(unhealthy_folder, "leaf_analysis_unhealthy.csv")

for path in [csv_path_healthy, csv_path_unhealthy]:
    if not os.path.exists(path):
        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

# === Thresholds ===
CHL_THRESHOLD = 2.5
VITA_THRESHOLD = 1.8
VITC_THRESHOLD = 2.0
MOISTURE_THRESHOLD = 2.0

band_labels = [f"Band{i+1}" for i in range(18)]

# === Image Capture Setup ===
save_image_dir = os.path.join(desktop_path, "SampleImages")
os.makedirs(save_image_dir, exist_ok=True)

# === Perform Calibration ===
dark_ref, white_ref = capture_calibration(sensor)
print("✅ Calibration done. Ready to scan leaves...\n")

# === Main Loop ===
try:
    while True:
        user_input = input("\n👉 Press Enter to scan a leaf or type 'exit' to quit: ")
        if user_input.lower() == "exit":
            print("👋 Monitoring stopped.")
            break

        # Capture Image
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        image_path = os.path.join(save_image_dir, f"image_{timestamp}.jpg")
        print(f"📷 Capturing image to {image_path}...")
        result = os.system(f"libcamera-still -o \"{image_path}\" --nopreview --immediate")
        if result == 0:
            print("✅ Image saved.")
        else:
            print("❌ Failed to capture image.")

        # Turn on white LEDs before sample measurement
        sensor.enable_bulb(0)
        sensor.enable_bulb(1)
        sensor.enable_bulb(2)

        # Sensor Measurement
        sensor.take_measurements()
        raw = get_all_calibrated_values(sensor)
        if raw is None:
            print("❌ Failed to read sensor data. Try again.")
            continue

        bands = normalize_spectral(raw, dark_ref, white_ref)

        # Define ranges
        chl_signal = sum(bands[7:13])       # H to L
        vit_a_signal = sum(bands[13:18])    # R to W
        vit_c_signal = sum(bands[4:7])      # E, F, G
        moisture_signal = sum(bands[1:4])   # B, C, D

        # Check status
        chl_status = "Sufficient" if chl_signal > CHL_THRESHOLD else "Low"
        vit_a_status = "Sufficient" if vit_a_signal > VITA_THRESHOLD else "Low"
        vit_c_status = "Sufficient" if vit_c_signal > VITC_THRESHOLD else "Low"
        moisture_status = "Sufficient" if moisture_signal > MOISTURE_THRESHOLD else "Low"

        # Determine health
        is_healthy = all(status == "Sufficient" for status in [chl_status, vit_a_status, vit_c_status, moisture_status])

        if is_healthy:
            green_led.on()
            red_led.off()
            print("✅ Healthy leaf detected")
            folder = healthy_folder
            csv_path = csv_path_healthy
        else:
            green_led.off()
            red_led.on()
            print("⚠️  Unhealthy leaf detected")
            folder = unhealthy_folder
            csv_path = csv_path_unhealthy

        # Print statuses
        print(f"Chlorophyll: {chl_status}")
        print(f"Vitamin A  : {vit_a_status}")
        print(f"Vitamin C  : {vit_c_status}")
        print(f"Moisture   : {moisture_status}")
        print("-" * 40)

        # Save to CSV
        with open(csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp] + bands + [chl_status, vit_a_status, vit_c_status, moisture_status])

        # Save Graphs
        save_graph(band_labels[7:13], bands[7:13], "Chlorophyll", folder, timestamp)
        save_graph(band_labels[13:18], bands[13:18], "Vitamin A", folder, timestamp)
        save_graph(band_labels[4:7], bands[4:7], "Vitamin C", folder, timestamp)
        save_graph(band_labels[1:4], bands[1:4], "Moisture", folder, timestamp)

except KeyboardInterrupt:
    print("\n👋 Monitoring stopped.")
    green_led.off()
    red_led.off()