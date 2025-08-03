**Car Counting System (ESP32 + Ultrasonic Sensor)**

****

A self-sufficient car detection system designed to count vehicles entering a facility using an ESP32 microcontroller and ultrasonic sensing. The data is locally broadcast via Wi-Fi and displayed in real-time inside the building.

Overview:
This project was developed to help a community organization manage logistics during large events. Prior to this, they had no way to anticipate how many cars and attendees were on site until events were already underway. This solution provides an automated, low-cost car counting mechanism to assist with food, space, and resource planning.

Features
- ESP32 + Ultrasonic Sensor for vehicle detection
- Debounce logic to reduce false positives
- 3D-printed housing for reliable outdoor use
- Wi-Fi Access Point mode — connects to a dedicated Android device for live readout
- No cloud dependency — data stays on-site

Hardware
- ESP32 Dev Module
- HC-SR04 Ultrasonic Sensor
- Custom 3D-printed enclosure
- Power bank for off-grid operation
- Android phone for display

How It Works
- An ultrasonic sensor mounted roadside detects cars based on distance and duration.
- The ESP32 increments a counter when a valid vehicle passes.
- A web server hosted on the ESP32 displays the car count.
- An Android phone inside the building stays connected to the ESP32’s Wi-Fi and shows the count to staff.
  
Future Improvements
- Add two-way traffic detection
- Push data to cloud or mobile app
- Solar-powered version for permanent outdoor deployment

Demo Coming Soon
