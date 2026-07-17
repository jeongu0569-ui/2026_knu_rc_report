import threading
import time
import cv2
import numpy as np
import json
import os
import queue
import pyaudio
import wave
from flask import Flask, render_template, Response, jsonify, request
from pop import Pilot, LiDAR, Cds
import io

app = Flask(__name__)

# ====================================================
# 1. ⚙️ 하드웨어 및 데이터 로드
# ====================================================
folder_path = os.path.expanduser("~/Project/python/notebook/Untitled Folder")
steer_json_path = os.path.join(folder_path, "steering_calibration.json")
cds_json_path = os.path.join(folder_path, "cds_dnn_weights.json")
sound_dir = os.path.join("sounds")

# [ESC 조향 캘리브레이션 로드]
try:
    with open(steer_json_path, "r", encoding="utf-8") as f:
        calib_data = json.load(f)
    BASE_GYRO = float(calib_data["base_gyro"])
    USABLE_GYRO = float(calib_data["usable_gyro"])
    ZERO_CORRECTION = float(calib_data["zero_correction"])
    print(f"🎯 조향 보정 로드 성공! [직진 자이로]: {BASE_GYRO:.2f} | [가용 회전량]: {USABLE_GYRO:.2f}")
except Exception as e:
    print(f"⚠️ 조향 JSON 로드 실패(기본값 가동): {e}")
    BASE_GYRO = 0.0
    USABLE_GYRO = 1347.14
    ZERO_CORRECTION = 0.0

# [초경량 DNN 가중치 로드]
try:
    with open(cds_json_path, "r", encoding="utf-8") as f:
        weights_data = json.load(f)
    W_0, b_0 = np.array(weights_data["W_0"]), np.array(weights_data["b_0"])
    W_1, b_1 = np.array(weights_data["W_1"]), np.array(weights_data["b_1"])
    W_2, b_2 = np.array(weights_data["W_2"]), np.array(weights_data["b_2"])
    W_3, b_3 = np.array(weights_data["W_3"]), np.array(weights_data["b_3"])
    print("🧠 DNN 가중치 로드 성공!")
    dnn_loaded = True
except Exception as e:
    print(f"⚠️ DNN JSON 로드 실패: {e}")
    dnn_loaded = False

# 하드웨어 드라이버 기동
try:
    cds_sensor = Cds(7)
except Exception:
    cds_sensor = None

try:
    LED = Pilot.PWM(1, 0x5c)
    LED.setFreq(50)
    for channel in range(4):
        LED.setDuty(channel, 0)
except Exception:
    LED = None

try:
    car
except NameError:
    car = Pilot.get_Control()

try:
    lidar = LiDAR.Rplidar()
    lidar.connect()
    lidar.startMotor()
except Exception:
    lidar = None

cam = Pilot.Camera(width=300, height=300)

# 전역 공유 상태 구조체 (Flask 컨트롤러 대응 조이스틱 값 추가)
shared_data = {
    "drive_mode": "MANUAL",    
    "manual_cmd": "stop",      
    "js_steer": 0.0,            # 조이스틱에서 입력받은 수동 조향 오프셋 (-1.0 ~ 1.0)
    "js_throttle": 0.0,         # 조이스틱에서 입력받은 수동 전후진 오프셋 (-1.0 ~ 1.0)
    "ai_steer_value": 0.0,     
    "ai_steer_derivative": 0.0, 
    "latest_frame": None,      
    "is_emergency": False,     
    "nearest_obstacle": 9999.0,
    "current_lux": 1000.0      
}

data_lock = threading.Lock()
is_running = True

# 오디오 시스템
audio_queue = queue.Queue()
pyaudio_instance = pyaudio.PyAudio()

# ====================================================
# 🔈 2. 비동기 백그라운드 오디오 전용 재생 엔진
# ====================================================
def play_wav_file(file_path):
    if not os.path.exists(file_path):
        return
    try:
        w = wave.open(file_path, "rb")
        stream = pyaudio_instance.open(
            format=pyaudio_instance.get_format_from_width(w.getsampwidth()),
            channels=w.getnchannels(),
            rate=w.getframerate(),
            output=True
        )
        data = w.readframes(1024)
        while len(data) > 0 and is_running:
            stream.write(data)
            data = w.readframes(1024)
        stream.stop_stream()
        stream.close()
        w.close()
    except Exception:
        pass

def audio_play_thread_loop():
    global is_running
    while is_running:
        try:
            sound_file = audio_queue.get(timeout=0.2)
            full_path = os.path.join(sound_dir, sound_file)
            play_wav_file(full_path)
            audio_queue.task_done()
        except queue.Empty:
            continue
        except Exception:
            pass

def trigger_audio(file_name):
    audio_queue.put(file_name)

def predict_lux_numpy(cds_value):
    if not dnn_loaded:
        return 1000.0 if cds_value > 500 else 100.0
    def relu(x): return np.maximum(0, x)
    x = np.array([[cds_value]], dtype=float)
    h1 = relu(np.dot(x, W_0) + b_0)
    h2 = relu(np.dot(h1, W_1) + b_1)
    h3 = relu(np.dot(h2, W_2) + b_2)
    return float((np.dot(h3, W_3) + b_3)[0][0])

# ====================================================
# 📡 3. 독립 LiDAR 장애물 감시 스레드
# ====================================================
def lidar_scan_thread_loop():
    global is_running
    last_emergency_state = False
    while is_running:
        if lidar is None:
            time.sleep(0.1)
            continue
        try:
            vectors = lidar.getVectors() 
            collision_detected = False
            min_dist = 9999.0
            for v in vectors:
                degree = v[0]
                distance = v[1]
                if degree <= 60 or degree >= 300:
                    if 0.0 < distance <= 400.0:
                        collision_detected = True
                        if distance < min_dist:
                            min_dist = distance
            
            if collision_detected and not last_emergency_state:
                trigger_audio("obstacle_alert.wav")
                
            last_emergency_state = collision_detected

            with data_lock:
                shared_data["is_emergency"] = collision_detected
                shared_data["nearest_obstacle"] = min_dist if collision_detected else 9999.0
        except Exception:
            pass
        time.sleep(0.02)

# ====================================================
# 🌙 4. 오토 라이트 스레드
# ====================================================
def auto_headlight_thread_loop():
    global is_running
    while is_running:
        if cds_sensor is None or LED is None:
            time.sleep(0.5)
            continue
        try:
            raw_cds = cds_sensor.readAverage()
            predicted_lux = predict_lux_numpy(raw_cds)
            with data_lock:
                shared_data["current_lux"] = predicted_lux
            
            if predicted_lux <= 450.0:
                LED.setDuty(0, 90)
                LED.setDuty(1, 90)
            else:
                LED.setDuty(0, 0)
                LED.setDuty(1, 0)
        except Exception:
            pass
        time.sleep(0.1)

# ====================================================
# 🧠 5. OpenCV 라인 가공 스레드 (최적화 버전)
# ====================================================
def vision_opencv_thread_loop():
    global is_running
    last_error = 0.0
    last_time = time.time()
    
    while is_running:
        try:
            frame = cam.value
            current_time = time.time()
            dt = current_time - last_time
            if dt <= 0:
                dt = 0.02
                
            if frame is not None:
                h, w, _ = frame.shape
                roi_top = int(h * 0.6)
                roi = frame[roi_top:h, 0:w]
                
                blurred = cv2.GaussianBlur(roi, (5, 5), 0)
                hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
                
                lower_yellow = np.array([15, 80, 80])
                upper_yellow = np.array([36, 255, 255])
                thresh = cv2.inRange(hsv, lower_yellow, upper_yellow)
                M = cv2.moments(thresh)
                
                steer_x = 0.0
                derivative = 0.0
                
                draw_frame = frame.copy()
                center_screen = int(w / 2)
                cv2.line(draw_frame, (center_screen, 0), (center_screen, h), (255, 0, 0), 2)
                cv2.line(draw_frame, (0, roi_top), (w, roi_top), (0, 255, 0), 2)
                
                if M["m00"] > 120:  
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"]) + roi_top
                    error = (cx - center_screen) / center_screen
                    steer_x = max(-1.0, min(1.0, error))
                    derivative = (error - last_error) / dt
                    last_error = error
                    
                    cv2.circle(draw_frame, (cx, cy), 8, (0, 255, 255), -1)
                    cv2.line(draw_frame, (center_screen, cy), (cx, cy), (0, 0, 255), 2)
                else:
                    steer_x = 0.0
                    derivative = 0.0
                
                last_time = current_time
                
                with data_lock:
                    emergency = shared_data["is_emergency"]
                    obs_dist = shared_data["nearest_obstacle"]
                    lux_val = shared_data["current_lux"]
                    mode = shared_data["drive_mode"]
                
                # 가독성 UI 오버레이 추가
                cv2.putText(draw_frame, f"MODE: {mode}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(draw_frame, f"Light: {lux_val:.1f} Lux", (10, h - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if lux_val > 450.0 else (0, 165, 255), 2)
                
                if emergency:
                    if LED is not None:
                        LED.setDuty(2, 90)
                        LED.setDuty(3, 90)
                    cv2.rectangle(draw_frame, (0, 0), (w, h), (0, 0, 255), 10)
                    cv2.putText(draw_frame, f"EMERGENCY! {obs_dist:.0f}mm", (15, 55), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
                else:
                    if LED is not None:
                        LED.setDuty(2, 0)
                        LED.setDuty(3, 0)
                
                with data_lock:
                    shared_data["ai_steer_value"] = steer_x
                    shared_data["ai_steer_derivative"] = derivative
                    shared_data["latest_frame"] = draw_frame
        except Exception:
            pass
        time.sleep(0.01)

# ====================================================
# 🏎️ 6. 50Hz 메인 제어 루프 (PD 게인 튜닝 & 조이스틱 매핑)
# ====================================================
def main_control_loop():
    global is_running
    default_speed = 50
    
    # 🎯 지그재그 제어 오버슈팅 방지 파라미터
    Kp = 0.5   
    Kd = 0.22  
    
    while is_running:
        with data_lock:
            mode = shared_data["drive_mode"]
            cmd = shared_data["manual_cmd"]
            js_steer = shared_data["js_steer"]
            js_throttle = shared_data["js_throttle"]
            ai_steer = shared_data["ai_steer_value"]
            ai_deriv = shared_data["ai_steer_derivative"]
            emergency = shared_data["is_emergency"]
        
        # 비상 제동 상황 공통 처리
        if emergency:
            car.stop()
            car.steering = ZERO_CORRECTION
            time.sleep(0.02)
            continue

        # 1. 자율 주행 모드
        if mode == "AUTO":
            car.forward(default_speed)
            pd_steer_value = (ai_steer * Kp) + (ai_deriv * Kd)
            base_target_steer = pd_steer_value * 1.2
            
            current_gyro = car.getGyro("z")
            expected_gyro = BASE_GYRO + (base_target_steer * USABLE_GYRO)
            gyro_error = current_gyro - expected_gyro
            
            esc_counter_value = -(gyro_error / USABLE_GYRO)
            esc_stabilizer = esc_counter_value / 3.0
            
            final_steering = base_target_steer + ZERO_CORRECTION + esc_stabilizer
            car.steering = max(-1.0, min(1.0, final_steering))
                
        # 2. 수동 제어 모드 (조이스틱 매핑 최우선)
        elif mode == "MANUAL":
            # 조이스틱 입력 유무 검사 (임계치 0.05 설정)
            if abs(js_throttle) > 0.05 or abs(js_steer) > 0.05:
                # 전진 / 후진 서보 제어
                target_speed = int(abs(js_throttle) * default_speed)
                if js_throttle > 0:
                    car.forward(target_speed)
                else:
                    car.backward(target_speed)
                
                # 조향각 보정 결합 적용
                car.steering = max(-1.0, min(1.0, js_steer + ZERO_CORRECTION))
            else:
                # 조이스틱 입력이 없으면 버튼 인터페이스의 기본 커맨드로 백업 처리
                if cmd == "forward": 
                    car.steering = ZERO_CORRECTION
                    car.forward(default_speed)
                elif cmd == "backward": 
                    car.steering = ZERO_CORRECTION
                    car.backward(default_speed)
                elif cmd == "left": 
                    car.steering = -1.0
                    car.forward(default_speed)
                elif cmd == "right": 
                    car.steering = 1.0
                    car.forward(default_speed)
                else:
                    car.stop()
                    car.steering = ZERO_CORRECTION
            
        time.sleep(0.02)

# ====================================================
# 🌐 7. Flask 라우터 구성
# ====================================================
@app.route('/')
def index():
    # templates/index.html 페이지를 렌더링
    return render_template('index.html')

def gen_frames():
    """웹 브라우저를 향한 실시간 MJPEG 스트리밍 생성기"""
    while is_running:
        with data_lock:
            frame = shared_data["latest_frame"]
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    """주기적인 클라이언트 UI 업데이트를 위한 상태 반환 API"""
    with data_lock:
        status = {
            "drive_mode": shared_data["drive_mode"],
            "is_emergency": shared_data["is_emergency"],
            "nearest_obstacle": round(shared_data["nearest_obstacle"], 1),
            "current_lux": round(shared_data["current_lux"], 1),
            "js_steer": round(shared_data["js_steer"], 2)
        }
    return jsonify(status)

@app.route('/control/action', methods=['POST'])
def control_action():
    """모드 스위칭, 경적 등 기본 이벤트 처리 API"""
    data = request.json
    action = data.get("action")
    
    global is_running
    with data_lock:
        if action == "auto":
            shared_data["drive_mode"] = "AUTO"
            trigger_audio("auto_mode.wav")
        elif action == "manual":
            shared_data["drive_mode"] = "MANUAL"
            shared_data["manual_cmd"] = "stop"
            shared_data["js_steer"] = 0.0
            shared_data["js_throttle"] = 0.0
            trigger_audio("manual_mode.wav")
        elif action == "horn":
            trigger_audio("horn.wav")
        elif action == "stop":
            shared_data["manual_cmd"] = "stop"
            shared_data["js_steer"] = 0.0
            shared_data["js_throttle"] = 0.0
            
    return jsonify({"status": "success", "action": action})

@app.route('/control/joystick', methods=['POST'])
def control_joystick():
    """HTML5 가상 조이스틱 값 수집 API"""
    data = request.json
    steer = float(data.get("steer", 0.0))       # -1.0(좌) ~ 1.0(우)
    throttle = float(data.get("throttle", 0.0)) # -1.0(후진) ~ 1.0(전진)
    
    with data_lock:
        # 수동 모드일 때만 적용
        if shared_data["drive_mode"] == "MANUAL":
            shared_data["js_steer"] = steer
            shared_data["js_throttle"] = throttle
            
    return jsonify({"status": "success"})

# ====================================================
# 🧹 8. 안전 소등 및 기기 해제 소멸자
# ====================================================
def cleanup():
    global is_running
    is_running = False
    time.sleep(0.5)
    try:
        car.stop()
        car.steering = ZERO_CORRECTION
    except Exception: pass
    if LED is not None:
        try:
            for c in range(4): LED.setDuty(c, 0)
        except Exception: pass
    if lidar:
        try: lidar.stopMotor()
        except Exception: pass
    try: pyaudio_instance.terminate()
    except Exception: pass
    print("🧹 모든 센서 및 장치 릴리즈 완료.")

if __name__ == "__main__":
    # 제어용 핵심 스레드 기동
    threading.Thread(target=main_control_loop, daemon=True).start()
    threading.Thread(target=vision_opencv_thread_loop, daemon=True).start()
    threading.Thread(target=lidar_scan_thread_loop, daemon=True).start()
    threading.Thread(target=auto_headlight_thread_loop, daemon=True).start()
    threading.Thread(target=audio_play_thread_loop, daemon=True).start()
    
    try:
        # Flask 가동 (주피터 환경 외부 접속을 위해 0.0.0.0 포트 5000 바인딩)
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
