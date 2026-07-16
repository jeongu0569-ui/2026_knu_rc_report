import threading
import time
import cv2
import numpy as np
import json
import os
import queue  # 오디오 재생 큐 관리용
import pyaudio
import wave
from pop import Pilot, LiDAR, Cds
from IPython.display import display
import PIL.Image
import io

# ====================================================
# 1. ⚙️ 하드웨어 및 다른 폴더의 JSON 데이터 로드
# ====================================================
folder_path = os.path.expanduser("~/Project/python/notebook/Untitled Folder")
steer_json_path = os.path.join(folder_path, "steering_calibration.json")
cds_json_path = os.path.join(folder_path, "cds_dnn_weights.json")

# 오디오 파일이 존재하는 sounds 폴더 경로 지정
sound_dir = os.path.join("sounds")

# ⚖️ [ESC 조향 캘리브레이션 로드]
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

# 🧠 [초경량 DNN 가중치 로드 및 행렬 구성]
try:
    with open(cds_json_path, "r", encoding="utf-8") as f:
        weights_data = json.load(f)
    
    W_0 = np.array(weights_data["W_0"])
    b_0 = np.array(weights_data["b_0"])
    W_1 = np.array(weights_data["W_1"])
    b_1 = np.array(weights_data["b_1"])
    W_2 = np.array(weights_data["W_2"])
    b_2 = np.array(weights_data["b_2"])
    W_3 = np.array(weights_data["W_3"])
    b_3 = np.array(weights_data["b_3"])
    
    print("🧠 DNN 가중치 로드 성공! 텐서플로우 없이 실시간 예측 엔진 구동 시작.")
    dnn_loaded = True
except Exception as e:
    print(f"⚠️ DNN JSON 로드 실패: {e}")
    dnn_loaded = False

# 💡 하드웨어 기기들 기동
try:
    cds_sensor = Cds(7)
    print("💡 7번 핀 조도 센서(Cds) 가동 완료.")
except Exception as e:
    print(f"⚠️ 조도 센서 초기화 실패: {e}")
    cds_sensor = None

try:
    LED = Pilot.PWM(1, 0x5c)
    LED.setFreq(50)
    for channel in range(4):
        LED.setDuty(channel, 0)
except Exception as e:
    print(f"⚠️ LED 하드웨어 초기화 실패: {e}")
    LED = None

try:
    car
except NameError:
    print("⚙️ 오토카 객체 초기화...")
    car = Pilot.get_Control()

print("📡 LiDAR 센서 모터 기동 중...")
try:
    lidar = LiDAR.Rplidar()
    lidar.connect()
    lidar.startMotor()
except Exception as e:
    print(f"⚠️ LiDAR 연결 실패: {e}")
    lidar = None

print("📸 비동기 카메라 시스템 로드...")
cam = Pilot.Camera(width=300, height=300)

# 전역 공유 자원 구조 (PD 제어용 미분값 변수 추가)
shared_data = {
    "drive_mode": "MANUAL",    
    "manual_cmd": "stop",      
    "ai_steer_value": 0.0,     
    "ai_steer_derivative": 0.0, # 조향 오차 변화율(D 제어용)
    "latest_frame": None,      
    "is_emergency": False,     
    "nearest_obstacle": 9999.0,
    "current_lux": 1000.0      
}

data_lock = threading.Lock()
is_running = True

# 오디오 제어를 위한 비동기 큐 & PyAudio 초기화
audio_queue = queue.Queue()
pyaudio_instance = pyaudio.PyAudio()

# ====================================================
# 🔈 2-1. 비동기 백그라운드 오디오 전용 재생 엔진
# ====================================================
def play_wav_file(file_path):
    """지정된 wav 파일을 PyAudio 스트림을 통해 차단 없이 재생하는 함수"""
    if not os.path.exists(file_path):
        print(f"⚠️ 오디오 파일 없음: {file_path}")
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
    except Exception as e:
        print(f"⚠️ 오디오 재생 오류: {e}")

def audio_play_thread_loop():
    """큐에 들어온 오디오 파일명들을 순차적으로 비동기 재생하는 백그라운드 루프"""
    global is_running
    print("🔈 [Audio System] 오디오 재생 엔진 가동 성공.")
    while is_running:
        try:
            # 큐가 비어있을 때 대기하되, 프로그램 종료 시 신속하게 빠져나가기 위해 타임아웃을 둡니다.
            sound_file = audio_queue.get(timeout=0.2)
            full_path = os.path.join(sound_dir, sound_file)
            play_wav_file(full_path)
            audio_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            pass

def trigger_audio(file_name):
    """외부에서 메인 스레드를 멈추지 않고 소리 출력을 요청하는 헬퍼 함수"""
    audio_queue.put(file_name)

# ====================================================
# 2-2. ⚡ 텐서플로우 없는 순수 NumPy DNN 예측 함수
# ====================================================
def predict_lux_numpy(cds_value):
    if not dnn_loaded:
        return 1000.0 if cds_value > 500 else 100.0
        
    def relu(x):
        return np.maximum(0, x)
        
    x = np.array([[cds_value]], dtype=float)
    
    h1 = relu(np.dot(x, W_0) + b_0)
    h2 = relu(np.dot(h1, W_1) + b_1)
    h3 = relu(np.dot(h2, W_2) + b_2)
    out = np.dot(h3, W_3) + b_3
    
    return float(out[0][0])

# ====================================================
# 3. 📡 독립 LiDAR 장애물 감시 스레드 (오디오 비동기 경고 연동)
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
                    if 0.0 < distance <= 400.0: # 40cm 기준 유지
                        collision_detected = True
                        if distance < min_dist:
                            min_dist = distance
            
            # 비상 상황 감지 시점 변화 추적하여 최초 감지 시 오디오 재생 트리거
            if collision_detected and not last_emergency_state:
                trigger_audio("obstacle_alert.wav")
                
            last_emergency_state = collision_detected

            with data_lock:
                shared_data["is_emergency"] = collision_detected
                shared_data["nearest_obstacle"] = min_dist if collision_detected else 9999.0
        except Exception as e:
            pass
        time.sleep(0.02)

# ====================================================
# 4. 🌙 초스피드 오토 라이트 스레드
# ====================================================
def auto_headlight_thread_loop():
    global is_running
    print("🌙 [Auto Light] 초경량 실시간 DNN 조도 예측 스레드 가동...")
    while is_running:
        if cds_sensor is None or LED is None:
            time.sleep(0.5)
            continue
        try:
            raw_cds = cds_sensor.readAverage()
            predicted_lux = predict_lux_numpy(raw_cds)
            
            with data_lock:
                shared_data["current_lux"] = predicted_lux
            
            # 💡 조도 임계값 450 Lux 기준 라이트 제어
            if predicted_lux <= 450.0:
                LED.setDuty(0, 90)
                LED.setDuty(1, 90)
            else:
                LED.setDuty(0, 0)
                LED.setDuty(1, 0)
                
        except Exception as e:
            pass
        time.sleep(0.1)

# ====================================================
# 5. 🧠 백그라운드 OpenCV 라인 연산 스레드 (연산 속도 극대화 버전)
# ====================================================
def vision_opencv_thread_loop():
    global is_running
    print("👁️ [OpenCV Vision] 연산 최적화 노란색 라인 추적 및 오차 연산 개시...")
    
    last_error = 0.0
    last_time = time.time()
    
    while is_running:
        try:
            # 딜레이 최소화를 위해 1회만 고속 취득
            frame = cam.value
            current_time = time.time()
            dt = current_time - last_time
            if dt <= 0:
                dt = 0.02
                
            if frame is not None:
                h, w, _ = frame.shape
                roi_top = int(h * 0.6)
                
                # [최적화] 큰 이미지를 가공하지 않고 ROI 관심영역을 먼저 크롭하여 연산량 60% 감소시킴
                roi = frame[roi_top:h, 0:w]
                
                # 가우시안 블러 및 HSV 변환을 ROI에만 제한 적용
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
                    
                    # 수렴 오차의 급변(미분치) 연산으로 반대 방향 댐핑력 계산
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
                
                cv2.putText(draw_frame, f"Light: {lux_val:.1f} Lux", (10, h - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if lux_val > 450.0 else (0, 165, 255), 2)
                
                if emergency:
                    if LED is not None:
                        LED.setDuty(2, 90)
                        LED.setDuty(3, 90)
                    cv2.rectangle(draw_frame, (0, 0), (w, h), (0, 0, 255), 10)
                    cv2.putText(draw_frame, f"EMERGENCY! {obs_dist:.0f}mm", (15, 45), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
                else:
                    cv2.putText(draw_frame, f"Steer: {steer_x:+.2f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    if LED is not None:
                        LED.setDuty(2, 0)
                        LED.setDuty(3, 0)
                
                with data_lock:
                    shared_data["ai_steer_value"] = steer_x
                    shared_data["ai_steer_derivative"] = derivative
                    shared_data["latest_frame"] = draw_frame
        except Exception as e:
            pass
        # 리소스 스케줄링 간섭을 피하기 위한 최소한의 슬립
        time.sleep(0.01)

# ====================================================
# 6. 🎨 비동기 주피터 디스플레이 전용 스레드
# ====================================================
def display_thread_loop():
    global is_running
    display_handle = display(None, display_id=True)
    while is_running:
        try:
            with data_lock:
                draw_frame = shared_data["latest_frame"]
            if draw_frame is not None:
                _, encoded_img = cv2.imencode('.jpg', draw_frame)
                img_bytes = io.BytesIO(encoded_img.tobytes())
                pil_img = PIL.Image.open(img_bytes)
                display_handle.update(pil_img)
        except Exception as e:
            pass
        time.sleep(0.06)

# ====================================================
# 7. ⚡ 50Hz 메인 제어 루프 (지그재그 제어 안정화 게인 튜닝 버전)
# ====================================================
def main_control_loop():
    global is_running
    print("\n🏎️ [Main Control] 사령탑 가동 (PD 오버슈팅 방지 튜닝 완료)")
    
    current_speed = 50
    car.setSpeed(current_speed)
    
    # 🎯 지그재그 방지를 위해 조향 게인값 하향 및 댐핑 게인값 대폭 상향 조정
    Kp = 0.5   # 비례 게인 (급진적인 직진 구역 흔들림 제어)
    Kd = 0.22  # 미분 게인 (차량이 회전 관성에 의해 반대로 튀는 성질 억제)
    
    while is_running:
        with data_lock:
            mode = shared_data["drive_mode"]
            cmd = shared_data["manual_cmd"]
            ai_steer = shared_data["ai_steer_value"]
            ai_deriv = shared_data["ai_steer_derivative"]
            emergency = shared_data["is_emergency"]
        
        # 1. 자율주행 모드(AUTO) 제어
        if mode == "AUTO":
            if emergency:
                car.stop()
                car.steering = ZERO_CORRECTION
            else:
                car.forward(current_speed)
                
                # 🚀 튜닝된 안정형 PD 제어량 연산
                pd_steer_value = (ai_steer * Kp) + (ai_deriv * Kd)
                base_target_steer = pd_steer_value * 1.2
                
                current_gyro = car.getGyro("z")
                expected_gyro = BASE_GYRO + (base_target_steer * USABLE_GYRO)
                gyro_error = current_gyro - expected_gyro
                
                esc_counter_value = -(gyro_error / USABLE_GYRO)
                esc_stabilizer = esc_counter_value / 3.0
                
                final_steering = base_target_steer + ZERO_CORRECTION + esc_stabilizer
                car.steering = max(-1.0, min(1.0, final_steering))
                
        # 2. 수동 제어 모드(MANUAL) 제어
        elif mode == "MANUAL":
            if cmd == "forward": 
                car.steering = ZERO_CORRECTION
                car.forward(current_speed)
            elif cmd == "backward": 
                car.steering = ZERO_CORRECTION
                car.backward(current_speed)
            elif cmd == "left": 
                car.steering = -1.0
                car.forward(current_speed)
            elif cmd == "right": 
                car.steering = 1.0
                car.forward(current_speed)
            elif cmd == "stop": 
                car.stop()
                car.steering = ZERO_CORRECTION
            
        time.sleep(0.02)
        
# ====================================================
# 8. 인프라 실행 및 안전 종료 컨트롤러 (Main Thread)
# ====================================================
if __name__ == "__main__":
    is_running = True
    
    # 스레드 시작
    threading.Thread(target=main_control_loop, daemon=True).start()
    threading.Thread(target=vision_opencv_thread_loop, daemon=True).start()
    threading.Thread(target=display_thread_loop, daemon=True).start()
    threading.Thread(target=lidar_scan_thread_loop, daemon=True).start()
    threading.Thread(target=auto_headlight_thread_loop, daemon=True).start()
    threading.Thread(target=audio_play_thread_loop, daemon=True).start() # 오디오 비동기 스레드 실행
    
    time.sleep(1.0)
    
    print("\n=== 🕹️ 오토카 통합 제어 콘솔 (카메라 지연 및 조향 최적화 버전) ===")
    print("수동 명령어: w(전진), s(후진), a(좌회전), d(우회전), x(정지), h(경적)")
    print("모드 변경: auto(자율주행), manual(수동모드)")
    print("종료: q")
    print("==============================================================\n")
    
    while is_running:
        user_input = input("🎮 명령을 입력하세요: ").strip().lower()
        if user_input == 'q':
            is_running = False
            break
        with data_lock:
            if user_input == 'auto':
                shared_data["drive_mode"] = "AUTO"
                print("🔄 [모드 변경] 자율주행(AUTO) 시작!")
                trigger_audio("auto_mode.wav")  # 비동기 오디오 트리거
                
            elif user_input == 'manual':
                shared_data["drive_mode"] = "MANUAL"
                shared_data["manual_cmd"] = "stop"
                print("🔄 [모드 변경] 수동(MANUAL) 모드로 전환!")
                trigger_audio("manual_mode.wav") # 비동기 오디오 트리거
                
            elif shared_data["drive_mode"] == "MANUAL":
                if user_input == 'w': 
                    shared_data["manual_cmd"] = "forward"
                elif user_input == 's': 
                    shared_data["manual_cmd"] = "backward"
                    trigger_audio("back.wav") # 후진 시 알림음 트리거
                elif user_input == 'a': 
                    shared_data["manual_cmd"] = "left"
                elif user_input == 'd': 
                    shared_data["manual_cmd"] = "right"
                elif user_input == 'x': 
                    shared_data["manual_cmd"] = "stop"
                elif user_input == 'h':
                    trigger_audio("horn.wav") # 수동 경적 트리거

    # ====================================================
    # 🧹 [안전 종료 클린업]
    # ====================================================
    print("\n🛑 오토카 종료 시퀀스를 시작합니다...")
    time.sleep(0.3)
    
    try:
        car.stop()
        car.steering = ZERO_CORRECTION
        print("✅ 오토카 주행 정지 완료.")
    except Exception as e:
        print(f"⚠️ 주행 정지 실패: {e}")

    if LED is not None:
        try:
            for channel in range(4):
                LED.setDuty(channel, 0)
            print("✅ 전방 및 후방 LED 전원 소등 완료.")
        except Exception as e:
            print(f"⚠️ LED 소등 실패: {e}")

    if lidar:
        try:
            lidar.stopMotor()
            print("✅ LiDAR 모터 기동 정지 완료.")
        except Exception as e:
            print(f"⚠️ LiDAR 정지 실패: {e}")

    # PyAudio 리소스 해제
    try:
        pyaudio_instance.terminate()
        print("✅ 오디오 출력 기기 해제 완료.")
    except Exception as e:
        pass

    print("\n🏎️ 모든 센서와 디바이스가 안전하게 클린업되었습니다. 프로그램을 완전히 종료합니다.")
