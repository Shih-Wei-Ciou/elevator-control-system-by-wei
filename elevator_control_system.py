import cv2
import numpy as np
from ultralytics import YOLO
import pprint
from flask import Flask, Response, render_template, jsonify
import threading
import mysql.connector
from datetime import datetime

# Flask 設定路徑
app = Flask(__name__,
    static_url_path='',
    static_folder='static',
    template_folder='templates')

# YOLO model
model = YOLO("C:\python_yolov8_flaskBrowser1\pythonProject\pt\yolov8n.pt")

# Global variables  狀態
current_status = {
    "coverage": 0,
    "status": "Not Full",
}

current_counts = {
    'person': 0,
    'cart': 0,
    'basket': 0
}

threshold = 70
video_camera = None

save_dir = 'imageRecords'
counter = 1  # 用來命名檔案的編號

# 資料庫
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "elevator_system"  # 要創建的資料庫名稱
}

class VideoCamera:
    def __init__(self): #self用來存取屬性與方法，攝影機的影像擷取與處理，背景捕捉與影像文字顯示
        self.cap = cv2.VideoCapture(0)
        self.current_frame = None #存取當前影像
        self.n = 0 #是否為第一次擷取影像
        self.original_image = None #捕捉第一次的影像
        self.is_running = True #控制攝影機狀態
        # 啟動攝像頭捕獲線程
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.is_running:
            success, frame = self.cap.read() #做截取
            if success:
                rotated_frame = cv2.flip(frame, 1) #做水平翻轉

                # 初次設置原圖像為背景
                if self.n == 0:
                    self.original_image = rotated_frame
                    self.n += 1

                # 添加文字
                text_full_display = current_status["status"]
                coverage_text = f"Coverage: {current_status['coverage']}%"
                org = (50, 50)
                org_full_display = (50, 100)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                color = (255, 255, 255)
                thickness = 2

                cv2.putText(rotated_frame, coverage_text, org, font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(rotated_frame, text_full_display, org_full_display, font, font_scale, color, thickness,
                            cv2.LINE_AA)

                self.current_frame = rotated_frame

    def get_frame(self):#回傳最新的影像
        return self.current_frame

    def __del__(self):#停止與釋放
        self.is_running = False
        self.cap.release()



# 儲存數據至資料庫
def save_to_database(person, shopping_trolley, shopping_basket, coverage, elevator_status):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        query = """
        INSERT INTO system_status 
        (person, ShoppingTrolley, ShoppingBasket, coverage, elevator_status)
        VALUES (%s, %s, %s, %s, %s)
        """

        cursor.execute(query, (person, shopping_trolley, shopping_basket, coverage, elevator_status))
        conn.commit()
        print("檢測數據已成功儲存至資料庫。")

    except mysql.connector.Error as err:
        print(f"資料庫錯誤: {err}")
        print(f"錯誤代碼: {err.errno}")
        print(f"SQL State: {err.sqlstate}")
        print(f"錯誤訊息: {err.msg}")
    except Exception as e:
        print(f"其他錯誤: {e}")
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()


def create_mask(image_shape):
    points = [(267, 305), (133, 717), (828, 717), (711, 322)]
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    pts = np.array([points], np.int32)
    cv2.fillPoly(mask, pts, 255)
    return mask


def process_image(rotated_frame, model):
    global current_counts  # 添加全局變量聲明
    results = model(rotated_frame)
    image = results[0].plot()
    boxes = results[0].boxes.data
    formatted_boxes = []

    # 重置計數器
    counts = {
        'person': 0,
        'cart': 0,
        'basket': 0
    }

    for box in boxes:
        formatted_box = [float(f"{num:.2f}") for num in box.tolist()]
        formatted_boxes.append(formatted_box)

        class_id = int(formatted_box[5])
        if class_id == 0:
            counts['person'] += 1
        elif class_id == 1:
            counts['cart'] += 1
        elif class_id == 2:
            counts['basket'] += 1

    # 更新全局計數器
    current_counts.update(counts)

    return image, formatted_boxes, counts


# 保持原有的 frame_function 和 white_area 函數不變
def frame_function(frame_image, original_image, formatted_boxes, mask):
    frame = cv2.absdiff(original_image, frame_image)  # 原圖减有人
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 轉灰階
    _, output_frame = cv2.threshold(img_gray, 30, 255, cv2.THRESH_BINARY)
    #kernel1 = np.ones((5, 5), np.uint8)
    #output_frame = cv2.dilate(output_frame1, kernel1, iterations=1)
    #output_frame2 = output_frame.copy()
    addimage1 = cv2.subtract(mask, output_frame)
    inverted_image = cv2.bitwise_not(addimage1)
    inverted_image1 = inverted_image.copy()
    # 核心大小
    kernel_1 = np.ones((10, 10), np.uint8)
    kernel_8 = np.ones((5, 5), np.uint8)
    kernel_4 = np.ones((5, 5), np.uint8)

    for box in formatted_boxes:
        x, y, w, h = map(int, box[:4])
        category = int(box[-1])

        # 確定座標
        x = max(0, min(x, inverted_image.shape[1] - 1))
        y = max(0, min(y, inverted_image.shape[0] - 1))
        w = min(w, inverted_image.shape[1] - x)
        h = min(h, inverted_image.shape[0] - y)

        # 選擇不同比例處理
        if category in [1, 8, 4]:
            roi = inverted_image[y:y + h, x:x + w]
            if category == 1:
                dilated_roi = cv2.dilate(roi, kernel_1, iterations=1)
            elif category == 8:
                dilated_roi = cv2.dilate(roi, kernel_8, iterations=1)
            elif category == 4:
                dilated_roi = cv2.dilate(roi, kernel_4, iterations=1)
            inverted_image1[y:y + h, x:x + w] = dilated_roi

    dilated3 = cv2.resize(inverted_image1, (mask.shape[1], mask.shape[0]))
    addimage = cv2.subtract(mask, dilated3)
    return inverted_image, output_frame,addimage


# 計算白區域比例
def white_area(floor_mask_result, addimage):
    floor_mask_result = np.where(floor_mask_result > 150, 255, 0)
    addimage = np.where(addimage > 150, 255, 0)
    floor_white = np.sum(floor_mask_result == 255)
    remain_white = np.sum(addimage == 255)
    num = remain_white / floor_white
    num = num * 100
    rounded_num = round(num)
    return floor_white, remain_white, rounded_num

#本地投影
def local_display():

    global video_camera, counter
    video_camera = VideoCamera()

    while True:
        frame = video_camera.get_frame()
        if frame is not None:
            cv2.imshow('Local Video', frame)

            key = cv2.waitKey(20) & 0xFF
            if key == ord('s'):
                frame_image, formatted_boxes, counts = process_image(frame, model)
                frame_image_path = f"imageRecords/frame_image_image_{counter:03d}.jpg"
                cv2.imwrite(frame_image_path, frame_image)
                mask = create_mask((frame.shape[0], frame.shape[1]))
                pprint.pprint(formatted_boxes)

                if all(box[5] != 0.0 for box in formatted_boxes):
                    print("無人搭乘")
                    continue

                try:
                    inverted_image, output_frame, addimage = frame_function(frame, video_camera.original_image,
                                                                            formatted_boxes, mask)
                    _, _, num = white_area(mask, addimage)

                    current_status["coverage"] = 100 - num
                    coverage=100 - num
                    if coverage > threshold:
                        current_status["status"] = "Full"
                    else:
                        current_status["status"] = "Not Full"

                    print(f"覆蓋比例: {current_status['coverage']}%")
                    print(f"客滿狀態: {current_status['status']}")

                    # 儲存數據到資料庫
                    save_to_database(
                        person=counts['person'],
                        shopping_trolley=counts['cart'],
                        shopping_basket=counts['basket'],
                        coverage=current_status["coverage"],
                        elevator_status=current_status["status"]
                    )

                    # 儲存影像至 imageRecords 資料夾
                    inverted_image_path = f"imageRecords/inverted_image_{counter:03d}.jpg"
                    output_frame_path = f"imageRecords/output_frame_{counter:03d}.jpg"
                    addimage_path = f"imageRecords/addimage_{counter:03d}.jpg"

                    cv2.imwrite(inverted_image_path, inverted_image)
                    cv2.imwrite(output_frame_path, output_frame)
                    cv2.imwrite(addimage_path, addimage)

                    print(f"影像已儲存至: {inverted_image_path}, {output_frame_path}, {addimage_path}")
                    counter += 1
                except Exception as e:
                    print(f"Error during processing: {e}")

            elif key == ord('e'):
                break

    cv2.destroyAllWindows()


def gen_frames():
    while True:
        try:
            if video_camera is not None:
                frame = video_camera.get_frame()
                if frame is not None:
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"視訊串流錯誤: {e}")
            yield b''

#處理不同路徑
@app.route('/')
@app.route('/login.html')
def login():
    return render_template('login.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/results.html')
def results():
    return render_template('results.html')

@app.route('/EnergyConsumption.html')
def EnergyConsumption():
    return render_template('EnergyConsumption.html')

@app.route('/testData.html')
def testData():
    return render_template('testData.html')

@app.route('/charts.html')
def charts():
    return render_template('charts.html')

@app.route('/roi.html')
def roi():
    return render_template('roi.html')

@app.route('/ModeSettings.html')
def ModeSettings():
    return render_template('ModeSettings.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/current_status')
def current_status_api():
    return jsonify(current_status)

@app.route('/api/system_status')
@app.route('/api/system_status')
def get_system_status():
    try:
        return jsonify({
            'coverage': current_status["coverage"],
            'status': current_status["status"],
            'counts': {
                'person': current_counts["person"],
                'cart': current_counts["cart"],
                'basket': current_counts["basket"]
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    try:
        # 確保 imageRecords 資料夾存在
        import os

        if not os.path.exists('imageRecords'):
            os.makedirs('imageRecords')

        # 啟動本地顯示線程
        display_thread = threading.Thread(target=local_display)
        display_thread.daemon = True
        display_thread.start()

        # 啟動 Flask
        app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
    except Exception as e:
        print(f"程式啟動錯誤: {e}")

