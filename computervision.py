import cv2
import numpy as np

# تشغيل الكاميرا
cap = cv2.VideoCapture(0)

# إنشاء لوحة الرسم
canvas = None

# حدود اللون الأخضر في HSV
lower_green = np.array([35, 80, 80])
upper_green = np.array([85, 255, 255])

previous_point = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # عكس الصورة
    frame = cv2.flip(frame, 1)

    # إنشاء لوحة فارغة بنفس حجم الصورة
    if canvas is None:
        canvas = np.zeros_like(frame)

    # تحويل إلى HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # إنشاء ماسك للون الأخضر
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # تحسين الماسك
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # العثور على الحدود
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) > 500:
            x, y, w, h = cv2.boundingRect(largest)
            center = (x + w // 2, y + h // 2)

            # رسم نقطة على الجسم
            cv2.circle(frame, center, 8, (0, 0, 255), -1)

            # رسم الخط
            if previous_point is not None:
                cv2.line(canvas, previous_point, center, (255, 0, 0), 5)

            previous_point = center
    else:
        previous_point = None

    # دمج الكاميرا مع اللوحة
    output = cv2.add(frame, canvas)

    # كتابة تعليمات
    cv2.putText(output, "Press C to clear | Press Q to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2)

    cv2.imshow("Virtual Drawing Pad", output)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF

    # مسح الرسم
    if key == ord('c'):
        canvas = np.zeros_like(frame)

    # الخروج
    elif key == ord('q'):
        break

# إغلاق الكاميرا
cap.release()
cv2.destroyAllWindows()