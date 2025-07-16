import cv2

# Yüz algılayıcı model yükleniyor
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Kamerayı başlat
kamera = cv2.VideoCapture(0)

while True:
    ret, frame = kamera.read()
    if not ret:
        break

    # Gri tonlamaya çevir
    gri = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüzleri tespit et
    yuzler = face_cascade.detectMultiScale(gri, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in yuzler:
        # Yüz bölgesini al
        yuz = frame[y:y+h, x:x+w]
        # Bulanıklaştır
        bulanik_yuz = cv2.GaussianBlur(yuz, (99, 99), 30)
        # Orijinal görüntüdeki yerine koy
        frame[y:y+h, x:x+w] = bulanik_yuz

    # Görüntüyü göster
    cv2.imshow("Yüz Bulanıklaştırma", frame)

    # Çıkmak için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()
