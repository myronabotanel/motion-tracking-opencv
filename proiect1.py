import cv2
import numpy as np

# Deschide camera pentru a prelua fluxul video
cap = cv2.VideoCapture(0)

# Initializează un obiect de substracție a fundalului
# Acest obiect identifică regiunile în mișcare pe baza diferențelor față de fundalul static
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Inițializare Kalman Filter
# Kalman Filter este utilizat pentru a prezice poziția obiectului pe baza mișcării anterioare
kalman = cv2.KalmanFilter(4, 2)
# Matricea de măsurare (relatia dintre măsurători și starea curentă)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0]], np.float32)
# Matricea de tranziție (cum evoluează starea în timp)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                     [0, 1, 0, 1],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]], np.float32)
# Matricea de covarianță a zgomotului de proces (pentru modelul predictiv)
kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32) * 0.03

# Poziția inițială (fără măsurători la început)
# Utilizată pentru a inițializa ultima măsurătoare și predicție
last_measurement = np.array((0, 0), np.float32)
last_prediction = np.array((0, 0), np.float32)

while True:
    # Citește un cadru din video
    ret, frame = cap.read()
    if not ret:
        break

    # Aplicăm substracția de fundal pentru a obține masca prim-planului
    fgmask = fgbg.apply(frame)

    # Aplicăm operații morfologice pentru a reduce zgomotul din mască
    # Kernel-ul definește dimensiunea operației morfologice
    kernel = np.ones((7, 7), np.uint8)  # Kernel mai mare pentru a reduce zgomotul
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)  # Închidere (pentru a umple goluri)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)   # Deschidere (pentru a elimina mici zgomote)

    # Găsește contururi pe masca de fundal
    # Contururile identifică regiunile de mișcare semnificativă
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Detectăm obiectul cel mai mare dintre contururi (dacă există)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)  # Găsim cel mai mare contur
        if cv2.contourArea(largest_contour) > 1000:  # Filtrăm contururile mici pentru a evita zgomotul
            x, y, w, h = cv2.boundingRect(largest_contour)  # Găsim coordonatele dreptunghiului

            # Măsurare: calculăm centrul obiectului detectat
            current_measurement = np.array([[np.float32(x + w // 2)], [np.float32(y + h // 2)]]);
            kalman.correct(current_measurement)  # Corectăm predicția cu măsurătoarea curentă

            # Desenează dreptunghiul pe obiectul detectat
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Dreptunghi verde pentru obiect detectat

            last_measurement = current_measurement  # Actualizăm ultima măsurătoare

    # Predicție Kalman pentru următoarea poziție
    current_prediction = kalman.predict()  # Prezicem poziția viitoare a obiectului
    predict_x, predict_y = int(current_prediction[0]), int(current_prediction[1])

    # Desenează punctul prezis de Kalman
    cv2.circle(frame, (predict_x, predict_y), 5, (0, 0, 255), -1)  # Punct roșu pentru predicție

    # Afișează obiectele detectate și predicțiile pe cadrul video
    cv2.imshow("Object Tracking with Kalman Filter", frame)

    # Ieși din buclă când apăsăm tasta 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Eliberează camera și închide ferestrele
cap.release()  # Oprește camera
cv2.destroyAllWindows()  # Închide toate ferestrele
