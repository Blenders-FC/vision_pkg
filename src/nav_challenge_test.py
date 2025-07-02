import cv2 as cv
import numpy as np

# --- Parámetros HSV para rojo y azul ---
lower_red1 = np.array([0, 100, 100], np.uint8)
upper_red1 = np.array([10, 255, 255], np.uint8)
lower_red2 = np.array([170, 100, 100], np.uint8)
upper_red2 = np.array([180, 255, 255], np.uint8)

azul_low = np.array([100, 100, 100], np.uint8)
azul_high = np.array([140, 255, 255], np.uint8)

kernel = np.ones((5, 5), np.uint8)

# --- Función para detectar obstáculos ---
def deteccionEquipo(subsection):
    hsv = cv.cvtColor(subsection, cv.COLOR_BGR2HSV)

    mask_red1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv.bitwise_or(mask_red1, mask_red2)
    mask_red_clean = cv.morphologyEx(mask_red, cv.MORPH_OPEN, kernel)

    mask_blue = cv.inRange(hsv, azul_low, azul_high)
    mask_blue_clean = cv.morphologyEx(mask_blue, cv.MORPH_OPEN, kernel)

    return np.any(mask_red_clean) or np.any(mask_blue_clean)

# --- Función para navegación recursiva ---
def navigation(frame):
    height, width = frame.shape[:2]
    lower_half = frame[height // 2:, :width]
    xi, xf = 0, lower_half.shape[1]
    yi, yf = 0, lower_half.shape[0]

    max_sections = 20
    sections = 3

    print("Iniciando navegación en zona inferior")

    while sections <= max_sections:
        div_x = (xf - xi) // sections

        # Buscar primero desde el centro hacia la derecha
        for offset in range(sections // 2, sections):
            start = offset * div_x
            end = min((offset + 1) * div_x, xf)
            section = lower_half[yi:yf, start:end]

            if not deteccionEquipo(section):
                print(f"Vía libre en subzona [{start}, {end}] ({sections} divisiones)")
                cv.rectangle(lower_half, (start, 0), (end, yf), (0, 255, 0), 2)
                return

        # Luego desde el centro hacia la izquierda
        for offset in reversed(range(0, sections // 2)):
            start = offset * div_x
            end = min((offset + 1) * div_x, xf)
            section = lower_half[yi:yf, start:end]

            if not deteccionEquipo(section):
                print(f"Vía libre en subzona [{start}, {end}] ({sections} divisiones)")
                cv.rectangle(lower_half, (start, 0), (end, yf), (0, 255, 0), 2)
                return

        print(f"Obstáculos en {sections} zonas, refinando...")
        sections += 2

    print("No se encontró vía libre tras múltiples divisiones")


# --- Procesar y visualizar ---
def procesar_frame(frame):
    navigation(frame)
    # Solo para mostrar el frame original 
    cv.imshow("Frame completo", frame)

def main():
    # Imagen fija:
    #frame = cv.imread("C:/Users/Ivani/humanoides/vision_pkg/media/1.jpg")
    #frame = cv.imread("C:/Users/Ivani/humanoides/vision_pkg/media/2.jpg")
    #frame = cv.imread("C:/Users/Ivani/humanoides/vision_pkg/media/3.jpg")
    #frame = cv.imread("C:/Users/Ivani/humanoides/vision_pkg/media/4.jpg")

    #if frame is None:
    #     print("Imagen no encontrada")
    #     return
    #procesar_frame(frame)
    #cv.waitKey(0)

    # O usar cámara:
    cap = cv.VideoCapture("C:/Users/Ivani/humanoides/vision_pkg/media/video2.mp4")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error al leer la cámara")
            break

        procesar_frame(frame)

        if cv.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
