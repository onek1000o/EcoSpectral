
import os
import torch
import warnings
import numpy as np
from spectral import *
import spectral as ssd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import scipy.spatial.distance as ssd
import matplotlib.patches as mpatches

# Importaciones para CNN
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# Configuración de advertencias
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ? Documenta Acción
# * Documenta función
# ~ Reducción de Acciones [Eliminar a futuro]
# ! Títulos de funciones y acciones

# ! ================= |/Funciones Generales/| =================
# ? Permite al usuario seleccionar un archivo de imagen
def cargar_imagen():
    # * Args:
    # *
    # * Returns:
    # * ruta_imagen: Ruta del archivo de imagen seleccionado

    Tk().withdraw()
    ruta_imagen = filedialog.askopenfilename(title="Selecciona una imagen HDR", filetypes=[("Archivos HDR", "*.hdr")])
    if not ruta_imagen:
        print("No se seleccionó ninguna imagen.")
        return None, None
    return ruta_imagen


# ? Aplica filtrado y normalización a la imagen
def preprocesar_imagen(ruta_imagen):
    # * Args:
    # * ruta_imagen: Ruta del archivo de imagen seleccionado
    # *
    # * Returns:
    # * arr_norm: Array normalizado de la imagen

    print("\n⏳ Preprocesando imagen...")

    try:
        img = open_image(ruta_imagen)
        arr = np.asarray(img.load())

        # ? Normalización por banda
        max_val = np.max(arr)
        min_val = np.min(arr)
        arr_norm = (arr - min_val) / (max_val - min_val)

        print(f"🟢 Preprocesamiento completado: {arr.shape[0]}x{arr.shape[1]} píxeles, {arr.shape[2]} bandas")
        return arr_norm

    except Exception as e:
        print(f"❌ Error al cargar la imagen: {str(e)}")
        return None


# ? Permite al usuario seleccionar el metodo de análisis por consola
def seleccionar_metodo():
    # * Args:
    # *
    # * Returns:
    # * El metodo seleccionado

    print("\n=== SELECCIÓN DE MÉTODO DE ANÁLISIS ===")
    print("1. Método de Correlación."
          "\n2. Método de Distancia Euclidiana."
          "\n3. Red Neuronal Convolucional (CNN)."
          )

    opcion = input("\nIngrese el número del método (1-3): ").strip()

    if opcion == "2":
        return "euclidean"
    elif opcion == "3":
        print("\nNota: El método CNN está en fase experimental. Algunos resultados pueden ser limitados.")
        return "cnn"
    else:
        return "correlation"


# ! ================= |/Funciones de CNN/| =================
def cargar_modelo(ruta_cnnm_vd, ruta_cnnm_refo):
    try:
        print(f"\n⏳ Cargando modelos...")
        cnn_vd = keras.models.load_model(ruta_cnnm_vd)
        cnn_refo = keras.models.load_model(ruta_cnnm_refo)
        print("✅ Modelos cargados.")
    except Exception as e:
        cnn_vd = None
        cnn_refo = None

    return cnn_vd, cnn_refo


# ? Prepara los datos para la CNN

def analizar_imagen(arr_norm, cnn_vd, cnn_refo, ruta_imagen, ruta_base):
    """
    Analiza cada píxel de la imagen usando las CNNs de forma optimizada usando predicción en batches.
    """

    print("\n⏳ Analizando imagen con modelos (optimizado)...")

    nombre_archivo = os.path.basename(ruta_imagen).split('.')[0]
    size_img_x, size_img_y, num_bandas = arr_norm.shape
    batch_size = 16

    non_zero_indices = np.nonzero(np.any(arr_norm != 0, axis=2))
    valid_pixel_indices = list(zip(non_zero_indices[0], non_zero_indices[1]))
    valid_pixel_data = arr_norm[non_zero_indices]

    valid_pixel_data_cnn = valid_pixel_data.reshape(-1, num_bandas, 1)

    print(f"⏳ Prediciendo con cnn_vd para {len(valid_pixel_data)} píxeles válidos...")
    pred_vd_all = cnn_vd.predict(valid_pixel_data_cnn, batch_size=batch_size, verbose=1)
    print("✅ Predicción con cnn_vd completada.")

    print(f"⏳ Prediciendo con cnn_refo para {len(valid_pixel_data)} píxeles válidos...")
    pred_refo_all = cnn_refo.predict(valid_pixel_data_cnn, batch_size=batch_size, verbose=1)
    print("✅ Predicción con cnn_refo completada.")

    resultados_vd = np.zeros((size_img_x, size_img_y, 3))  # 3 clases para vd
    resultados_refo = np.zeros((size_img_x, size_img_y, 3))  # 3 clases para refo

    for i, (x, y) in enumerate(valid_pixel_indices):
        resultados_vd[x, y, :] = pred_vd_all[i]
        resultados_refo[x, y, :] = pred_refo_all[i]

    print("✅ Análisis de imagen completado. Resultados son probabilidades por clase.")

    """Visualiza los resultados de la CNN como "mapas de similitud"."""
    print("\n⏳ Procesando resultados de la CNN para visualización de similitud...")

    size_img_x, size_img_y, _ = arr_norm.shape

    mapa_similitud_defo = resultados_vd[:, :, 0]  # Probabilidad de ser Deforestación
    mapa_similitud_veg = resultados_vd[:, :, 2]  # Probabilidad de ser Vegetación

    # Convertir a RGB para visualización
    arr_rgb = np.copy(arr_norm[:, :, [29, 19, 2]])

    # Visualización del mapa de similitud de Vegetación
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.imshow(arr_rgb)  # Mostrar la imagen original como fondo
    im1 = ax1.imshow(mapa_similitud_veg, cmap='viridis', alpha=0.6)  # Mostrar el mapa de similitud
    fig1.colorbar(im1, label='Probabilidad de Vegetación (Similitud)')
    ax1.set_title('Mapa de Similitud con Vegetación (CNN)')
    ruta_salida = os.path.join(ruta_base, "img_processed", f"{nombre_archivo}_analisis_vd_cnn.png")
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    fig1.savefig(ruta_salida)
    plt.show()

    # Visualización del mapa de similitud de Deforestación
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.imshow(arr_rgb)  # Mostrar la imagen original como fondo
    im2 = ax2.imshow(mapa_similitud_defo, cmap='plasma', alpha=0.6)  # Mostrar el mapa de similitud
    fig2.colorbar(im2, label='Probabilidad de Deforestación (Similitud)')
    ax2.set_title('Mapa de Similitud con Deforestación (CNN)')
    ruta_salida = os.path.join(ruta_base, "img_processed", f"{nombre_archivo}_analisis_rf_cnn.png")
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    fig2.savefig(ruta_salida)
    plt.show()


# *******************************************************************************+

# ! ================= |/Funciones de Analisis/| =================

# ? Calcula umbrales automáticos basados en percentiles de distancias
def calcular_umbrales_auto(arr_norm, pix_prom_veg, pix_prom_defo, metodo="correlation"):
    # * Args:
    # * arr_norm: Array normalizado de la imagen
    # * pix_prom_veg: Firma espectral del pixel promedio de vegetación
    # * pix_prom_defo: Firma espectral del pixel promedio de deforestación
    # * metodo: Selección del tipo de metodo de distancia (euclidean o correlation)
    # *
    # * Returns:
    # * umbral_veg: El umbral desde el cual se catalogara como un pixel de tipo vegetación
    # * umbral_defo: El umbral desde el cual se catalogara como un pixel de tipo deforestación

    umbral_defo = None
    umbral_veg = None

    # * Metodo de distancia euclidiana
    if metodo == "euclidean":

        euclidean_defo = []
        euclidean_veg = []
        size_img_x, size_img_y, _ = arr_norm.shape
        for i in range(size_img_x):
            for j in range(size_img_y):
                arr_temp = arr_norm[i, j, :]
                if np.count_nonzero(arr_temp) != 0:
                    dist_veg = np.linalg.norm(arr_temp - pix_prom_veg)
                    dist_defo = np.linalg.norm(arr_temp - pix_prom_defo)

                    max_dist_posible = dist_veg + dist_defo

                    eucl_defo = ssd.euclidean(pix_prom_defo, arr_temp)
                    eucl_veg = ssd.euclidean(pix_prom_veg, arr_temp)

                    euclidean_defo.append(np.abs(1 - np.abs(eucl_defo / max_dist_posible)) * 100)
                    euclidean_veg.append(np.abs(1 - np.abs(eucl_veg / max_dist_posible)) * 100)

        if euclidean_defo:
            umbral_defo = max(euclidean_defo) * 0.85
        else:
            umbral_defo = 85.0

        if euclidean_defo:
            umbral_veg = max(euclidean_veg) * 0.85
        else:
            umbral_veg = 85.0

    # * Metodo de la Correlacion [Por Defecto]
    else:
        correlaciones_defo = []
        correlaciones_veg = []
        size_img_x, size_img_y, _ = arr_norm.shape
        for i in range(size_img_x):
            for j in range(size_img_y):
                arr_temp = arr_norm[i, j, :]
                if np.count_nonzero(arr_temp) != 0:
                    corr_defo = ssd.correlation(pix_prom_defo, arr_temp)
                    corr_veg = ssd.correlation(pix_prom_veg, arr_temp)
                    correlaciones_defo.append(np.abs(1 - np.abs(corr_defo)) * 100)
                    correlaciones_veg.append(np.abs(1 - np.abs(corr_veg)) * 100)

        if correlaciones_defo:
            umbral_defo = max(correlaciones_defo) * 0.85
        else:
            umbral_defo = 85.0

        if correlaciones_veg:
            umbral_veg = max(correlaciones_veg) * 0.85
        else:
            umbral_veg = 85.0

    return umbral_defo, umbral_veg


# ? Detecta píxeles intermedios usando el metodo seleccionado
def detectar_pixeles_intermedios(arr_norm, pix_prom_defo, pix_prom_veg, umbral_defo, umbral_veg, num_categorias=5,
                                 metodo="correlation"):
    # * Args:
    # * arr_norm: Array normalizado de la imagen
    # * pix_prom_veg: Firma espectral del pixel promedio de vegetación
    # * pix_prom_defo: Firma espectral del pixel promedio de deforestación
    # * metodo: Selección del tipo de metodo de distancia (euclidean o correlation)
    # * num_categorias: Es la cantidad en la que se categoriza un rango de píxeles específicos.
    # * umbral_veg: El umbral desde el cual se catalogara como un pixel de tipo vegetación
    # * umbral_defo: El umbral desde el cual se catalogara como un pixel de tipo deforestación
    # *
    # * Returns:
    # * categorias: Es un array con los pixeles de cada categoria seleccionados

    print(f"\n⏳ Detectando píxeles intermedios usando método {metodo}...")

    size_img_x, size_img_y, _ = arr_norm.shape
    categorias = []

    for i in range(num_categorias):
        categorias.append([])

    # * Metodo de distancia euclidiana
    if metodo == "euclidean":
        for i in range(size_img_x):
            for j in range(size_img_y):
                arr_temp = arr_norm[i, j, :]
                if np.count_nonzero(arr_temp) != 0:

                    dist_veg = np.linalg.norm(pix_prom_veg - arr_temp)
                    dist_defo = np.linalg.norm(pix_prom_defo - arr_temp)

                    # dist_refo = np.linalg.norm(pix_prom_refo - arr_temp)
                    max_dist_posible = dist_veg + dist_defo

                    porc_veg = 100 * (1 - min(dist_veg / max_dist_posible, 1.0))
                    porc_defo = 100 * (1 - min(dist_defo / max_dist_posible, 1.0))

                    if porc_defo >= umbral_defo * 0.85 or porc_veg >= umbral_veg * 0.65:

                        if porc_veg >= umbral_veg * 0.70:
                            categoria = num_categorias - 1

                        elif porc_defo >= umbral_defo:
                            categoria = 0

                        else:
                            ratio = porc_veg / (porc_veg + porc_defo)
                            categoria = round(ratio * num_categorias)
                            # ~ print(categoria)

                        categorias[categoria].append((j, i))
    # * Metodo de correlación
    else:
        for i in range(size_img_x):
            for j in range(size_img_y):
                arr_temp = arr_norm[i, j, :]
                if np.count_nonzero(arr_temp) != 0:

                    corr_defo = ssd.correlation(pix_prom_defo, arr_temp)
                    corr_veg = ssd.correlation(pix_prom_veg, arr_temp)

                    porc_defo = np.abs(1 - np.abs(corr_defo)) * 100
                    porc_veg = np.abs(1 - np.abs(corr_veg)) * 100

                    # Ajuste de umbrales para mayor rigurosidad
                    if porc_defo > umbral_defo * 0.80 and porc_veg > umbral_veg * 0.80:
                        if porc_veg >= umbral_veg:
                            categoria = num_categorias - 1

                        elif porc_defo >= umbral_defo:
                            categoria = 0

                        else:
                            ratio = porc_veg / (porc_veg + porc_defo)
                            categoria = int(ratio * (num_categorias))
                        categorias[categoria].append((j, i))

    print(
        f"🟢 Detección completada: {sum(len(cat) for cat in categorias)} pixeles intermedios de deforestación")
    return categorias


# ? Detecta píxeles con potencial de reforestación usando el metodo seleccionado
def detectar_suelos_reforestables(arr_norm, pix_prom_defo, pix_prom_veg, umbral_defo, umbral_veg, num_categorias=5,
                                  metodo="correlation"):
    # * Args:
    # * arr_norm: Array normalizado de la imagen
    # * pix_prom_veg: Firma espectral del pixel promedio de vegetación
    # * pix_prom_defo: Firma espectral del pixel promedio de deforestación
    # * metodo: Selección del tipo de metodo de distancia (euclidean o correlation)
    # * num_categorias: Es la cantidad en la que se categoriza un rango de píxeles específicos.
    # *
    # * Returns:
    # * categorias_reforestables: Es un array con los pixeles de cada categoria seleccionados

    print(f"\n⏳ Detectando suelos reforestables usando método {metodo}...")

    size_img_x, size_img_y, _ = arr_norm.shape
    categorias_reforestables = []
    for i in range(num_categorias):
        categorias_reforestables.append([])

    # * Metodo de distancia euclidiana
    if metodo == "euclidean":
        for i in range(size_img_x):
            for j in range(size_img_y):
                arr_temp = arr_norm[i, j, :]
                if np.count_nonzero(arr_temp) != 0:

                    dist_veg = np.linalg.norm(pix_prom_veg - arr_temp)
                    dist_defo = np.linalg.norm(pix_prom_defo - arr_temp)

                    # ~ dist_refo = np.linalg.norm(pix_prom_refo - arr_temp)
                    max_dist_posible = dist_veg + dist_defo

                    porc_veg = 100 * (1 - min(dist_veg / max_dist_posible, 1.0))
                    porc_defo = 100 * (1 - min(dist_defo / max_dist_posible, 1.0))

                    if porc_defo >= umbral_defo * 0.80 and umbral_veg * 0.20 <= porc_veg <= umbral_veg:

                        if porc_veg >= umbral_veg * 0.43:
                            categoria = num_categorias - 1

                        elif porc_defo >= umbral_defo:
                            categoria = 0

                        else:
                            ratio = porc_veg / (porc_defo * 0.8)
                            categoria = round(ratio * (num_categorias - 1))
                        categorias_reforestables[categoria].append((j, i))

    # * Metodo de correlación
    else:
        for i in range(size_img_x):
            for j in range(size_img_y):
                arr_temp = arr_norm[i, j, :]
                if np.count_nonzero(arr_temp) != 0:
                    corr_defo = ssd.correlation(pix_prom_defo, arr_temp)
                    corr_veg = ssd.correlation(pix_prom_veg, arr_temp)
                    porc_defo = np.abs(1 - np.abs(corr_defo)) * 100
                    porc_veg = np.abs(1 - np.abs(corr_veg)) * 100

                    if porc_defo > umbral_defo * 0.6 and porc_veg > umbral_veg * 0.3 and porc_veg < umbral_veg * 0.6:
                        potential_ratio = porc_veg / (porc_defo * 0.8)
                        potential_ratio = min(max(potential_ratio, 0), 1)
                        categoria = int(potential_ratio * (num_categorias - 1))

                        categorias_reforestables[categoria].append((j, i))

    print(
        f"🟢 Detección completada: {sum(len(cat) for cat in categorias_reforestables)} suelos reforestables clasificados")
    return categorias_reforestables


# ! ================= |/Funciones de Visualización/| =================

# ? Aplica colores a las categorías clasificadas con ajustes específicos según el metodo
def aplicar_colores(arr_norm, categorias):
    # * Args:
    # * arr_norm: Array normalizado de la imagen
    # * categorias: Es un array con los pixeles de cada categoria seleccionados
    # *
    # * Returns:
    # * arr_copia: Array normalizado de la imagen con los colores aplicados
    # * mascara_alpha: Matriz de transparencia con los colores aplicados
    # * contadores: Array de la cantidad de píxeles detectados

    print(f"\n⏳ Aplicando colores a la clasificación...")
    try:
        size_img_x, size_img_y, _ = arr_norm.shape
        arr_copia = np.copy(arr_norm)
        num_categorias = len(categorias)

        # * Inicializar máscara y contadores
        mascara_alpha = np.zeros((size_img_x, size_img_y, 4))
        contadores = [0] * num_categorias

        # * Generar esquemas de color según el metodo
        colores = []
        for i in range(num_categorias):
            # Esquema para metodo de correlación (naranja-verde)
            r = 1.0 - (i / (num_categorias - 1))
            g = 1.0
            b = 0.0

            alpha = 0.4
            colores.append([r, g, b, alpha])

        # * Aplicar colores con validación de coordenadas
        for cat_idx, categoria in enumerate(categorias):
            for x, y in categoria:
                try:
                    if 0 <= y < size_img_y or 0 <= x < size_img_x:
                        # * Ajustar la transparencia según la posición en la categoría
                        color = colores[cat_idx].copy()

                        # * Aumentar opacidad para categorías extremas (primera y última)
                        if cat_idx == 0 or cat_idx == num_categorias - 1:
                            color[3] *= 1.2  # * 20% más opaco para categorías extremas

                        mascara_alpha[y, x] = color
                        contadores[cat_idx] += 1
                except Exception as e:
                    print(f"⚠️ Error al procesar coordenadas ({x},{y}): {str(e)}")
                    continue

        # ~ Normalizar la máscara alpha para evitar saturación
        # ~ if np.max(mascara_alpha) > 1:
        # ~    mascara_alpha /= np.max(mascara_alpha)

        # * Aplicar suavizado a la máscara para mejorar visualización
        from scipy.ndimage import gaussian_filter
        mascara_alpha = gaussian_filter(mascara_alpha, sigma=0.5)

        print("🟢 Colores aplicados correctamente")
        return arr_copia, mascara_alpha, contadores

    except Exception as e:
        print(f"❌ Error al aplicar colores: {str(e)}")
        return arr_norm.copy(), np.zeros((arr_norm.shape[0], arr_norm.shape[1], 4)), [0] * len(categorias)


# ? Aplica colores a las categorías clasificadas
def aplicar_colores_reforestable(arr_norm, categorias_reforestables):
    # * Args:
    # * arr_norm: Array normalizado de la imagen
    # * categorias: Es un array con los pixeles de cada categoria seleccionados
    # *
    # * Returns:
    # * mascara_alpha: Matriz de transparencia con los colores aplicados
    # * contadores: Array de la cantidad de pixeles detectados

    print("\n⏳ Aplicando colores a la clasificación...")
    try:
        size_img_x, size_img_y, _ = arr_norm.shape
        mascara_alpha = np.zeros((size_img_x, size_img_y, 4))
        contadores = [0] * len(categorias_reforestables)

        colores = []
        num_categorias = len(categorias_reforestables)
        for i in range(len(categorias_reforestables)):
            r = 1.0
            g = (i / (len(categorias_reforestables) - 1)) if num_categorias > 1 else 0.5
            b = 0.0
            colores.append([r, g, b, 0.4])

        for cat_idx, categoria in enumerate(categorias_reforestables):
            for x, y in categoria:
                try:
                    if 0 <= y < size_img_x or 0 <= x < size_img_y:
                        mascara_alpha[y, x] = colores[cat_idx]
                        contadores[cat_idx] += 1
                except Exception as e:
                    print(f"⚠️ Error al procesar coordenadas ({x},{y}): {str(e)}")

        return mascara_alpha, contadores

    except Exception as e:
        print(f"❌ Error al aplicar colores: {str(e)}")
        return np.zeros((arr_norm.shape[0], arr_norm.shape[1], 4)), [0] * len(categorias_reforestables)


# ?
def plot_firmas_spectrales(pix_prom_defo, pix_prom_veg, pix_prom_refo):
    # * Variables:
    # * pix_prom_veg: Firma espectral del pixel promedio de vegetación
    # * pix_prom_defo: Firma espectral del pixel promedio de deforestación
    # * pix_prom_refo: Firma espectral del pixel promedio de reforestación
    # * contadores_veg: Es el numero de pixeles detectados para vegetación
    # * contadores_defo: Es el numero de pixeles detectados para deforestación
    # * contadores_refo: Es el numero de pixeles detectados para reforestación

    plt.figure(figsize=(15, 5))

    # * Graficar espectros promedio
    plt.plot(pix_prom_defo, color='orange', linewidth=2, linestyle='--', label=f'Promedio Deforestación')
    plt.plot(pix_prom_veg, color='green', linewidth=2, linestyle='-.', label=f'Promedio Vegetación')
    plt.plot(pix_prom_refo, color='red', linewidth=2, linestyle=':', label=f'Promedio Reforestables')

    plt.title('Espectros Promedio')
    plt.xlabel('Banda espectral')
    plt.ylabel('Reflectancia normalizada')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3)  # Leyenda debajo del gráfico
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ! ================= |/Funciones de Sistemas Informacion Geografico/| =================


# ! ================= |/Funciones de Sistemas/| =================
def so():
    system = sys.platform
    if system == 'linux':
        print(f"🐧 Sistema operativo Linux")
    elif system == 'win32' or system == 'cygwin' or system == 'msys' or system == 'win64' or system == 'x64':
        print(f"🪟 Sistema operativo Windows")
    elif system == 'darwin' or system == 'macos' or system == 'os2' or system == 'os2emx' or system == 'riscos' or system == 'atheos':
        print(f"🍎Sistema operativo MacOS")


def cgpus():
    # * Returns:
    # * str: Dispositivo a usar ('GPU' o 'CPU')

    # Verificar GPUs disponibles usando TensorFlow
    gpus_tf = tf.config.list_physical_devices('GPU')
    gpu_available_tf = len(gpus_tf) > 0

    # Verificar GPU disponible usando PyTorch
    gpu_available_tp = torch.cuda.is_available()

    if gpu_available_tf or gpu_available_tp:

        try:
            gpu_tipo = 'NVIDIA' if gpu_available_tf else 'AMD/Metal/otros'

            if gpu_available_tf:
                gpu_name = tf.test.gpu_device_name()
            else:
                gpu_name = 'N/A'

            try:
                # Configurar memoria de crecimiento para evitar problemas de OOM
                for gpu in gpus_tf:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(f"{len(gpus_tf)} GPU(s) física(s), {len(logical_gpus)} GPU(s) lógica(s)")
                return 'GPU'
            except RuntimeError as e:
                print(f"Error configurando GPU: {e}")

            # Seleccionar la primera GPU disponible
            print(f"🚀 GPU detectada: {gpu_tipo} (Tipo: {gpu_name})")

            # Configurar para usar la GPU
            tf.config.set_visible_devices(gpus_tf[0], 'GPU')
            return 'GPU'

        except RuntimeError as e:
            # Intentar configuración alternativa para GPUs no NVIDIA
            try:
                tf.config.set_visible_devices(gpus_tf[0], 'GPU')
                print(f"✅ Configuración alternativa exitosa para GPU: {gpus_tf[0].name}")
                return 'GPU'
            except Exception as alt_e:
                print(f"⚠️ Error en configuración alternativa: {alt_e}")
                print("🔄 Usando CPU como respaldo.")
                return 'CPU'
    else:
        print("🖥️ No se detectaron GPUs disponibles. Usando CPU.")
        return 'CPU'


# ! ================= |/Funcione de Software/| =================

# ? Función principal que orquesta el análisis de la imagen hiperespectral
def main():
    print("\n===== ANÁLISIS DE IMÁGENES HIPERESPECTRALES =====")
    print("Versión: 0.7.6 - Implementación de CNN")
    print("Desarrollado por: Carlos Javier Amaranto Mercado - 0221920008\n")
    so()
    dispositivo = cgpus()

    try:
        archivos_existen = False
        cnnm_existen = False
        ruta_base = os.getcwd()

        # * Buscar pixeles promedio .npy / .h5 / .keras
        posibles_ubicaciones = [
            os.path.join(ruta_base, "database", "fsp"),
            os.path.join(os.path.dirname(ruta_base), "src", "database", "fsp")
        ]

        posibles_ubicaciones_cnnm = [
            os.path.join(ruta_base, "database", "cnnm"),
            os.path.join(os.path.dirname(ruta_base), "src", "database", "cnnm")
        ]

        ruta_cnnm_vd = None
        ruta_cnnm_refo = None

        print(f"\n{posibles_ubicaciones_cnnm}")

        for base_dir_cnnm in posibles_ubicaciones_cnnm:
            try:
                ruta_cnnm_vd = os.path.join(base_dir_cnnm, "cnn_vd.keras")
                ruta_cnnm_refo = os.path.join(base_dir_cnnm, "cnn_refo.keras")
                if os.path.exists(ruta_cnnm_vd) and os.path.exists(ruta_cnnm_refo):
                    print(f"🟢 Archivos de CNN encontrados en: {base_dir_cnnm}")
                    cnnm_existen = True
                    break
                else:
                    ruta_cnnm_vd = os.path.join(base_dir_cnnm, "cnn_vd.h5")
                    ruta_cnnm_refo = os.path.join(base_dir_cnnm, "cnn_refo.h5")
                    if os.path.exists(ruta_cnnm_vd) and os.path.exists(ruta_cnnm_refo):
                        print(f"🟢 Archivos de CNN encontrados en: {base_dir_cnnm}")
                        cnnm_existen = True
                        break
            except Exception as e:
                print(f"❌ Error al buscar archivos de CNN: {str(e)}")
                continue

        cnnm_vd, cnnm_refo = cargar_modelo(ruta_cnnm_vd, ruta_cnnm_refo)

        if (cnnm_vd != None) and (cnnm_refo != None):
            print("\n❌ ERROR: No se encontraron los archivos de los modelos de referencia.")
            print("Por favor, verifique que existan los siguientes archivos:")
            print("- cnn_vd.keras/.h5 (Vegetación/Deforestación)")
            print("- cnn_refo.keras/.h5 (Promedio Reforestables)")
            cnnm_existen = False
            cnnm_vd = None
            cnnm_refo = None

        else:
            print(f"🟢 Archivos de CNN cargados correctamente")

        for base_dir_img in posibles_ubicaciones:
            try:
                base_dir_img = os.path.abspath(base_dir_img)
                ruta_veg = os.path.join(base_dir_img, "fsp_veg.npy")
                ruta_defo = os.path.join(base_dir_img, "fsp_defo.npy")
                ruta_refo = os.path.join(base_dir_img, "fsp_refo.npy")

                if all(os.path.exists(ruta) for ruta in [ruta_veg, ruta_defo, ruta_refo]):
                    print(f"🟢 Archivos de firmas espectrales encontrados en: {base_dir_img}")
                    archivos_existen = True
                    break
            except Exception:
                continue

        # * Buscar archivos de referencia
        print("\n⏳ Buscando archivos de firmas espectrales de referencia...")
        if not archivos_existen:
            print("\n❌ ERROR: No se encontraron los archivos de píxeles de referencia.")
            print("Por favor, verifique que existan los siguientes archivos:")
            print("- fsp_veg.npy (vegetación)")
            print("- fsp_defo.npy (deforestación)")
            print("- fsp_refo.npy (reforestables)")
            return

        # * Cargar firmas espectrales desde archivos .npy
        try:
            pix_prom_veg = np.load(ruta_veg)
            pix_prom_defo = np.load(ruta_defo)
            pix_prom_refo = np.load(ruta_refo)
            print("🟢 Firmas espectrales cargadas correctamente")
        except Exception as e:
            print(f"❌ Error al cargar firmas espectrales: {str(e)}")
            return

        while True:
            # * Cargar imagen
            print("\n⏳ Seleccionando imagen hiperespectral...")
            ruta_imagen = cargar_imagen()
            if not ruta_imagen:
                return  # Salir si no se selecciona imagen
            print(f"🟢 Imagen seleccionada en la ruta {ruta_imagen}")

            # * Preprocesar imagen
            arr_norm = preprocesar_imagen(ruta_imagen)
            if arr_norm is None:
                return

            # * Selección del metodo
            metodo = seleccionar_metodo()
            print(f"\n🟢 Método seleccionado: {metodo.upper()}")

            # * Cargar o crear modelo CNN si es necesario

            if metodo == "cnn":
                try:
                    if cnnm_existen:
                        # Analizar la imagen
                        analizar_imagen(arr_norm, cnnm_vd, cnnm_refo, ruta_imagen, ruta_base)
                        print("🟢 Análisis de imagen completado. Resultados son probabilidades por clase.")

                    else:
                        print("❌ ERROR: No se encontraron los archivos de CNN.")
                        print("Por favor, verifique que existan los siguientes archivos:")
                        print("- cnn_vd.keras/.h5 (Vegetación/Deforestación)")
                        print("- cnn_refo.keras/.h5 (Promedio Reforestables)")
                        return

                except Exception as e:
                    print(f"❌ Error al manejar modelo CNN: {str(e)}")
                    print("⚠️ Continuando con método de correlación como fallback")
                    metodo = "correlation"

            if metodo != 'cnn':
                # * Calcular umbrales (no se usan para CNN pero se mantienen para consistencia)
                print("\n⏳ Calculando umbrales de clasificación...")
                umbral_defo, umbral_veg = calcular_umbrales_auto(arr_norm, pix_prom_veg, pix_prom_defo, metodo)
                print(f"🟢 Umbrales calculados: Deforestación={umbral_defo:.2f}, Vegetación={umbral_veg:.2f}")

                # * Detectar píxeles intermedios
                categorias = detectar_pixeles_intermedios(arr_norm, pix_prom_defo, pix_prom_veg, umbral_defo,
                                                          umbral_veg,
                                                          metodo=metodo)  # * Se usa modelo V/D

                # * Detectar suelos reforestables
                categorias_reforestables = detectar_suelos_reforestables(arr_norm, pix_prom_defo, pix_prom_veg,
                                                                         umbral_defo,
                                                                         umbral_veg,
                                                                         metodo=metodo)  # * Se usa modelo P_R

                # * Aplicar colores
                arr_coloreado, mascara_color, contadores = aplicar_colores(arr_norm, categorias)
                mascara_reforestable, contadores_reforestables = aplicar_colores_reforestable(arr_norm,
                                                                                              categorias_reforestables)
                total_reforestables = sum(contadores_reforestables)

                # * Procesamiento geográfico
                print("\n⏳ Procesando resultados con información geográfica...")

                # * Crear imagen RGB para visualización (bandas 29, 19, 2 como R,G,B)
                arr_rgb = np.copy(arr_norm[:, :, [29, 19, 2]])

                # * Visualizar resultados
                try:
                    # * Visualización 1: Vegetación y Deforestación
                    fig1, ax1 = plt.subplots(figsize=(10, 8))
                    ax1.imshow(arr_rgb)
                    ax1.imshow(mascara_color)

                    handles = []
                    for i in range(len(categorias)):
                        r = 1.0 - (i / (len(categorias) - 1)) if len(categorias) > 1 else 0.5
                        g = 1.0
                        b = 0.0
                        if i == 0:
                            etiqueta = f"Defo ({contadores[i]})"
                        elif i == len(categorias) - 1:
                            etiqueta = f"Veg ({contadores[i]})"
                        else:
                            etiqueta = f"Int {i} ({contadores[i]})"
                        handles.append(mpatches.Patch(color=[r, g, b], label=etiqueta, alpha=0.4))

                    ax1.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=len(handles))
                    ax1.set_title('Análisis de Vegetación y Deforestación con Estados Intermedios', fontsize=16,
                                  fontweight='bold')

                    nombre_archivo = os.path.basename(ruta_imagen).split('.')[0]
                    ruta_salida1 = os.path.join(ruta_base, "img_processed", f"{nombre_archivo}_analisis_intermedio.png")
                    os.makedirs(os.path.dirname(ruta_salida1), exist_ok=True)
                    fig1.savefig(ruta_salida1)
                    plt.show()

                    # * Visualización 2: Suelos Reforestables
                    fig2, ax2 = plt.subplots(figsize=(10, 8))
                    ax2.imshow(arr_rgb)
                    ax2.imshow(mascara_reforestable)

                    handles_reforestables = []
                    for i in range(len(categorias_reforestables)):
                        r = 1.0
                        g = (i / (len(categorias_reforestables) - 1)) if len(categorias_reforestables) > 1 else 0.5
                        b = 0.0

                        etiqueta = f"Porc {i * 25} % ({contadores_reforestables[i]})"
                        handles_reforestables.append(mpatches.Patch(color=[r, g, b], label=etiqueta, alpha=0.4))

                    ax2.legend(handles=handles_reforestables, loc='lower center', bbox_to_anchor=(0.5, -0.15),
                               ncol=len(handles_reforestables))
                    ax2.set_title(f'Análisis de Suelos con Potencial de Reforestación ({total_reforestables} píxeles)',
                                  fontsize=16, fontweight='bold')

                    ruta_salida2 = os.path.join(ruta_base, "img_processed",
                                                f"{nombre_archivo}_analisis_reforestable.png")
                    fig2.savefig(ruta_salida2)
                    plt.show()

                    print("\n=== RESULTADOS DEL ANÁLISIS ===")
                    print(f"Imágenes guardadas en:")
                    print(f"- {ruta_salida1}")
                    print(f"- {ruta_salida2}")
                except Exception as e:
                    print(f"❌ Error al generar las imagenes: {str(e)}")
                    return

                # * Imprimir estadísticas
                print("\n===== ESTADÍSTICAS =====")
                print(f"\n📊 Vegetacion ➡ Reforestacion")
                for i, contador in enumerate(contadores):
                    categoria_nombre = "Deforestación" if i == 0 else (
                        "Vegetación" if i == len(contadores) - 1 else f"Intermedio {i}")
                    print(f"Píxeles de {categoria_nombre}: {contador}")

                print(f"\n📊 Potencial de Reforestacion")
                for i, contador in enumerate(contadores_reforestables):
                    potencial = "Bajo" if i == 0 else (
                        "Alto" if i == len(categorias_reforestables) - 1 else f"Medio {i}")
                    print(f"Potencial {potencial}: {contador} píxeles")
                print(f"Total de píxeles reforestables: {total_reforestables}")

                plot_firmas_spectrales(pix_prom_defo, pix_prom_veg, pix_prom_refo)

            print("\n✅ Análisis completo")

            # * Presentamos al usuario una opción para finalizar el programa
            opcion = input("¿Quiere analizar otra imagen? (S/N): ").strip().lower()

            if opcion == 'n':
                print("Cerrando el programa... ")
                break

    except Exception as e:
        print(f"\n❌ ERROR FATAL: {str(e)}")
        print("El programa se cerrará. Por favor, revise el error y vuelva a intentarlo.")


if __name__ == "__main__":
    main()