import os
import warnings
import numpy as np
from spectral import *
import sys

# Importaciones para CNN
import torch
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Configuración de advertencias
warnings.filterwarnings('ignore', category=RuntimeWarning)

def detectar_so():
    system = sys.platform
    if system == 'linux':
        print(f"🐧 Sistema operativo Linux")
    elif system == 'win32' or system == 'cygwin' or system == 'msys' or system == 'win64' or system == 'x64':
        print(f"🪟 Sistema operativo Windows")
    elif system == 'darwin' or system == 'macos' or system == 'os2' or system == 'os2emx' or system == 'riscos' or system == 'atheos':
        print(f"🍎Sistema operativo MacOS")

# ? Detecta y configura dispositivos GPU disponibles de cualquier tipo
def detectar_y_configurar_gpu():
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
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"🚀 GPU detectada: {gpu_tipo} (Tipo: {gpu_name})")

            # Configurar para usar la GPU
            tf.config.set_visible_devices(gpus_tf[0], 'GPU')
            return 'GPU'

        except RuntimeError as e:
            print(f"⚠️ Error configurando GPU: {e}")
            print("🔁 Intentando configuración alternativa...")

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

def preparar_datos_para_dispositivo(dispositivo):
    """
    Prepara los datos para el dispositivo seleccionado.

    Args:
        dispositivo (str): 'GPU' o 'CPU'

    Returns:
        Configuración específica para el dispositivo
    """
    if dispositivo == 'GPU':
        # Configuración optimizada para cualquier GPU
        try:
            # Configuración para NVIDIA CUDA
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.9  # Usar hasta 90% de la memoria

            # Configuración para AMD/Metal/otros
            try:
                tf.config.optimizer.set_jit(True)  # Habilitar XLA para cualquier GPU
                tf.config.threading.set_inter_op_parallelism_threads(4)
                tf.config.threading.set_intra_op_parallelism_threads(4)
            except:
                pass

            return tf.keras.backend.set_session(tf.compat.v1.Session(config=config))
        except:
            # Configuración genérica si falla la específica
            return None
    else:
        # Configuraciones optimizadas para CPU
        tf.config.threading.set_inter_op_parallelism_threads(8)
        tf.config.threading.set_intra_op_parallelism_threads(8)
        return None

# ? Prepara los datos para la CNN
def preparar_datos_cnn(arr_norm, pix_prom_defo, pix_prom_veg):
    print("\n⏳ Preparando datos para CNN (versión vectorizada)...")

    # Obtener dimensiones
    size_img_x, size_img_y, num_bandas = arr_norm.shape

    # Reshape para procesamiento vectorizado
    pixels = arr_norm.reshape(-1, num_bandas)
    mask = np.any(pixels != 0, axis=1)
    pixels = pixels[mask]

    # Calcular distancias vectorizadas
    dist_defo = np.linalg.norm(pixels - pix_prom_defo, axis=1)
    dist_veg = np.linalg.norm(pixels - pix_prom_veg, axis=1)

    # Calcular probabilidades
    total_dist = dist_defo + dist_veg
    prob_defo = np.divide(dist_defo, total_dist, out=np.full_like(dist_defo, 0.5), where=total_dist != 0)
    prob_veg = np.divide(dist_veg, total_dist, out=np.full_like(dist_veg, 0.5), where=total_dist != 0)

    # Asignar etiquetas
    y = np.select(
        [prob_defo < 0.3, prob_veg < 0.3],
        [0, 2],
        default=1
    )

    # Codificar etiquetas
    y = to_categorical(y)

    print(f"🟢 Datos preparados: {pixels.shape[0]} muestras, {num_bandas} bandas")
    return pixels, y

# ? Entrena el modelo CNN
def entrenar_modelo_cnn(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=8):
    # * Args:
    # * model: Modelo CNN a entrenar
    # * X_train, y_train: Datos de entrenamiento
    # * X_val, y_val: Datos de validación
    # * epochs: Número de épocas de entrenamiento
    # * batch_size: Tamaño del lote
    # *
    # * Returns:
    # * model: Modelo entrenado
    # * history: Historial de entrenamiento

    print("\n⏳ Entrenando modelo CNN...")

    # Callbacks para mejorar el entrenamiento
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    print("🟢 Modelo CNN entrenado")
    return model, history

# ? Guarda el modelo CNN entrenado en formato HDF5 (.h5)
def guardar_modelo_cnn(modelo, ruta_modelo):
    # * Guarda un modelo CNN
    # * Args:
    # * modelo: Modelo de Keras a guardar
    # * ruta_modelo: Ruta completa donde se guardará el modelo
    # *
    # * Returns:
    # * bool: True si se guardó correctamente, False si hubo error

    print(f"\n⏳ Guardando modelo CNN en {ruta_modelo}...")

    try:
        # Crear directorio si no existe
        directorio = os.path.dirname(ruta_modelo)
        if not os.path.exists(directorio):
            os.makedirs(directorio)
            print(f"🟢 Directorio creado: {directorio}")

        # Guardar el modelo
        modelo.save(ruta_modelo)
        print("🟢 Modelo CNN guardado correctamente")
        return True
    except Exception as e:
        print(f"❌ Error al guardar el modelo: {str(e)}")
        return False

# ? Entrena el modelo CNN y lo guarda
def entrenar_y_guardar_modelo_cnn(model, X_train, y_train, X_val, y_val, ruta_guardado, epochs=20, batch_size=32):
    # * Args:
    # * model: Modelo CNN a entrenar
    # * X_train, y_train: Datos de entrenamiento
    # * X_val, y_val: Datos de validación
    # * ruta_guardado: Ruta donde se guardará el modelo
    # * epochs: Número de épocas de entrenamiento
    # * batch_size: Tamaño del lote
    # *
    # * Returns:
    # * model: Modelo entrenado
    # * history: Historial de entrenamiento
    # * bool: True si se guardó correctamente

    print("\n⏳ Entrenando y guardando modelo CNN...")

    # Entrenar el modelo
    model, history = entrenar_modelo_cnn(model, X_train, y_train, X_val, y_val, epochs, batch_size)

    # Guardar el modelo
    guardado_exitoso = guardar_modelo_cnn(model, ruta_guardado)

    return model, history, guardado_exitoso

# ? Carga o crea un modelo CNN según disponibilidad
def cargar_o_crear_modelo(ruta_modelo, input_shape=None, num_classes=None, datos_entrenamiento=None):

    # * Args:
    # * ruta_modelo: Ruta donde buscar/guardar el modelo
    # * input_shape: Forma de entrada para crear nuevo modelo (opcional si hay que crearlo)
    # * num_classes: Número de clases para crear nuevo modelo (opcional si hay que crearlo)
    # * datos_entrenamiento: Tupla (X, y) para entrenamiento si hay que crear modelo (opcional)
    # *
    # * Returns:
    # * model: Modelo cargado o nuevo modelo entrenado
    # * history: Historial de entrenamiento (None si se cargó modelo existente)

    # ? Crea un modelo CNN adaptable
    def crear_modelo_cnn(input_shape, num_classes=3, dispositivo='CPU'):
        # * Args:
        # * input_shape: Forma de los datos de entrada (número de bandas)
        # * num_classes: Número de clases de salida
        # *
        # * Returns:
        # * model: Modelo CNN compilado

        print("\n⏳ Creando modelo CNN...")

        model = Sequential()

        # Capa convolucional 1D para manejar las bandas espectrales
        model.add(Conv1D(filters=64, kernel_size=5, activation='relu',
                         input_shape=input_shape, padding='same'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.3))

        # Segunda capa convolucional
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.3))

        # Tercera capa convolucional
        model.add(Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.3))

        # Aplanar y capas densas
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        # Compilar el modelo
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print("🟢 Modelo CNN creado y compilado")
        return model

    try:
        #? Intentar cargar modelo existente
        if os.path.exists(ruta_modelo):
            print(f"🟢 Cargando modelo CNN existente de {ruta_modelo}")
            model = tf.keras.models.load_model(ruta_modelo)
            return model, None
    except Exception as e:
        print(f"⚠️ Error al cargar modelo existente: {str(e)}")

    #? Si no se pudo cargar, crear uno nuevo
    print("⚠️ No se encontró modelo preentrenado. Creando nuevo modelo...")

    if input_shape is None or num_classes is None or datos_entrenamiento is None:
        raise ValueError("Se necesitan input_shape, num_classes y datos_entrenamiento para crear nuevo modelo")

    X, y = datos_entrenamiento
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    model = crear_modelo_cnn(input_shape, num_classes)
    model, history, _ = entrenar_y_guardar_modelo_cnn(
        model, X_train, y_train, X_val, y_val, ruta_modelo
    )

    return model, history


def main():
    # Configurar dispositivo primero
    detectar_so()
    dispositivo = detectar_y_configurar_gpu()
    preparar_datos_para_dispositivo(dispositivo)

    # * Buscar archivos de referencia
    print("\n⏳ Buscando archivos de firmas espectrales de referencia...")
    archivos_existen = False
    ruta_base = os.getcwd()

    # * Buscar pixeles promedio .npy / .h5
    posibles_ubicaciones = [
        os.path.join(ruta_base, "database", "fsp"),
        os.path.join(os.path.dirname(ruta_base), "src", "database", "fsp"),
        os.path.join(os.path.dirname(ruta_base), "src", "database", "cnnm")
    ]

    for base_dir_img in posibles_ubicaciones:
        try:
            base_dir_img = os.path.abspath(base_dir_img)
            ruta_veg = "/home/carlos-amaranto/Escritorio/Workspace/project.ecospectral.edu.udc/src/database/fsp/fsp_veg.npy"
            ruta_defo = "/home/carlos-amaranto/Escritorio/Workspace/project.ecospectral.edu.udc/src/database/fsp/fsp_defo.npy"
            ruta_refo = "/home/carlos-amaranto/Escritorio/Workspace/project.ecospectral.edu.udc/src/database/fsp/fsp_refo.npy"
            ruta_cnn_modelo = "src/database/cnnm/cnnm.h5"

            if all(os.path.exists(ruta) for ruta in [ruta_veg, ruta_defo, ruta_refo]):
                print(f"🟢 Archivos de firmas espectrales encontrados en: {base_dir_img}")
                archivos_existen = True
                break
        except Exception:
            continue

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
        ruta_imagen = "/home/carlos-amaranto/Escritorio/Workspace/project.ecospectral.edu.udc/src/database/img/cartagena_test.hdr"

        # * Si se seleccionó un .hdr, buscamos el .img correspondiente
        ruta_img = None
        if ruta_imagen.lower().endswith('.hdr'):
            ruta_img = ruta_imagen[:-4] + '.img'  # Reemplaza .hdr por .img
            if os.path.exists(ruta_img):
                print(f"🟢 Archivo .img encontrado: {ruta_img}")
            else:
                print(f"❌ No se encontró el archivo .img correspondiente a {ruta_imagen}")

        if not ruta_imagen and ruta_img:
            return  # Salir si no se selecciona imagen
        print(f"🟢 Imagen seleccionada en la ruta {ruta_imagen}")

        # * Preprocesar imagen
        # Cargamos la imagen usando spectral
        img = open_image(ruta_imagen)
        arr = np.asarray(img.load())

        # Normalización por banda
        max_val = np.max(arr)
        min_val = np.min(arr)
        arr_norm = (arr - min_val) / (max_val - min_val)

        print(f"🟢 Preprocesamiento completado: {arr.shape[0]}x{arr.shape[1]} píxeles, {arr.shape[2]} bandas")

        # * Cargar o crear modelo CNN si es necesario
        ruta_cnn_modelo = os.path.join(os.path.dirname(ruta_base), "src", "database", "cnnm", "cnnm.h5")

        try:
            if (dispositivo == 'GPU'):
                with tf.device('/device:GPU:0'):
                    print("🟢 GPU seleccionada")
                    # Preparar datos para CNN
                    X, y = preparar_datos_cnn(arr_norm, pix_prom_defo, pix_prom_veg)
                    input_shape = (X.shape[1], 1)
                    num_classes = y.shape[1]

                    # Cargar o crear modelo
                    cnn_modelo, _ = cargar_o_crear_modelo(
                        ruta_modelo=ruta_cnn_modelo,
                        input_shape=input_shape,
                        num_classes=num_classes,
                        datos_entrenamiento=(X, y))
            elif (dispositivo == 'CPU'):
                print("🟢 CPU seleccionada")
                # Preparar datos para CNN
                X, y = preparar_datos_cnn(arr_norm, pix_prom_defo, pix_prom_veg)
                input_shape = (X.shape[1], 1)
                num_classes = y.shape[1]

                # Cargar o crear modelo
                cnn_modelo, _ = cargar_o_crear_modelo(
                    ruta_modelo=ruta_cnn_modelo,
                    input_shape=input_shape,
                    num_classes=num_classes,
                    datos_entrenamiento=(X, y))
        except Exception as e:
            print(f"❌ Error al manejar modelo CNN: {str(e)}")

if __name__ == "__main__":
    main()

