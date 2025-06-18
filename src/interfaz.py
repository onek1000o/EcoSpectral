# Importación de librerías necesarias
import matplotlib.pyplot as plt  # Para visualización
import matplotlib.patches as mpatches  # Para leyendas personalizadas en imágenes
import tkinter as tk  # Interfaz gráfica
from tkinter import filedialog, messagebox, ttk  # Widgets avanzados de Tkinter
import os  # Manejo de rutas y sistema de archivos
import numpy as np  # Operaciones numéricas
from PIL import Image, ImageTk  # Manipulación de imágenes
import sys  # Acceso a funcionalidades del sistema
import io  # Entrada/Salida de consola
import threading  # Para ejecutar procesos en segundo plano

# Clase para redirigir la salida estándar (print) a un widget de texto
class ConsoleRedirector(io.StringIO):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def write(self, s):
        self.text_widget.insert(tk.END, s)
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()

    def flush(self):
        pass

# Importación de funciones personalizadas desde main.py
from main import (
    preprocesar_imagen,
    analizar_imagen,
    cargar_modelo,
    calcular_umbrales_auto,
    detectar_pixeles_intermedios,
    detectar_suelos_reforestables,
    aplicar_colores,
    aplicar_colores_reforestable,
    plot_firmas_spectrales
)

# Ruta base dinámica del archivo actual
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Clase principal de la interfaz gráfica
class InterfazGrafica:
    def __init__(self, root):
        self.root = root
        self.root.title("Análisis de Imágenes Hiperespectrales")

        # Configuración de tamaño y posición inicial
        ancho_ventana, alto_ventana = 800, 600
        x = (self.root.winfo_screenwidth() // 2) - (ancho_ventana // 2)
        y = 15
        self.root.geometry(f"{ancho_ventana}x{alto_ventana}+{x}+{y}")
        self.root.protocol("WM_DELETE_WINDOW", self.salir)

        # Variables de estado
        self.ruta_imagen = None
        self.arr_norm = None
        self.metodo = tk.StringVar(value="Seleccionar")
        self.imagenes_resultado = []
        self.pagina_actual = 0

        # Referencias a botones
        self.btn_cargar = None
        self.btn_analizar = None
        self.metodo_menu = None
        self.btn_siguiente = None

        # Construcción de la interfaz
        self.crear_widgets()

    def crear_widgets(self):
        # Panel lateral de botones y logo
        frame_botones = tk.Frame(self.root, bg="#dfe6df")
        frame_botones.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)

        # Carga de logo desde carpeta Logo
        try:
            ruta_logo = os.path.join(BASE_DIR, "Logo", "logo.png")
            if os.path.exists(ruta_logo):
                logo_img = Image.open(ruta_logo).resize((240, 240))
                self.logo_tk = ImageTk.PhotoImage(logo_img)
                tk.Label(frame_botones, image=self.logo_tk, bg="#dfe6df").pack(pady=(10, 20))
            else:
                tk.Label(frame_botones, text="LOGO", font=("Arial", 16), bg="#dfe6df").pack(pady=(10, 20))
        except Exception as e:
            print(f"No se pudo cargar el logo: {e}")

        # Estilo de botones
        estilo_boton = {
            "font": ("Helvetica", 14, "bold"),
            "bg": "#4CAF50",
            "fg": "white",
            "activebackground": "#45a049",
            "relief": tk.FLAT,
            "bd": 0,
            "height": 2,
            "width": 20,
            "highlightthickness": 0
        }

        # Contenedor central de botones
        frame_centro = tk.Frame(frame_botones, bg="#dfe6df")
        frame_centro.pack(expand=True)

        # Botón de carga de imagen
        self.btn_cargar = tk.Button(frame_centro, text="Cargar Imagen", command=self.cargar_imagen, **estilo_boton)
        self.btn_cargar.pack(pady=15)

        # Menú de selección de método
        tk.Label(frame_centro, text="Seleccionar Método:", bg="#dfe6df", font=("Helvetica", 15, "bold")).pack(pady=(10, 5))
        self.metodo_menu = tk.OptionMenu(frame_centro, self.metodo, "Correlation", "Euclidean", "CNN")
        self.metodo_menu.config(**estilo_boton)
        self.metodo_menu.pack(pady=(0, 15))

        # Botón para análisis
        self.btn_analizar = tk.Button(frame_centro, text="Ejecutar Análisis", command=self.ejecutar_analisis_en_hilo, **estilo_boton)
        self.btn_analizar.pack(pady=15)

        # Panel derecho con visualización y consola
        self.frame_imagen = tk.Frame(self.root, bg="#ffffff")
        self.frame_imagen.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Imagen de resultado
        self.label_imagen = tk.Label(self.frame_imagen, bg="#ffffff")
        self.label_imagen.pack(expand=True)

        # Consola inferior
        self.text_log = tk.Text(self.frame_imagen, height=10, width=60, bg="#ffffff", fg="black",
                        font=("Consolas", 10), borderwidth=0, relief=tk.FLAT)
        self.text_log.pack(pady=5)

        # Redirección de print a la consola
        sys.stdout = ConsoleRedirector(self.text_log)

    def cargar_imagen(self):
        # Diálogo para seleccionar imagen HDR
        ruta = filedialog.askopenfilename(title="Selecciona una imagen HDR", filetypes=[("Archivos HDR", "*.hdr")])
        if not ruta:
            messagebox.showwarning("Advertencia", "No se seleccionó ninguna imagen.")
            return

        # Procesamiento de imagen
        self.ruta_imagen = ruta
        self.arr_norm = preprocesar_imagen(ruta)

        if self.arr_norm is not None:
            messagebox.showinfo("Éxito", f"Imagen cargada:\n{ruta}")
            self.mostrar_imagen()
        else:
            messagebox.showerror("Error", "Error al procesar la imagen.")

    def mostrar_imagen(self):
        # Visualización de imagen RGB en la interfaz
        try:
            if self.arr_norm is not None:
                rgb = (self.arr_norm[:, :, [29, 19, 2]] * 255).astype(np.uint8)
                img = Image.fromarray(rgb).resize((500, 500))
                self.img_tk = ImageTk.PhotoImage(img)
                self.label_imagen.config(image=self.img_tk)
                self.imagenes_resultado = [self.img_tk]
                self.pagina_actual = 0
        except Exception as e:
            print(f"Error al mostrar la imagen: {e}")

    def ejecutar_analisis_en_hilo(self):
        # Análisis en segundo plano
        self.bloquear_controles()
        hilo = threading.Thread(target=self.ejecutar_analisis)
        hilo.daemon = True
        hilo.start()

    def bloquear_controles(self):
        # Desactivar controles durante análisis
        self.btn_cargar.config(state=tk.DISABLED)
        self.btn_analizar.config(state=tk.DISABLED)
        self.metodo_menu.config(state=tk.DISABLED)

    def desbloquear_controles(self):
        # Reactivar controles tras análisis
        self.btn_cargar.config(state=tk.NORMAL)
        self.btn_analizar.config(state=tk.NORMAL)
        self.metodo_menu.config(state=tk.NORMAL)

    def salir(self):
        # Salir de la aplicación
        self.root.destroy()

# Lanzamiento de la aplicación
if __name__ == "__main__":
    root = tk.Tk()
    app = InterfazGrafica(root)
    root.mainloop()
