
# Importación de librerías necesarias
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import numpy as np
from PIL import Image, ImageTk
import sys
import io
import threading  # Permite ejecutar tareas en segundo plano sin congelar la interfaz

# Redirección de print() a la consola en la interfaz
class ConsoleRedirector(io.StringIO):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def write(self, s):
        self.text_widget.insert(tk.END, s)  # Inserta el texto al final
        self.text_widget.see(tk.END)        # Hace scroll automático
        self.text_widget.update_idletasks() # Refresca de inmediato

    def flush(self):
        pass  # No se necesita funcionalidad adicional aquí

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

# Clase principal de la interfaz gráfica
class InterfazGrafica:
    def __init__(self, root):
        self.root = root
        self.root.title("Análisis de Imágenes Hiperespectrales")

        # Centrar la ventana en la parte superior de la pantalla
        ancho_ventana, alto_ventana = 800, 600
        x = (self.root.winfo_screenwidth() // 2) - (ancho_ventana // 2)
        y = 15
        self.root.geometry(f"{ancho_ventana}x{alto_ventana}+{x}+{y}")
        self.root.protocol("WM_DELETE_WINDOW", self.salir)

        # Variables de estado
        self.ruta_imagen = None              # Ruta de la imagen cargada
        self.arr_norm = None                 # Imagen normalizada
        self.metodo = tk.StringVar(value="Seleccionar")  # Método elegido
        self.imagenes_resultado = []         # Lista de imágenes a mostrar
        self.pagina_actual = 0               # Página actual mostrada

        # Referencias a botones para habilitar/deshabilitar
        self.btn_cargar = None
        self.btn_analizar = None
        self.metodo_menu = None
        self.btn_siguiente = None

        # Construcción de la interfaz
        self.crear_widgets()

    def crear_widgets(self):
        # Panel lateral izquierdo (botones y logo)
        frame_botones = tk.Frame(self.root, bg="#dfe6df")
        frame_botones.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)

        # Carga de logo
        try:
            ruta_logo = r"C:\Users\super\Downloads\Proyecto De Grado 1.1\project.ecospectral.edu.udc\src\Logo\logo.png"
            if os.path.exists(ruta_logo):
                logo_img = Image.open(ruta_logo).resize((240, 240))
                self.logo_tk = ImageTk.PhotoImage(logo_img)
                tk.Label(frame_botones, image=self.logo_tk, bg="#dfe6df").pack(pady=(10, 20))
            else:
                tk.Label(frame_botones, text="LOGO", font=("Arial", 16), bg="#dfe6df").pack(pady=(10, 20))
        except Exception as e:
            print(f"No se pudo cargar el logo: {e}")

        # Estilo base para botones
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

        # Contenedor de botones centrales
        frame_centro = tk.Frame(frame_botones, bg="#dfe6df")
        frame_centro.pack(expand=True)

        # Botón para cargar imagen
        self.btn_cargar = tk.Button(frame_centro, text="Cargar Imagen", command=self.cargar_imagen, **estilo_boton)
        self.btn_cargar.pack(pady=15)

        # Menú de selección de método
        tk.Label(frame_centro, text="Seleccionar Método:", bg="#dfe6df", font=("Helvetica", 15, "bold")).pack(pady=(10, 5))
        self.metodo_menu = tk.OptionMenu(frame_centro, self.metodo, "Correlation", "Euclidean", "CNN")
        self.metodo_menu.config(**estilo_boton)
        self.metodo_menu.pack(pady=(0, 15))

        # Botón para ejecutar análisis
        self.btn_analizar = tk.Button(frame_centro, text="Ejecutar Análisis", command=self.ejecutar_analisis_en_hilo, **estilo_boton)
        self.btn_analizar.pack(pady=15)

        # Panel derecho (imagen + consola)
        self.frame_imagen = tk.Frame(self.root, bg="#ffffff")
        self.frame_imagen.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Etiqueta que muestra la imagen
        self.label_imagen = tk.Label(self.frame_imagen, bg="#ffffff")
        self.label_imagen.pack(expand=True)

        # Consola inferior centrada
        self.text_log = tk.Text(self.frame_imagen, height=10, width=60, bg="#ffffff", fg="black",
                        font=("Consolas", 10), borderwidth=0, relief=tk.FLAT)
        self.text_log.pack(pady=5)

        # Redirige print() a consola
        sys.stdout = ConsoleRedirector(self.text_log)

    def cargar_imagen(self):
        # Selecciona imagen .hdr desde el sistema
        ruta = filedialog.askopenfilename(title="Selecciona una imagen HDR", filetypes=[("Archivos HDR", "*.hdr")])
        if not ruta:
            messagebox.showwarning("Advertencia", "No se seleccionó ninguna imagen.")
            return

        # Preprocesamiento
        self.ruta_imagen = ruta
        self.arr_norm = preprocesar_imagen(ruta)

        if self.arr_norm is not None:
            messagebox.showinfo("Éxito", f"Imagen cargada:\n{ruta}")
            self.mostrar_imagen()
        else:
            messagebox.showerror("Error", "Error al procesar la imagen.")

    def mostrar_imagen(self):
        try:
            if self.arr_norm is not None:
                # Selecciona bandas RGB y convierte en imagen
                rgb = (self.arr_norm[:, :, [29, 19, 2]] * 255).astype(np.uint8)
                img = Image.fromarray(rgb).resize((500, 500))
                self.img_tk = ImageTk.PhotoImage(img)

                # Muestra la imagen
                self.label_imagen.config(image=self.img_tk)
                self.imagenes_resultado = [self.img_tk]
                self.pagina_actual = 0
        except Exception as e:
            print(f"Error al mostrar la imagen: {e}")

    def ejecutar_analisis_en_hilo(self):
        # Ejecuta el análisis en segundo plano
        self.bloquear_controles()
        hilo = threading.Thread(target=self.ejecutar_analisis)
        hilo.daemon = True  #El hilo se cerrará cuando se cierre la app
        hilo.start()


    def ejecutar_analisis(self):
    # Validación: asegurar que haya imagen cargada y método seleccionado
        if self.ruta_imagen is None or self.arr_norm is None or not self.metodo.get():
            messagebox.showerror("Error", "Carga una imagen y selecciona un método antes de analizar.")
            self.desbloquear_controles()
            return

        try:
            metodo = self.metodo.get()
            print(f"Ejecutando análisis con método: {metodo}")

            # Rutas base de modelos y datos espectrales
            ruta_base = r"C:\Users\super\Downloads\Proyecto De Grado\project.ecospectral.edu.udc\src"
            ruta_veg = os.path.join(ruta_base, "database", "fsp", "fsp_veg.npy")
            ruta_defo = os.path.join(ruta_base, "database", "fsp", "fsp_defo.npy")
            ruta_refo = os.path.join(ruta_base, "database", "fsp", "fsp_refo.npy")

            # Carga de firmas espectrales promedio
            pix_prom_veg = np.load(ruta_veg)
            pix_prom_defo = np.load(ruta_defo)
            pix_prom_refo = np.load(ruta_refo)

            # Si el método es CNN
            if metodo.lower() == "cnn":
                ruta_cnnm_vd = os.path.join(ruta_base, "database", "cnnm", "cnn_vd.keras")
                ruta_cnnm_refo = os.path.join(ruta_base, "database", "cnnm", "cnn_refo.keras")
                cnn_vd, cnn_refo = cargar_modelo(ruta_cnnm_vd, ruta_cnnm_refo)
                analizar_imagen(self.arr_norm, cnn_vd, cnn_refo, self.ruta_imagen, ruta_base)

            else:
                # Calcular umbrales automáticos según método
                umbral_defo, umbral_veg = calcular_umbrales_auto(self.arr_norm, pix_prom_veg, pix_prom_defo, metodo)

                # Detectar píxeles intermedios
                categorias = detectar_pixeles_intermedios(self.arr_norm, pix_prom_defo, pix_prom_veg,
                                                        umbral_defo, umbral_veg, metodo=metodo)

                # Detectar suelos reforestables
                categorias_refo = detectar_suelos_reforestables(self.arr_norm, pix_prom_defo, pix_prom_veg,
                                                                umbral_defo, umbral_veg, metodo=metodo)

                # Aplicar colores a clasificación e imagen reforestada
                arr_coloreado, mascara_color, contadores = aplicar_colores(self.arr_norm, categorias)
                mascara_reforestable, contadores_reforestables = aplicar_colores_reforestable(self.arr_norm, categorias_refo)

                # Visualización 1: Clasificación de vegetación y deforestación
                arr_rgb = np.copy(self.arr_norm[:, :, [29, 19, 2]])
                fig1, ax1 = plt.subplots(figsize=(10, 8))
                ax1.imshow(arr_rgb)
                ax1.imshow(mascara_color)

                # Leyenda personalizada
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
                ax1.set_title('Análisis de Vegetación y Deforestación', fontsize=14, fontweight='bold')

                # Guardar imagen de clasificación
                nombre_archivo = os.path.basename(self.ruta_imagen).split('.')[0]
                ruta_salida1 = os.path.join(ruta_base, "img_processed", f"{nombre_archivo}_analisis_intermedio.png")
                os.makedirs(os.path.dirname(ruta_salida1), exist_ok=True)
                fig1.savefig(ruta_salida1)
                plt.show()

                # Visualización 2: Suelos reforestables
                fig2, ax2 = plt.subplots(figsize=(10, 8))
                ax2.imshow(arr_rgb)
                ax2.imshow(mascara_reforestable)

                handles_refo = []
                for i in range(len(categorias_refo)):
                    r = 1.0
                    g = (i / (len(categorias_refo) - 1)) if len(categorias_refo) > 1 else 0.5
                    b = 0.0
                    etiqueta = f"Porc {i * 25}% ({contadores_reforestables[i]})"
                    handles_refo.append(mpatches.Patch(color=[r, g, b], label=etiqueta, alpha=0.4))

                ax2.legend(handles=handles_refo, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=len(handles_refo))
                total_reforestables = sum(contadores_reforestables)
                ax2.set_title(f'Suelos Reforestables ({total_reforestables} píxeles)', fontsize=14, fontweight='bold')

                ruta_salida2 = os.path.join(ruta_base, "img_processed", f"{nombre_archivo}_analisis_reforestable.png")
                fig2.savefig(ruta_salida2)
                plt.show()

                # Estadísticas por consola
                print("\n===== ESTADÍSTICAS =====")
                print(f"\n📊 Vegetación ➡ Reforestación")
                for i, contador in enumerate(contadores):
                    nombre = "Deforestación" if i == 0 else (
                            "Vegetación" if i == len(contadores) - 1 else f"Intermedio {i}")
                    print(f"Píxeles de {nombre}: {contador}")

                print(f"\n📊 Potencial de Reforestación")
                for i, contador in enumerate(contadores_reforestables):
                    nivel = "Bajo" if i == 0 else (
                            "Alto" if i == len(contadores_reforestables) - 1 else f"Medio {i}")
                    print(f"Potencial {nivel}: {contador} píxeles")

                print(f"Total de píxeles reforestables: {total_reforestables}")

                # Gráfico de firmas espectrales
                plot_firmas_spectrales(pix_prom_defo, pix_prom_veg, pix_prom_refo)

            # Mensaje final
            print("Análisis completado correctamente.")
            messagebox.showinfo("Finalizado", "Análisis completado correctamente.")

        except Exception as e:
            # Manejo de errores
            messagebox.showerror("Error", f"Ocurrió un error:\n{str(e)}")

        finally:
            # Reactivar botones
            self.desbloquear_controles()


    def bloquear_controles(self):
        # Desactiva los botones durante el análisis
        self.btn_cargar.config(state=tk.DISABLED)
        self.btn_analizar.config(state=tk.DISABLED)
        self.metodo_menu.config(state=tk.DISABLED)

    def desbloquear_controles(self):
        # Reactiva los botones al finalizar
        self.btn_cargar.config(state=tk.NORMAL)
        self.btn_analizar.config(state=tk.NORMAL)
        self.metodo_menu.config(state=tk.NORMAL)

    def salir(self):
        # Cierra la ventana
        self.root.destroy()

# Ejecuta la aplicación
if __name__ == "__main__":
    root = tk.Tk()
    app = InterfazGrafica(root)
    root.mainloop()
