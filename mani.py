import os
import sys
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import scrolledtext
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import random
from datetime import datetime, timedelta

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("anestesia_analytics.log"), logging.StreamHandler()]
)
logger = logging.getLogger("AnestesiaAnalytics")

# Constantes del sistema
PARAMETROS_VITALES = [
    "Frecuencia Cardíaca", "Presión Arterial Sistólica", "Presión Arterial Diastólica",
    "SpO2", "ETCO2", "Temperatura", "BIS", "TOF", "Frecuencia Respiratoria",
    "Presión Media", "Nivel de Sedación"
]

# Importar librería adicional para visualización del corazón
import io
import base64
from PIL import Image, ImageTk
import math

RANGOS_NORMALES = {
    "Frecuencia Cardíaca": (60, 100),
    "Presión Arterial Sistólica": (90, 140),
    "Presión Arterial Diastólica": (60, 90),
    "SpO2": (95, 100),
    "ETCO2": (35, 45),
    "Temperatura": (36, 37.5),
    "BIS": (40, 60),
    "TOF": (0, 25),
    "Frecuencia Respiratoria": (12, 20),
    "Presión Media": (70, 100),
    "Nivel de Sedación": (3, 4),
}

COLORES_ESTADO = {
    "Normal": "#4CAF50",  # Verde
    "Advertencia": "#FFC107",  # Amarillo
    "Peligro": "#F44336",  # Rojo
}

# Modelo CNN para análisis de datos de anestesia
class ModeloCNNAnestesia(nn.Module):
    def __init__(self, input_features=11, output_classes=3):
        """
        Modelo CNN para analizar datos de anestesia
        
        Args:
            input_features: Número de parámetros vitales de entrada
            output_classes: Número de clases de salida (Normal, Advertencia, Peligro)
        """
        super(ModeloCNNAnestesia, self).__init__()
        
        # Arquitectura simplificada sin BatchNorm para evitar problemas con muestras individuales
        self.fc1 = nn.Linear(input_features, 128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, output_classes)
        
    def forward(self, x):
        # x tiene forma [batch_size, features]
        
        # Aplicar capas totalmente conectadas sin BatchNorm
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x

# Función para generar datos de simulación
def generar_datos_simulacion(num_muestras=1, variacion=0.1, estado_paciente="Normal"):
    """
    Genera datos de simulación para parámetros vitales del paciente.
    
    Args:
        num_muestras: Número de muestras a generar
        variacion: Nivel de variación en los datos
        estado_paciente: "Normal", "Advertencia" o "Peligro"
    
    Returns:
        DataFrame con los datos generados
    """
    datos = []
    timestamp_actual = datetime.now()
    
    for i in range(num_muestras):
        muestra = {}
        muestra["Timestamp"] = timestamp_actual - timedelta(seconds=num_muestras-i)
        
        for parametro in PARAMETROS_VITALES:
            min_val, max_val = RANGOS_NORMALES[parametro]
            valor_base = (min_val + max_val) / 2
            
            # Ajustar el valor base según el estado del paciente
            if estado_paciente == "Advertencia":
                # Para advertencia, desplazamos un poco hacia los límites
                if random.random() > 0.5:
                    valor_base = min_val + (max_val - min_val) * 0.2
                else:
                    valor_base = max_val - (max_val - min_val) * 0.2
            elif estado_paciente == "Peligro":
                # Para peligro, desplazamos fuera de los límites
                if random.random() > 0.5:
                    valor_base = min_val - (max_val - min_val) * 0.1
                else:
                    valor_base = max_val + (max_val - min_val) * 0.1
            
            # Añadir variación aleatoria
            variacion_actual = (max_val - min_val) * variacion
            valor = valor_base + random.uniform(-variacion_actual, variacion_actual)
            
            muestra[parametro] = round(valor, 2)
        
        datos.append(muestra)
    
    return pd.DataFrame(datos)

# Función para evaluar el estado del paciente basado en los parámetros vitales
def evaluar_estado_paciente(datos_vitales):
    """
    Evalúa el estado del paciente basado en sus signos vitales.
    
    Args:
        datos_vitales: Dict o Series con los parámetros vitales
    
    Returns:
        Estado ("Normal", "Advertencia", "Peligro") y detalles
    """
    num_fuera_rango = 0
    num_peligro = 0
    detalles = []
    
    for parametro in PARAMETROS_VITALES:
        if parametro in datos_vitales:
            valor = datos_vitales[parametro]
            min_val, max_val = RANGOS_NORMALES[parametro]
            
            # Determinar si está fuera de rango
            if valor < min_val or valor > max_val:
                num_fuera_rango += 1
                
                # Determinar si está en rango de peligro (10% fuera del límite)
                margen_peligro = (max_val - min_val) * 0.1
                if valor < min_val - margen_peligro or valor > max_val + margen_peligro:
                    num_peligro += 1
                    detalles.append(f"{parametro}: {valor} (PELIGRO)")
                else:
                    detalles.append(f"{parametro}: {valor} (ADVERTENCIA)")
            else:
                detalles.append(f"{parametro}: {valor}")
    
    # Determinar estado general
    if num_peligro > 0:
        return "Peligro", detalles
    elif num_fuera_rango > 0:
        return "Advertencia", detalles
    else:
        return "Normal", detalles

# Preprocesamiento de datos para el modelo CNN
def preprocesar_datos(df, ventana_tiempo=10):
    """
    Preprocesa los datos para alimentar al modelo CNN.
    
    Args:
        df: DataFrame con los datos vitales
        ventana_tiempo: Número de muestras en la ventana temporal
    
    Returns:
        Datos procesados como tensor
    """
    # Si hay menos datos que la ventana, duplicamos los existentes
    if len(df) < ventana_tiempo:
        df = pd.concat([df] * (ventana_tiempo // len(df) + 1)).reset_index(drop=True)
        df = df.iloc[:ventana_tiempo]
    
    # Tomamos los últimos 'ventana_tiempo' registros
    df = df.iloc[-ventana_tiempo:]
    
    # Extraer solo los parámetros vitales numéricos
    X = df[PARAMETROS_VITALES].values
    
    # Normalizar los datos
    for i, parametro in enumerate(PARAMETROS_VITALES):
        min_val, max_val = RANGOS_NORMALES[parametro]
        rango = max_val - min_val
        X[:, i] = (X[:, i] - min_val) / rango
    
    # Convertir a tensor de PyTorch - formato para último registro
    # Redimensionar a [batch_size=1, features]
    X_tensor = torch.FloatTensor(X[-1]).unsqueeze(0)
    
    return X_tensor

# Clase principal de la aplicación
class AplicacionAnestesiaAnalytics(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Sistema de Análisis para Anestesiólogos")
        self.geometry("1400x800")
        self.configure(bg="#f0f0f0")
        
        # Variables de estado
        self.datos_paciente = None
        self.historial_datos = pd.DataFrame()
        self.simulacion_activa = False
        self.modelo_cnn = None
        self.estado_actual = "Normal"
        
        # Inicializar componentes de la interfaz
        self.crear_menu()
        self.crear_interfaz()
        
        # Cargar o inicializar modelo CNN
        self.inicializar_modelo()
    
    def crear_menu(self):
        """Crea la barra de menú de la aplicación"""
        menu_principal = tk.Menu(self)
        self.config(menu=menu_principal)
        
        # Menú Archivo
        menu_archivo = tk.Menu(menu_principal, tearoff=0)
        menu_principal.add_cascade(label="Archivo", menu=menu_archivo)
        menu_archivo.add_command(label="Cargar Datos", command=self.cargar_datos)
        menu_archivo.add_command(label="Guardar Informe", command=self.guardar_informe)
        menu_archivo.add_separator()
        menu_archivo.add_command(label="Salir", command=self.quit)
        
        # Menú Simulación
        menu_simulacion = tk.Menu(menu_principal, tearoff=0)
        menu_principal.add_cascade(label="Simulación", menu=menu_simulacion)
        menu_simulacion.add_command(label="Iniciar Simulación", command=self.iniciar_simulacion)
        menu_simulacion.add_command(label="Detener Simulación", command=self.detener_simulacion)
        menu_simulacion.add_separator()
        menu_simulacion.add_command(label="Configurar Simulación", command=self.configurar_simulacion)
        menu_simulacion.add_separator()
        menu_simulacion.add_command(label="Mostrar Simulación del Corazón", command=self.mostrar_simulacion_corazon)
        
        # Menú Análisis
        menu_analisis = tk.Menu(menu_principal, tearoff=0)
        menu_principal.add_cascade(label="Análisis", menu=menu_analisis)
        menu_analisis.add_command(label="Analizar Datos Actuales", command=self.analizar_datos)
        menu_analisis.add_command(label="Ver Tendencias", command=self.ver_tendencias)
        menu_analisis.add_command(label="Generar Recomendaciones", command=self.generar_recomendaciones)
        
        # Menú Ayuda
        menu_ayuda = tk.Menu(menu_principal, tearoff=0)
        menu_principal.add_cascade(label="Ayuda", menu=menu_ayuda)
        menu_ayuda.add_command(label="Manual de Usuario", command=self.mostrar_manual)
        menu_ayuda.add_command(label="Acerca de", command=self.mostrar_acerca_de)
    
    def crear_interfaz(self):
        """Crea la interfaz principal de la aplicación"""
        # Frame principal con grid 2x2
        self.frame_principal = ttk.Frame(self)
        self.frame_principal.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 1. Panel de Información del Paciente (arriba-izquierda)
        self.frame_info_paciente = ttk.LabelFrame(self.frame_principal, text="Información del Paciente")
        self.frame_info_paciente.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Campos de información básica
        ttk.Label(self.frame_info_paciente, text="ID Paciente:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.id_paciente_var = tk.StringVar(value="P-12345")
        ttk.Entry(self.frame_info_paciente, textvariable=self.id_paciente_var, width=15).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(self.frame_info_paciente, text="Nombre:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.nombre_paciente_var = tk.StringVar(value="Paciente de Simulación")
        ttk.Entry(self.frame_info_paciente, textvariable=self.nombre_paciente_var, width=25).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(self.frame_info_paciente, text="Edad:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.edad_paciente_var = tk.StringVar(value="45")
        ttk.Entry(self.frame_info_paciente, textvariable=self.edad_paciente_var, width=10).grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(self.frame_info_paciente, text="Peso (kg):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.peso_paciente_var = tk.StringVar(value="70")
        ttk.Entry(self.frame_info_paciente, textvariable=self.peso_paciente_var, width=10).grid(row=3, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(self.frame_info_paciente, text="Tipo de Cirugía:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.tipo_cirugia_var = tk.StringVar(value="Simulación")
        ttk.Entry(self.frame_info_paciente, textvariable=self.tipo_cirugia_var, width=25).grid(row=4, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(self.frame_info_paciente, text="Estado:").grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.estado_paciente_var = tk.StringVar(value="Normal")
        self.lbl_estado_paciente = ttk.Label(self.frame_info_paciente, textvariable=self.estado_paciente_var, background=COLORES_ESTADO["Normal"], width=15)
        self.lbl_estado_paciente.grid(row=5, column=1, padx=5, pady=5, sticky="w")
        
        # 2. Panel de Monitores (arriba-derecha)
        self.frame_monitores = ttk.LabelFrame(self.frame_principal, text="Monitores en Tiempo Real")
        self.frame_monitores.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        # Crear gráficos para los monitores principales
        self.fig_monitores = plt.Figure(figsize=(6, 4), dpi=100)
        self.canvas_monitores = FigureCanvasTkAgg(self.fig_monitores, master=self.frame_monitores)
        self.canvas_monitores.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Inicializar los subplots
        self.ax1 = self.fig_monitores.add_subplot(221)
        self.ax2 = self.fig_monitores.add_subplot(222)
        self.ax3 = self.fig_monitores.add_subplot(223)
        self.ax4 = self.fig_monitores.add_subplot(224)
        
        self.ax1.set_title("Frecuencia Cardíaca")
        self.ax2.set_title("Presión Arterial")
        self.ax3.set_title("SpO2")
        self.ax4.set_title("ETCO2")
        
        self.fig_monitores.tight_layout()
        
        # 3. Panel de Parámetros Vitales (abajo-izquierda)
        self.frame_vitales = ttk.LabelFrame(self.frame_principal, text="Parámetros Vitales")
        self.frame_vitales.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        # Crear tabla para mostrar parámetros vitales
        self.tabla_vitales = ttk.Treeview(self.frame_vitales, columns=("Parámetro", "Valor", "Rango Normal", "Estado"), show="headings", height=10)
        self.tabla_vitales.heading("Parámetro", text="Parámetro")
        self.tabla_vitales.heading("Valor", text="Valor")
        self.tabla_vitales.heading("Rango Normal", text="Rango Normal")
        self.tabla_vitales.heading("Estado", text="Estado")
        
        self.tabla_vitales.column("Parámetro", width=150)
        self.tabla_vitales.column("Valor", width=80)
        self.tabla_vitales.column("Rango Normal", width=120)
        self.tabla_vitales.column("Estado", width=80)
        
        scrollbar = ttk.Scrollbar(self.frame_vitales, orient=tk.VERTICAL, command=self.tabla_vitales.yview)
        self.tabla_vitales.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tabla_vitales.pack(fill=tk.BOTH, expand=True)
        
        # 4. Panel de Análisis y Recomendaciones (abajo-derecha)
        self.frame_analisis = ttk.LabelFrame(self.frame_principal, text="Análisis y Recomendaciones")
        self.frame_analisis.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        
        # Texto informativo para análisis
        self.txt_analisis = scrolledtext.ScrolledText(self.frame_analisis, wrap=tk.WORD, height=10)
        self.txt_analisis.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.txt_analisis.insert(tk.END, "Bienvenido al Sistema de Análisis para Anestesiólogos.\n\n")
        self.txt_analisis.insert(tk.END, "Este sistema utiliza redes neuronales convolucionales (CNN) para analizar los datos del paciente y proporcionar recomendaciones en tiempo real.\n\n")
        self.txt_analisis.insert(tk.END, "Inicie una simulación o cargue datos reales para comenzar el análisis.")
        self.txt_analisis.config(state=tk.DISABLED)
        
        # Configurar pesos de las filas y columnas
        self.frame_principal.grid_rowconfigure(0, weight=1)
        self.frame_principal.grid_rowconfigure(1, weight=1)
        self.frame_principal.grid_columnconfigure(0, weight=1)
        self.frame_principal.grid_columnconfigure(1, weight=1)
        
        # Barra de estado
        self.barra_estado = ttk.Label(self, text="Listo", relief=tk.SUNKEN, anchor=tk.W)
        self.barra_estado.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Control de simulación
        self.frame_control = ttk.Frame(self)
        self.frame_control.pack(fill=tk.X, padx=10, pady=5)
        
        self.btn_iniciar = ttk.Button(self.frame_control, text="Iniciar Simulación", command=self.iniciar_simulacion)
        self.btn_iniciar.pack(side=tk.LEFT, padx=5)
        
        self.btn_detener = ttk.Button(self.frame_control, text="Detener Simulación", command=self.detener_simulacion, state=tk.DISABLED)
        self.btn_detener.pack(side=tk.LEFT, padx=5)
        
        # ComboBox para seleccionar el estado de simulación
        ttk.Label(self.frame_control, text="Estado simulado:").pack(side=tk.LEFT, padx=5)
        self.estado_simulacion_var = tk.StringVar(value="Normal")
        self.combo_estado = ttk.Combobox(self.frame_control, textvariable=self.estado_simulacion_var, values=["Normal", "Advertencia", "Peligro"], state="readonly", width=15)
        self.combo_estado.pack(side=tk.LEFT, padx=5)
        self.combo_estado.bind("<<ComboboxSelected>>", self.cambiar_estado_simulacion)
        
        # Control de eventos
        self.btn_evento = ttk.Button(self.frame_control, text="Simular Evento", command=self.simular_evento)
        self.btn_evento.pack(side=tk.LEFT, padx=5)
    
    def inicializar_modelo(self):
        """Inicializa o carga el modelo CNN"""
        try:
            # Intentar cargar modelo preentrenado si existe
            if os.path.exists("modelo_anestesia_cnn.pth"):
                self.modelo_cnn = ModeloCNNAnestesia()
                self.modelo_cnn.load_state_dict(torch.load("modelo_anestesia_cnn.pth"))
                self.modelo_cnn.eval()
                logger.info("Modelo CNN cargado correctamente")
            else:
                # Crear modelo nuevo
                self.modelo_cnn = ModeloCNNAnestesia()
                logger.info("Nuevo modelo CNN inicializado")
            
            # Mostrar información en la barra de estado
            self.barra_estado.config(text="Modelo CNN inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar el modelo: {e}")
            messagebox.showerror("Error", f"No se pudo inicializar el modelo CNN: {e}")
    
    def cargar_datos(self):
        """Carga datos desde un archivo CSV"""
        try:
            filename = filedialog.askopenfilename(
                title="Seleccionar archivo de datos",
                filetypes=(("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*"))
            )
            
            if filename:
                # Cargar datos
                self.historial_datos = pd.read_csv(filename)
                
                # Actualizar interfaz con los datos cargados
                self.actualizar_interfaz_con_datos(self.historial_datos.iloc[-1])
                
                # Mostrar mensaje de éxito
                num_registros = len(self.historial_datos)
                self.barra_estado.config(text=f"Datos cargados correctamente: {num_registros} registros")
                
                # Actualizar gráficos
                self.actualizar_graficos()
                
                # Analizar datos
                self.analizar_datos()
        except Exception as e:
            logger.error(f"Error al cargar datos: {e}")
            messagebox.showerror("Error", f"No se pudieron cargar los datos: {e}")
    
    def guardar_informe(self):
        """Guarda un informe con el análisis actual"""
        try:
            if self.historial_datos.empty:
                messagebox.showwarning("Advertencia", "No hay datos para generar un informe.")
                return
            
            filename = filedialog.asksaveasfilename(
                title="Guardar informe",
                defaultextension=".html",
                filetypes=(("Archivo HTML", "*.html"), ("Todos los archivos", "*.*"))
            )
            
            if filename:
                # Crear informe HTML
                informe = self.generar_informe_html()
                
                # Guardar a archivo
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(informe)
                
                # Mostrar mensaje de éxito
                self.barra_estado.config(text=f"Informe guardado correctamente en {filename}")
                messagebox.showinfo("Éxito", f"Informe guardado correctamente.")
        except Exception as e:
            logger.error(f"Error al guardar informe: {e}")
            messagebox.showerror("Error", f"No se pudo guardar el informe: {e}")
    
    def generar_informe_html(self):
        """Genera un informe HTML con el análisis actual"""
        # Obtener datos actuales
        ultimo_registro = self.historial_datos.iloc[-1] if not self.historial_datos.empty else None
        estado, detalles = evaluar_estado_paciente(ultimo_registro) if ultimo_registro is not None else ("Sin datos", [])
        
        # Generar HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Informe de Anestesia - {self.nombre_paciente_var.get()}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2962FF; }}
                .paciente-info {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .parametros {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .parametros th, .parametros td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .parametros th {{ background-color: #2962FF; color: white; }}
                .normal {{ background-color: #E8F5E9; }}
                .advertencia {{ background-color: #FFF9C4; }}
                .peligro {{ background-color: #FFEBEE; }}
                .estado-{estado.lower()} {{ font-weight: bold; color: {COLORES_ESTADO[estado]}; }}
                .recomendaciones {{ background-color: #E3F2FD; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Informe de Anestesia</h1>
            <div class="paciente-info">
                <h2>Información del Paciente</h2>
                <p><strong>ID:</strong> {self.id_paciente_var.get()}</p>
                <p><strong>Nombre:</strong> {self.nombre_paciente_var.get()}</p>
                <p><strong>Edad:</strong> {self.edad_paciente_var.get()} años</p>
                <p><strong>Peso:</strong> {self.peso_paciente_var.get()} kg</p>
                <p><strong>Tipo de Cirugía:</strong> {self.tipo_cirugia_var.get()}</p>
                <p><strong>Estado:</strong> <span class="estado-{estado.lower()}">{estado}</span></p>
                <p><strong>Fecha y Hora:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
            </div>
            
            <h2>Parámetros Vitales</h2>
        """
        
        # Añadir tabla de parámetros vitales
        if ultimo_registro is not None:
            html += """
            <table class="parametros">
                <tr>
                    <th>Parámetro</th>
                    <th>Valor</th>
                    <th>Rango Normal</th>
                    <th>Estado</th>
                </tr>
            """
            
            for parametro in PARAMETROS_VITALES:
                if parametro in ultimo_registro:
                    valor = ultimo_registro[parametro]
                    min_val, max_val = RANGOS_NORMALES[parametro]
                    
                    # Determinar estado
                    estado_param = "Normal"
                    if valor < min_val or valor > max_val:
                        margen_peligro = (max_val - min_val) * 0.1
                        if valor < min_val - margen_peligro or valor > max_val + margen_peligro:
                            estado_param = "Peligro"
                        else:
                            estado_param = "Advertencia"
                    
                    html += f"""
                    <tr class="{estado_param.lower()}">
                        <td>{parametro}</td>
                        <td>{valor}</td>
                        <td>{min_val} - {max_val}</td>
                        <td>{estado_param}</td>
                    </tr>
                    """
            
            html += "</table>"
        else:
            html += "<p>No hay datos disponibles para mostrar.</p>"
        
        # Añadir recomendaciones
        recomendaciones = self.generar_recomendaciones(devolver_texto=True)
        html += f"""
            <h2>Análisis y Recomendaciones</h2>
            <div class="recomendaciones">
                {recomendaciones}
            </div>
            
            <h2>Notas</h2>
            <p>Este informe fue generado automáticamente por el Sistema de Análisis para Anestesiólogos.</p>
            <p>Fecha y hora de generación: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
        </body>
        </html>
        """
        
        return html
    
    def iniciar_simulacion(self):
        """Inicia la simulación de datos del paciente"""
        if not self.simulacion_activa:
            self.simulacion_activa = True
            self.btn_iniciar.config(state=tk.DISABLED)
            self.btn_detener.config(state=tk.NORMAL)
            
            # Inicializar historial de datos si está vacío
            if self.historial_datos.empty:
                self.historial_datos = generar_datos_simulacion(
                    num_muestras=1, 
                    estado_paciente=self.estado_simulacion_var.get()
                )
            
            # Actualizar la interfaz
            self.barra_estado.config(text="Simulación en curso...")
            
            # Iniciar función de simulación recurrente
            self.simular_datos()
    
    def detener_simulacion(self):
        """Detiene la simulación de datos"""
        self.simulacion_activa = False
        self.btn_iniciar.config(state=tk.NORMAL)
        self.btn_detener.config(state=tk.DISABLED)
        self.barra_estado.config(text="Simulación detenida")
        
        # Si la simulación del corazón está activa, actualizar su estado
        if hasattr(self, 'corazon_activo') and self.corazon_activo and hasattr(self, 'ventana_corazon') and self.ventana_corazon is not None and self.ventana_corazon.winfo_exists() and hasattr(self, 'txt_info_cardiaca'):
            self.txt_info_cardiaca.config(state=tk.NORMAL)
            self.txt_info_cardiaca.insert(tk.END, "\n• Simulación detenida. Los datos cardíacos se han congelado.\n")
            self.txt_info_cardiaca.see(tk.END)
            self.txt_info_cardiaca.config(state=tk.DISABLED)
    
    def simular_datos(self):
        """Genera un nuevo conjunto de datos simulados y actualiza la interfaz"""
        if self.simulacion_activa:
            # Generar nuevos datos
            nuevos_datos = generar_datos_simulacion(
                num_muestras=1, 
                estado_paciente=self.estado_simulacion_var.get()
            )
            
            # Añadir al historial
            self.historial_datos = pd.concat([self.historial_datos, nuevos_datos], ignore_index=True)
            
            # Si el historial es muy largo, mantener solo los últimos 100 registros
            if len(self.historial_datos) > 100:
                self.historial_datos = self.historial_datos.iloc[-100:]
            
            # Actualizar la interfaz con los nuevos datos
            self.actualizar_interfaz_con_datos(nuevos_datos.iloc[0])
            
            # Programar la próxima actualización (cada 1 segundo)
            self.after(1000, self.simular_datos)
    
    def actualizar_interfaz_con_datos(self, datos):
        """Actualiza la interfaz con los datos proporcionados"""
        # Actualizar tabla de parámetros vitales
        self.tabla_vitales.delete(*self.tabla_vitales.get_children())
        
        for parametro in PARAMETROS_VITALES:
            if parametro in datos:
                valor = datos[parametro]
                min_val, max_val = RANGOS_NORMALES[parametro]
                
                # Determinar estado
                estado = "Normal"
                tags = ("normal",)
                
                if valor < min_val or valor > max_val:
                    margen_peligro = (max_val - min_val) * 0.1
                    if valor < min_val - margen_peligro or valor > max_val + margen_peligro:
                        estado = "Peligro"
                        tags = ("peligro",)
                    else:
                        estado = "Advertencia"
                        tags = ("advertencia",)
                
                self.tabla_vitales.insert("", tk.END, values=(
                    parametro, 
                    f"{valor:.1f}", 
                    f"{min_val} - {max_val}", 
                    estado
                ), tags=tags)
        
        # Configurar colores de las filas según el estado
        self.tabla_vitales.tag_configure("normal", background="#E8F5E9")
        self.tabla_vitales.tag_configure("advertencia", background="#FFF9C4")
        self.tabla_vitales.tag_configure("peligro", background="#FFEBEE")
        
        # Evaluar estado general del paciente
        estado, detalles = evaluar_estado_paciente(datos)
        
        # Actualizar estado en la interfaz
        self.estado_paciente_var.set(estado)
        self.lbl_estado_paciente.config(background=COLORES_ESTADO[estado])
        
        # Actualizar gráficos
        self.actualizar_graficos()
        
        # Actualizar simulación del corazón si está activa
        if hasattr(self, 'corazon_activo') and self.corazon_activo and hasattr(self, 'ventana_corazon') and self.ventana_corazon is not None and self.ventana_corazon.winfo_exists():
            # No llamamos directamente a actualizar_simulacion_corazon() ya que 
            # tiene su propio ciclo de actualización
            
            # Actualizar etiquetas de información cardíaca
            if hasattr(self, 'lbl_fc') and "Frecuencia Cardíaca" in datos:
                self.lbl_fc.config(text=f"Frecuencia Cardíaca: {datos['Frecuencia Cardíaca']:.0f} lpm")
            if hasattr(self, 'lbl_pas') and "Presión Arterial Sistólica" in datos:
                self.lbl_pas.config(text=f"Presión Arterial Sistólica: {datos['Presión Arterial Sistólica']:.0f} mmHg")
            if hasattr(self, 'lbl_pad') and "Presión Arterial Diastólica" in datos:
                self.lbl_pad.config(text=f"Presión Arterial Diastólica: {datos['Presión Arterial Diastólica']:.0f} mmHg")
            
            if hasattr(self, 'lbl_estado_cardiaco'):
                self.lbl_estado_cardiaco.config(text=f"Estado: {estado}")
        
        # Analizar datos con CNN si hay suficientes datos
        if len(self.historial_datos) > 5:
            self.analizar_datos()
    
    def actualizar_graficos(self):
        """Actualiza los gráficos de monitoreo en tiempo real"""
        if self.historial_datos.empty:
            return
        
        # Limpiar gráficos anteriores
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # Configurar títulos
        self.ax1.set_title("Frecuencia Cardíaca")
        self.ax2.set_title("Presión Arterial")
        self.ax3.set_title("SpO2")
        self.ax4.set_title("ETCO2")
        
        # Limitar a los últimos 30 registros para los gráficos
        datos_grafico = self.historial_datos.tail(30)
        
        # Convertir timestamp a índice si existe
        if "Timestamp" in datos_grafico.columns:
            x = pd.to_datetime(datos_grafico["Timestamp"])
            x_fmt = [ts.strftime('%H:%M:%S') for ts in x]
        else:
            x = range(len(datos_grafico))
            x_fmt = x
        
        # Gráfico de frecuencia cardíaca
        if "Frecuencia Cardíaca" in datos_grafico.columns:
            self.ax1.plot(x, datos_grafico["Frecuencia Cardíaca"], 'r-')
            min_val, max_val = RANGOS_NORMALES["Frecuencia Cardíaca"]
            self.ax1.axhspan(min_val, max_val, color='g', alpha=0.2)
            self.ax1.set_ylim(min_val - 20, max_val + 20)
        
        # Gráfico de presión arterial
        if "Presión Arterial Sistólica" in datos_grafico.columns and "Presión Arterial Diastólica" in datos_grafico.columns:
            self.ax2.plot(x, datos_grafico["Presión Arterial Sistólica"], 'b-', label='Sistólica')
            self.ax2.plot(x, datos_grafico["Presión Arterial Diastólica"], 'g-', label='Diastólica')
            self.ax2.legend(loc='upper right', fontsize='small')
            min_val_s, max_val_s = RANGOS_NORMALES["Presión Arterial Sistólica"]
            min_val_d, max_val_d = RANGOS_NORMALES["Presión Arterial Diastólica"]
            self.ax2.axhspan(min_val_s, max_val_s, color='b', alpha=0.1)
            self.ax2.axhspan(min_val_d, max_val_d, color='g', alpha=0.1)
            self.ax2.set_ylim(min_val_d - 20, max_val_s + 20)
        
        # Gráfico de SpO2
        if "SpO2" in datos_grafico.columns:
            self.ax3.plot(x, datos_grafico["SpO2"], 'c-')
            min_val, max_val = RANGOS_NORMALES["SpO2"]
            self.ax3.axhspan(min_val, max_val, color='c', alpha=0.2)
            self.ax3.set_ylim(min_val - 5, max_val + 5)
        
        # Gráfico de ETCO2
        if "ETCO2" in datos_grafico.columns:
            self.ax4.plot(x, datos_grafico["ETCO2"], 'm-')
            min_val, max_val = RANGOS_NORMALES["ETCO2"]
            self.ax4.axhspan(min_val, max_val, color='m', alpha=0.2)
            self.ax4.set_ylim(min_val - 10, max_val + 10)
        
        # Ajustar y redibujar
        self.fig_monitores.tight_layout()
        self.canvas_monitores.draw()
    
    def analizar_datos(self):
        """Analiza los datos actuales usando el modelo CNN"""
        if self.historial_datos.empty or self.modelo_cnn is None:
            return
        
        try:
            # Preprocesar datos para el modelo
            X = preprocesar_datos(self.historial_datos)
            
            # Hacer predicción con el modelo
            with torch.no_grad():
                outputs = self.modelo_cnn(X)  # Ya no necesitamos unsqueeze(0)
                _, prediccion = torch.max(outputs, 1)
                
                # Convertir a estado (0=Normal, 1=Advertencia, 2=Peligro)
                estados = ["Normal", "Advertencia", "Peligro"]
                estado_predicho = estados[prediccion.item()]
            
            # Evaluar estado basado en reglas
            ultimo_registro = self.historial_datos.iloc[-1]
            estado_reglas, detalles = evaluar_estado_paciente(ultimo_registro)
            
            # Actualizar análisis en la interfaz
            self.actualizar_analisis(estado_predicho, estado_reglas, detalles)
            
            # Actualizar barra de estado
            self.barra_estado.config(text=f"Análisis completado. Estado: {estado_predicho}")
            
        except Exception as e:
            logger.error(f"Error en el análisis de datos: {e}")
            # Usar fallback a análisis basado en reglas si el modelo falla
            ultimo_registro = self.historial_datos.iloc[-1]
            estado_reglas, detalles = evaluar_estado_paciente(ultimo_registro)
            self.actualizar_analisis("N/A", estado_reglas, detalles)
            self.barra_estado.config(text=f"Análisis basado en reglas. Estado: {estado_reglas}")
    
    def actualizar_analisis(self, estado_predicho, estado_reglas, detalles):
        """Actualiza el panel de análisis con la información actual"""
        # Habilitar edición del text widget
        self.txt_analisis.config(state=tk.NORMAL)
        
        # Limpiar contenido anterior
        self.txt_analisis.delete(1.0, tk.END)
        
        # Añadir nueva información
        self.txt_analisis.insert(tk.END, f"ANÁLISIS DE ESTADO DEL PACIENTE\n", "titulo")
        self.txt_analisis.insert(tk.END, f"Fecha y hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
        
        self.txt_analisis.insert(tk.END, f"Estado según CNN: ", "subtitulo")
        self.txt_analisis.insert(tk.END, f"{estado_predicho}\n", f"estado_{estado_predicho.lower()}")
        
        self.txt_analisis.insert(tk.END, f"Estado según reglas: ", "subtitulo")
        self.txt_analisis.insert(tk.END, f"{estado_reglas}\n\n", f"estado_{estado_reglas.lower()}")
        
        self.txt_analisis.insert(tk.END, "Detalles de parámetros:\n", "subtitulo")
        for detalle in detalles:
            if "PELIGRO" in detalle:
                self.txt_analisis.insert(tk.END, f"• {detalle}\n", "peligro")
            elif "ADVERTENCIA" in detalle:
                self.txt_analisis.insert(tk.END, f"• {detalle}\n", "advertencia")
            else:
                self.txt_analisis.insert(tk.END, f"• {detalle}\n")
        
        self.txt_analisis.insert(tk.END, "\nRECOMENDACIONES:\n", "titulo")
        
        # Generar recomendaciones basadas en el estado
        if estado_predicho == "Peligro" or estado_reglas == "Peligro":
            self.txt_analisis.insert(tk.END, "• ATENCIÓN INMEDIATA REQUERIDA\n", "peligro")
            
            # Identificar parámetros específicos en peligro
            for detalle in detalles:
                if "PELIGRO" in detalle:
                    param = detalle.split(":")[0].strip()
                    
                    if "Frecuencia Cardíaca" in param:
                        if float(detalle.split(":")[1].split()[0]) > RANGOS_NORMALES[param][1]:
                            self.txt_analisis.insert(tk.END, "• Considerar administración de betabloqueantes\n")
                        else:
                            self.txt_analisis.insert(tk.END, "• Considerar administración de atropina\n")
                    
                    elif "Presión Arterial" in param:
                        if float(detalle.split(":")[1].split()[0]) > RANGOS_NORMALES[param][1]:
                            self.txt_analisis.insert(tk.END, "• Reducir profundidad anestésica y considerar antihipertensivos\n")
                        else:
                            self.txt_analisis.insert(tk.END, "• Administrar fluidos y considerar vasopresores\n")
                    
                    elif "SpO2" in param:
                        self.txt_analisis.insert(tk.END, "• Verificar vía aérea y aumentar FiO2\n")
                        self.txt_analisis.insert(tk.END, "• Considerar ventilación manual\n")
                    
                    elif "ETCO2" in param:
                        if float(detalle.split(":")[1].split()[0]) > RANGOS_NORMALES[param][1]:
                            self.txt_analisis.insert(tk.END, "• Verificar ventilación y ajustar parámetros\n")
                        else:
                            self.txt_analisis.insert(tk.END, "• Verificar posible embolismo o hipotermia\n")
            
            self.txt_analisis.insert(tk.END, "• Notificar al equipo quirúrgico\n")
            self.txt_analisis.insert(tk.END, "• Preparar medicamentos de emergencia\n")
            
        elif estado_predicho == "Advertencia" or estado_reglas == "Advertencia":
            self.txt_analisis.insert(tk.END, "• Monitorización continua requerida\n", "advertencia")
            
            # Identificar parámetros específicos en advertencia
            for detalle in detalles:
                if "ADVERTENCIA" in detalle:
                    param = detalle.split(":")[0].strip()
                    
                    if "Frecuencia Cardíaca" in param:
                        self.txt_analisis.insert(tk.END, "• Vigilar evolución de la frecuencia cardíaca\n")
                    
                    elif "Presión Arterial" in param:
                        if float(detalle.split(":")[1].split()[0]) > RANGOS_NORMALES[param][1]:
                            self.txt_analisis.insert(tk.END, "• Considerar ajuste en profundidad anestésica\n")
                        else:
                            self.txt_analisis.insert(tk.END, "• Considerar administración de fluidos\n")
                    
                    elif "SpO2" in param:
                        self.txt_analisis.insert(tk.END, "• Verificar vía aérea y oxigenación\n")
                    
                    elif "ETCO2" in param:
                        self.txt_analisis.insert(tk.END, "• Verificar parámetros ventilatorios\n")
            
            self.txt_analisis.insert(tk.END, "• Mantener vigilancia estrecha\n")
            
        else:  # Estado normal
            self.txt_analisis.insert(tk.END, "• Continuar monitorización estándar\n")
            self.txt_analisis.insert(tk.END, "• Mantener protocolo anestésico actual\n")
            self.txt_analisis.insert(tk.END, "• Registrar parámetros cada 5 minutos\n")
        
        # Configurar estilos de texto
        self.txt_analisis.tag_configure("titulo", font=("Arial", 12, "bold"), foreground="#2962FF")
        self.txt_analisis.tag_configure("subtitulo", font=("Arial", 10, "bold"))
        self.txt_analisis.tag_configure("estado_normal", foreground=COLORES_ESTADO["Normal"], font=("Arial", 10, "bold"))
        self.txt_analisis.tag_configure("estado_advertencia", foreground=COLORES_ESTADO["Advertencia"], font=("Arial", 10, "bold"))
        self.txt_analisis.tag_configure("estado_peligro", foreground=COLORES_ESTADO["Peligro"], font=("Arial", 10, "bold"))
        self.txt_analisis.tag_configure("peligro", foreground=COLORES_ESTADO["Peligro"])
        self.txt_analisis.tag_configure("advertencia", foreground=COLORES_ESTADO["Advertencia"])
        
        # Deshabilitar edición
        self.txt_analisis.config(state=tk.DISABLED)
    
    def ver_tendencias(self):
        """Muestra ventana con gráficos de tendencias a lo largo del tiempo"""
        if self.historial_datos.empty:
            messagebox.showwarning("Sin datos", "No hay datos suficientes para mostrar tendencias.")
            return
        
        # Crear ventana para tendencias
        ventana_tendencias = tk.Toplevel(self)
        ventana_tendencias.title("Tendencias - Sistema de Análisis para Anestesiólogos")
        ventana_tendencias.geometry("900x600")
        
        # Crear figura para gráficos
        fig = plt.Figure(figsize=(9, 8), dpi=100)
        
        # Seleccionar los parámetros más importantes para las tendencias
        parametros_tendencia = [
            "Frecuencia Cardíaca", "Presión Arterial Sistólica", "Presión Arterial Diastólica",
            "SpO2", "ETCO2", "BIS"
        ]
        
        # Crear subplots
        n_params = len(parametros_tendencia)
        axes = []
        for i in range(n_params):
            ax = fig.add_subplot(n_params, 1, i+1)
            axes.append(ax)
            
            # Configuración del subplot
            if parametros_tendencia[i] in self.historial_datos.columns:
                datos = self.historial_datos[parametros_tendencia[i]]
                
                # Crear eje x basado en timestamp o índice
                if "Timestamp" in self.historial_datos.columns:
                    x = pd.to_datetime(self.historial_datos["Timestamp"])
                else:
                    x = range(len(datos))
                
                # Graficar datos
                ax.plot(x, datos, label=parametros_tendencia[i])
                
                # Añadir líneas de rango normal
                min_val, max_val = RANGOS_NORMALES[parametros_tendencia[i]]
                ax.axhspan(min_val, max_val, color='g', alpha=0.1)
                
                # Configurar etiquetas y límites
                ax.set_ylabel(parametros_tendencia[i])
                ax.grid(True, linestyle="--", alpha=0.7)
                
                # Ajustar límites y para el último eje, mostrar etiqueta x
                if i == n_params - 1:
                    ax.set_xlabel("Tiempo")
        
        # Ajustar layout
        fig.tight_layout()
        
        # Añadir figura al canvas
        canvas = FigureCanvasTkAgg(fig, master=ventana_tendencias)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def generar_recomendaciones(self, devolver_texto=False):
        """Genera recomendaciones basadas en el estado actual del paciente"""
        if self.historial_datos.empty:
            if devolver_texto:
                return "No hay datos suficientes para generar recomendaciones."
            else:
                messagebox.showwarning("Sin datos", "No hay datos suficientes para generar recomendaciones.")
            return
        
        # Obtener último registro
        ultimo_registro = self.historial_datos.iloc[-1]
        
        # Evaluar estado
        estado, detalles = evaluar_estado_paciente(ultimo_registro)
        
        # Generar texto de recomendaciones
        texto_recomendaciones = f"Recomendaciones basadas en el estado actual: {estado}\n\n"
        
        if estado == "Peligro":
            texto_recomendaciones += "ACCIONES URGENTES REQUERIDAS:\n"
            
            # Identificar parámetros específicos en peligro
            parametros_peligro = []
            for detalle in detalles:
                if "PELIGRO" in detalle:
                    param = detalle.split(":")[0].strip()
                    parametros_peligro.append(param)
                    
                    if "Frecuencia Cardíaca" in param:
                        valor = float(detalle.split(":")[1].split()[0])
                        if valor > RANGOS_NORMALES[param][1]:
                            texto_recomendaciones += "• Taquicardia severa detectada. Considerar:\n"
                            texto_recomendaciones += "  - Administración de betabloqueantes\n"
                            texto_recomendaciones += "  - Verificar profundidad anestésica\n"
                            texto_recomendaciones += "  - Descartar hipertermia maligna\n"
                        else:
                            texto_recomendaciones += "• Bradicardia severa detectada. Considerar:\n"
                            texto_recomendaciones += "  - Administración de atropina\n"
                            texto_recomendaciones += "  - Verificar bloqueo vagal\n"
                            texto_recomendaciones += "  - Reducir dosis de opioides\n"
                    
                    elif "Presión Arterial" in param:
                        valor = float(detalle.split(":")[1].split()[0])
                        if valor > RANGOS_NORMALES[param][1]:
                            texto_recomendaciones += "• Hipertensión severa detectada. Considerar:\n"
                            texto_recomendaciones += "  - Aumentar profundidad anestésica\n"
                            texto_recomendaciones += "  - Administrar antihipertensivos\n"
                            texto_recomendaciones += "  - Verificar dolor o estimulación quirúrgica\n"
                        else:
                            texto_recomendaciones += "• Hipotensión severa detectada. Considerar:\n"
                            texto_recomendaciones += "  - Administrar fluidos\n"
                            texto_recomendaciones += "  - Usar vasopresores (efedrina, fenilefrina)\n"
                            texto_recomendaciones += "  - Reducir anestésicos inhalados\n"
                    
                    elif "SpO2" in param:
                        texto_recomendaciones += "• Desaturación severa detectada. Acciones inmediatas:\n"
                        texto_recomendaciones += "  - Verificar vía aérea\n"
                        texto_recomendaciones += "  - Aumentar FiO2 a 100%\n"
                        texto_recomendaciones += "  - Considerar ventilación manual\n"
                        texto_recomendaciones += "  - Verificar posición del tubo endotraqueal\n"
                    
                    elif "ETCO2" in param:
                        valor = float(detalle.split(":")[1].split()[0])
                        if valor > RANGOS_NORMALES[param][1]:
                            texto_recomendaciones += "• Hipercapnia severa detectada. Considerar:\n"
                            texto_recomendaciones += "  - Aumentar ventilación minuto\n"
                            texto_recomendaciones += "  - Verificar funcionamiento del ventilador\n"
                            texto_recomendaciones += "  - Descartar hipertermia maligna\n"
                        else:
                            texto_recomendaciones += "• Hipocapnia severa detectada. Considerar:\n"
                            texto_recomendaciones += "  - Reducir ventilación minuto\n"
                            texto_recomendaciones += "  - Verificar posible embolismo\n"
                            texto_recomendaciones += "  - Verificar hipotermia\n"
                    
                    elif "BIS" in param:
                        valor = float(detalle.split(":")[1].split()[0])
                        if valor < RANGOS_NORMALES[param][0]:
                            texto_recomendaciones += "• Excesiva profundidad anestésica. Considerar:\n"
                            texto_recomendaciones += "  - Reducir dosis de agentes anestésicos\n"
                        else:
                            texto_recomendaciones += "• Anestesia demasiado superficial. Considerar:\n"
                            texto_recomendaciones += "  - Aumentar dosis de anestésicos\n"
                            texto_recomendaciones += "  - Verificar riesgo de awareness\n"
            
            texto_recomendaciones += "\nACCIONES GENERALES:\n"
            texto_recomendaciones += "• Notificar inmediatamente al equipo quirúrgico\n"
            texto_recomendaciones += "• Preparar medicamentos de emergencia\n"
            texto_recomendaciones += "• Verificar accesos venosos\n"
            texto_recomendaciones += "• Considerar detener el procedimiento quirúrgico si es necesario\n"
            
        elif estado == "Advertencia":
            texto_recomendaciones += "ACCIONES RECOMENDADAS:\n"
            
            # Identificar parámetros específicos en advertencia
            for detalle in detalles:
                if "ADVERTENCIA" in detalle:
                    param = detalle.split(":")[0].strip()
                    
                    if "Frecuencia Cardíaca" in param:
                        valor = float(detalle.split(":")[1].split()[0])
                        if valor > RANGOS_NORMALES[param][1]:
                            texto_recomendaciones += "• Tendencia a taquicardia. Considerar:\n"
                            texto_recomendaciones += "  - Vigilar profundidad anestésica\n"
                            texto_recomendaciones += "  - Evaluar balance hídrico\n"
                        else:
                            texto_recomendaciones += "• Tendencia a bradicardia. Considerar:\n"
                            texto_recomendaciones += "  - Vigilar efectos de opioides\n"
                            texto_recomendaciones += "  - Tener atropina disponible\n"
                    
                    elif "Presión Arterial" in param:
                        valor = float(detalle.split(":")[1].split()[0])
                        if valor > RANGOS_NORMALES[param][1]:
                            texto_recomendaciones += "• Tendencia hipertensiva. Considerar:\n"
                            texto_recomendaciones += "  - Verificar analgesia adecuada\n"
                            texto_recomendaciones += "  - Evaluar profundidad anestésica\n"
                        else:
                            texto_recomendaciones += "• Tendencia hipotensiva. Considerar:\n"
                            texto_recomendaciones += "  - Evaluar requerimientos de fluidos\n"
                            texto_recomendaciones += "  - Ajustar dosis de anestésicos\n"
                    
                    elif "SpO2" in param:
                        texto_recomendaciones += "• Tendencia a desaturación. Considerar:\n"
                        texto_recomendaciones += "  - Aumentar FiO2\n"
                        texto_recomendaciones += "  - Verificar parámetros ventilatorios\n"
                    
                    elif "ETCO2" in param:
                        texto_recomendaciones += "• Alteración en ETCO2. Considerar:\n"
                        texto_recomendaciones += "  - Ajustar parámetros ventilatorios\n"
                        texto_recomendaciones += "  - Verificar metabolismo del paciente\n"
            
            texto_recomendaciones += "\nACCIONES GENERALES:\n"
            texto_recomendaciones += "• Mantener vigilancia estrecha\n"
            texto_recomendaciones += "• Registrar parámetros cada 3-5 minutos\n"
            texto_recomendaciones += "• Preparar intervención si los parámetros empeoran\n"
            
        else:  # Estado normal
            texto_recomendaciones += "El paciente se encuentra en condición estable.\n\n"
            texto_recomendaciones += "ACCIONES RECOMENDADAS:\n"
            texto_recomendaciones += "• Continuar monitorización estándar\n"
            texto_recomendaciones += "• Mantener protocolo anestésico actual\n"
            texto_recomendaciones += "• Registrar parámetros cada 5-10 minutos\n"
            texto_recomendaciones += "• Anticipar cambios según la fase quirúrgica\n"
        
        if devolver_texto:
            return texto_recomendaciones
        else:
            # Mostrar las recomendaciones en una ventana de diálogo
            dialogo = tk.Toplevel(self)
            dialogo.title("Recomendaciones")
            dialogo.geometry("600x500")
            
            # Área de texto para mostrar las recomendaciones
            txt_recomendaciones = scrolledtext.ScrolledText(dialogo, wrap=tk.WORD)
            txt_recomendaciones.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            txt_recomendaciones.insert(tk.END, texto_recomendaciones)
            txt_recomendaciones.config(state=tk.DISABLED)
            
            # Botón para cerrar
            ttk.Button(dialogo, text="Cerrar", command=dialogo.destroy).pack(pady=10)
    
    def configurar_simulacion(self):
        """Muestra ventana para configurar parámetros de simulación"""
        # Crear ventana de configuración
        ventana_config = tk.Toplevel(self)
        ventana_config.title("Configurar Simulación")
        ventana_config.geometry("400x300")
        ventana_config.resizable(False, False)
        
        # Frame principal
        frame = ttk.Frame(ventana_config, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Variables para configuración
        variacion_var = tk.DoubleVar(value=0.1)
        
        # Controles de configuración
        ttk.Label(frame, text="Variación de parámetros:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Scale(frame, from_=0.01, to=0.5, variable=variacion_var, length=200).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(frame, textvariable=tk.StringVar(value=f"{variacion_var.get():.2f}")).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(frame, text="Estado simulado:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        combo_estado = ttk.Combobox(frame, textvariable=self.estado_simulacion_var, values=["Normal", "Advertencia", "Peligro"], state="readonly", width=15)
        combo_estado.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Botones
        frame_botones = ttk.Frame(frame)
        frame_botones.grid(row=2, column=0, columnspan=3, pady=20)
        
        ttk.Button(frame_botones, text="Aceptar", command=ventana_config.destroy).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_botones, text="Cancelar", command=ventana_config.destroy).pack(side=tk.LEFT, padx=5)
    
    def simular_evento(self):
        """Simula un evento aleatorio como hipotensión, bradicardia, etc."""
        if self.historial_datos.empty:
            messagebox.showwarning("Sin datos", "Inicie la simulación antes de simular un evento.")
            return
        
        # Seleccionar un evento aleatorio
        eventos = [
            "Hipotensión", "Hipertensión", "Bradicardia", "Taquicardia",
            "Desaturación", "Hipocapnia", "Hipercapnia"
        ]
        evento = random.choice(eventos)
        
        # Generar datos según el evento
        if evento == "Hipotensión":
            nuevos_datos = generar_datos_simulacion(num_muestras=1)
            nuevos_datos.loc[0, "Presión Arterial Sistólica"] *= 0.7
            nuevos_datos.loc[0, "Presión Arterial Diastólica"] *= 0.7
            nuevos_datos.loc[0, "Presión Media"] *= 0.7
            mensaje = "Simulando evento: Hipotensión"
            
        elif evento == "Hipertensión":
            nuevos_datos = generar_datos_simulacion(num_muestras=1)
            nuevos_datos.loc[0, "Presión Arterial Sistólica"] *= 1.3
            nuevos_datos.loc[0, "Presión Arterial Diastólica"] *= 1.3
            nuevos_datos.loc[0, "Presión Media"] *= 1.3
            mensaje = "Simulando evento: Hipertensión"
            
        elif evento == "Bradicardia":
            nuevos_datos = generar_datos_simulacion(num_muestras=1)
            nuevos_datos.loc[0, "Frecuencia Cardíaca"] *= 0.6
            mensaje = "Simulando evento: Bradicardia"
            
        elif evento == "Taquicardia":
            nuevos_datos = generar_datos_simulacion(num_muestras=1)
            nuevos_datos.loc[0, "Frecuencia Cardíaca"] *= 1.4
            mensaje = "Simulando evento: Taquicardia"
            
        elif evento == "Desaturación":
            nuevos_datos = generar_datos_simulacion(num_muestras=1)
            nuevos_datos.loc[0, "SpO2"] = 88
            mensaje = "Simulando evento: Desaturación"
            
        elif evento == "Hipocapnia":
            nuevos_datos = generar_datos_simulacion(num_muestras=1)
            nuevos_datos.loc[0, "ETCO2"] = 25
            mensaje = "Simulando evento: Hipocapnia"
            
        elif evento == "Hipercapnia":
            nuevos_datos = generar_datos_simulacion(num_muestras=1)
            nuevos_datos.loc[0, "ETCO2"] = 55
            mensaje = "Simulando evento: Hipercapnia"
        
        # Añadir al historial
        self.historial_datos = pd.concat([self.historial_datos, nuevos_datos], ignore_index=True)
        
        # Actualizar interfaz
        self.actualizar_interfaz_con_datos(nuevos_datos.iloc[0])
        
        # Mostrar mensaje
        self.barra_estado.config(text=mensaje)
    
    def cambiar_estado_simulacion(self, event=None):
        """Cambia el estado de simulación según la selección del combobox"""
        nuevo_estado = self.estado_simulacion_var.get()
        self.barra_estado.config(text=f"Estado de simulación cambiado a: {nuevo_estado}")
    
    def inicializar_simulacion_corazon(self):
        """Inicializa las variables y recursos para la simulación del corazón"""
        # No es necesario inicializar estas variables aquí nuevamente
        # ya que ahora se inicializan en __init__
        
        # Crear SVG base del corazón
        self.svg_corazon_base = """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400" width="400" height="400">
            <!-- Fondo del corazón -->
            <path class="fondo-corazon" d="M200,390 C180,370 100,290 60,250 C20,210 0,170 0,130 C0,80 30,40 80,40 C120,40 160,70 200,110 C240,70 280,40 320,40 C370,40 400,80 400,130 C400,170 380,210 340,250 C300,290 220,370 200,390 Z" fill="#e74c3c" stroke="#c0392b" stroke-width="5" />
            
            <!-- Aurículas -->
            <path class="auricula-derecha" d="M80,120 C60,100 50,80 50,60 C50,40 70,20 90,20 C110,20 130,40 140,60 C150,80 150,100 140,120 Z" fill="#e74c3c" stroke="#c0392b" stroke-width="3" />
            <path class="auricula-izquierda" d="M320,120 C340,100 350,80 350,60 C350,40 330,20 310,20 C290,20 270,40 260,60 C250,80 250,100 260,120 Z" fill="#e74c3c" stroke="#c0392b" stroke-width="3" />
            
            <!-- Ventrículos -->
            <path class="ventriculo-izquierdo" d="M200,350 C190,340 130,280 100,250 C70,220 60,190 60,160 C60,130 80,100 120,100 C150,100 180,120 200,150 L200,350 Z" fill="#ff6b6b" stroke="#c0392b" stroke-width="0" />
            <path class="ventriculo-derecho" d="M200,350 C210,340 270,280 300,250 C330,220 340,190 340,160 C340,130 320,100 280,100 C250,100 220,120 200,150 L200,350 Z" fill="#ff6b6b" stroke="#c0392b" stroke-width="0" />
            
            <!-- Línea de separación entre ventrículos -->
            <path d="M200,150 L200,350" fill="none" stroke="#c0392b" stroke-width="3" />
            
            <!-- Arterias principales -->
            <path class="arteria-pulmonar" d="M150,80 C150,60 160,40 180,30 C200,20 230,20 240,40 C250,60 240,80 220,90" fill="none" stroke="#2e86de" stroke-width="8" />
            <path class="aorta" d="M250,80 C250,60 240,40 220,30 C200,20 170,20 160,40 C150,60 160,80 180,90" fill="none" stroke="#e74c3c" stroke-width="8" />
            
            <!-- Venas principales -->
            <path class="vena-cava-superior" d="M100,70 C80,60 70,40 80,20" fill="none" stroke="#3498db" stroke-width="6" />
            <path class="vena-cava-inferior" d="M90,120 C70,140 60,170 60,200" fill="none" stroke="#3498db" stroke-width="6" />
            <path class="venas-pulmonares" d="M300,70 C320,60 330,40 320,20 M310,120 C330,140 340,170 340,200" fill="none" stroke="#3498db" stroke-width="5" />
            
            <!-- Nodos y sistema de conducción -->
            <circle class="nodo-sa" cx="100" cy="80" r="8" fill="#f1c40f" />
            <circle class="nodo-av" cx="200" cy="160" r="8" fill="#f1c40f" />
            <path class="haz-his" d="M200,170 L200,220" fill="none" stroke="#f1c40f" stroke-width="3" />
            <path class="rama-derecha" d="M200,220 C180,240 160,270 150,300" fill="none" stroke="#f1c40f" stroke-width="2" />
            <path class="rama-izquierda" d="M200,220 C220,240 240,270 250,300" fill="none" stroke="#f1c40f" stroke-width="2" />
            
            <!-- Etiquetas -->
            <text x="90" y="175" font-family="Arial" font-size="12" fill="#333">VD</text>
            <text x="300" y="175" font-family="Arial" font-size="12" fill="#333">VI</text>
            <text x="80" y="90" font-family="Arial" font-size="12" fill="#333">AD</text>
            <text x="310" y="90" font-family="Arial" font-size="12" fill="#333">AI</text>
            
            <!-- Vasos coronarios -->
            <path class="coronaria-izquierda" d="M210,90 C230,100 250,120 260,140 C270,160 270,180 260,200" fill="none" stroke="#e74c3c" stroke-width="2" />
            <path class="coronaria-derecha" d="M190,90 C170,100 150,120 140,140 C130,160 130,180 140,200" fill="none" stroke="#e74c3c" stroke-width="2" />
        </svg>
        """
        
        # Crear SVGs animados para diferentes estados
        self.crear_svgs_estados()
        
        # Crear SVGs animados para diferentes estados
        self.crear_svgs_estados()
    
    def crear_svgs_estados(self):
        """Crea diferentes versiones del SVG del corazón para diferentes estados y frecuencias"""
        self.svg_corazon_normal = self.svg_corazon_base.replace('#e74c3c', '#e74c3c')  # Color normal
        self.svg_corazon_advertencia = self.svg_corazon_base.replace('#e74c3c', '#f39c12')  # Color amarillo
        self.svg_corazon_peligro = self.svg_corazon_base.replace('#e74c3c', '#c0392b')  # Color rojo oscuro
        self.svg_sistole = self.svg_corazon_base.replace('width="400" height="400"', 'width
