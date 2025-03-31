# Sistema de Análisis para Anestesiólogos

Un sistema avanzado de análisis y monitoreo para anestesiólogos que utiliza inteligencia artificial (CNN) para procesar, analizar y visualizar datos de pacientes durante procedimientos anestésicos. El sistema proporciona recomendaciones en tiempo real y visualizaciones interactivas para mejorar la toma de decisiones clínicas.

![Captura de pantalla del sistema](screenshots/dashboard.png)

## Características Principales

- **Monitorización en tiempo real** de parámetros vitales con actualizaciones visuales y alertas
- **Análisis mediante CNN** (Redes Neuronales Convolucionales) para predecir estados de riesgo
- **Simulación del corazón en vivo** con visualización anatómica detallada que responde a los parámetros del paciente
- **Sistema de recomendaciones inteligente** basado en la condición actual del paciente
- **Simulador de eventos críticos** para entrenamiento y educación médica
- **Generación de informes detallados** para documentación y análisis posterior
- **Interfaz gráfica intuitiva** con paneles configurables
- **Visualización de tendencias** de parámetros vitales a lo largo del tiempo

## Requisitos Técnicos

### Software
- Python 3.8 o superior
- Bibliotecas principales:
  - PyTorch >= 1.9.0
  - NumPy >= 1.20.0
  - Pandas >= 1.3.0
  - Matplotlib >= 3.4.0
  - OpenCV >= 4.5.0
  - Tkinter (incluido en la mayoría de las instalaciones de Python)
  - PIL (Pillow) >= 8.2.0

### Hardware Recomendado
- Procesador: Intel Core i5 o superior
- RAM: Mínimo 8GB (16GB recomendado)
- Espacio en disco: 500MB para la aplicación
- Pantalla: Resolución mínima 1366x768 (Full HD recomendado)

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/yourusername/sistema-anestesia-analytics.git
cd sistema-anestesia-analytics
```

2. Crear y activar un entorno virtual (opcional pero recomendado):
```bash
python -m venv venv
# En Windows
venv\Scripts\activate
# En macOS/Linux
source venv/bin/activate
```

3. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

4. Ejecutar la aplicación:
```bash
python app.py
```

## Uso Básico

### Inicio de la Aplicación
Al iniciar la aplicación, se presenta la interfaz principal con cuatro paneles:
- **Panel de Información del Paciente**: Datos básicos e identificación
- **Panel de Monitores en Tiempo Real**: Gráficos de los principales parámetros vitales
- **Panel de Parámetros Vitales**: Tabla detallada con indicadores visuales de estado
- **Panel de Análisis y Recomendaciones**: Texto con análisis automático y sugerencias

### Simulación de Datos
Para fines de prueba y entrenamiento, el sistema permite simular datos de pacientes:
1. Haga clic en "Iniciar Simulación" para generar datos sintéticos
2. Use "Configurar Simulación" para ajustar variables como frecuencia y variabilidad
3. "Simular Evento" genera situaciones críticas aleatorias como hipotensión o bradicardia

### Simulación del Corazón
Para activar la visualización anatómica interactiva:
1. Vaya a Simulación > Mostrar Simulación del Corazón
2. Observe cómo el corazón late con la frecuencia cardíaca actual
3. Vea cómo cambia el color según el estado del paciente (verde, amarillo o rojo)
4. El ECG simulado se actualiza en tiempo real

### Generación de Recomendaciones
Para obtener recomendaciones clínicas basadas en los datos:
1. Vaya a Análisis > Generar Recomendaciones
2. El sistema analizará los datos actuales y generará sugerencias personalizadas
3. Las recomendaciones varían según el estado (Normal, Advertencia, Peligro)

### Generación de Informes
Para generar un informe detallado:
1. Vaya a Archivo > Guardar Informe
2. Seleccione la ubicación para guardar el informe HTML
3. El informe incluye todos los parámetros, gráficos, estados y recomendaciones

## Estructura del Código

El sistema está organizado en varias clases y funciones principales:

- **`AplicacionAnestesiaAnalytics`**: Clase principal que gestiona la interfaz y la lógica de la aplicación
- **`ModeloCNNAnestesia`**: Implementación de la red neuronal para análisis de datos
- **`preprocesar_datos`**: Función para normalizar y preparar datos para el análisis
- **`generar_datos_simulacion`**: Genera datos sintéticos para simulaciones
- **`evaluar_estado_paciente`**: Analiza los parámetros vitales y determina el estado

### Archivos Principales
- **`app.py`**: Punto de entrada principal de la aplicación
- **`models.py`**: Define el modelo CNN y funciones relacionadas
- **`utils.py`**: Funciones de utilidad para procesamiento de datos
- **`config.py`**: Configuraciones y constantes del sistema

## Descripción Técnica del Modelo CNN

El sistema utiliza una red neuronal de arquitectura feed-forward para analizar los datos vitales del paciente:

```python
class ModeloCNNAnestesia(nn.Module):
    def __init__(self, input_features=11, output_classes=3):
        super(ModeloCNNAnestesia, self).__init__()
        
        # Arquitectura feed-forward
        self.fc1 = nn.Linear(input_features, 128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, output_classes)
        
    def forward(self, x):
        # Aprendizaje de características
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # Clasificación final (Normal, Advertencia, Peligro)
        x = self.fc3(x)
        
        return x
```

Este modelo analiza los parámetros vitales del paciente y clasifica su estado en tres categorías: Normal, Advertencia o Peligro, lo que permite intervenir de manera preventiva.

## Simulación Anatómica del Corazón

La simulación del corazón utiliza gráficos SVG para representar la anatomía cardíaca con detalles precisos:

- **Aurículas y ventrículos** diferenciados con colores anatómicos
- **Sistema de conducción eléctrica** (nodo SA, nodo AV, haz de His)
- **Vasos principales** (aorta, arteria pulmonar, venas cavas)
- **Animación dinámica** de sístole y diástole basada en frecuencia cardíaca real
- **Cambios de color** para reflejar el estado hemodinámico
- **ECG simulado** que refleja el estado eléctrico del corazón

La visualización anatómica proporciona una retroalimentación visual inmediata sobre el estado del paciente, complementando los datos numéricos y gráficos.

## Contribuciones

Las contribuciones son bienvenidas. Para contribuir:

1. Haga un fork del repositorio
2. Cree una rama para su característica (`git checkout -b feature/nueva-caracteristica`)
3. Realice sus cambios y haga commit (`git commit -m 'Añadir nueva característica'`)
4. Envíe a la rama (`git push origin feature/nueva-caracteristica`)
5. Abra un Pull Request

### Áreas para mejoras futuras
- Integración con sistemas hospitalarios reales mediante HL7/FHIR
- Implementación de algoritmos más avanzados (aprendizaje por refuerzo)
- Soporte para más dispositivos de monitoreo
- Integración con dispositivos de realidad aumentada
- Añadir más simulaciones anatómicas (pulmones, cerebro)

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - vea el archivo [LICENSE](LICENSE) para más detalles.

## Autor

Desarrollado por [Tu Nombre]

Para contacto o soporte: [tuemail@ejemplo.com]

---

©️ 2025 Sistema de Análisis para Anestesiólogos
