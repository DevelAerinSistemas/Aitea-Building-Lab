# Documentación del Proyecto AItea Building Lab

## Índice

1.  Introducción
    *   1.1. Propósito del Documento
    *   1.2. Resumen del Proyecto AItea Building Lab

2.  Instalación y Configuración
    *   2.1. Requisitos del Sistema
        *   2.1.1. Hardware
        *   2.1.2. Software (Debian, Python, InfluxDB, Nuitka, etc.)
    *   2.2. Configuración del Entorno Python
        *   2.2.1. Creación y Activación del Entorno Virtual
        *   2.2.2. Instalación de Dependencias (requirements.txt)
    

3.  Arquitectura del Proyecto
    *   3.1. Descripción General de la Arquitectura
    *   3.2. Componentes Principales
        *   3.2.1. Módulos y Paquetes
        *   3.2.2. Flujo de Datos
    *   3.3. Diagrama de Arquitectura (opcional)

4.  Componentes Clave
    *   4.1. Carga de Librerías (library_loader/so_library_loader.py)
        *   4.1.1. Descripción de la Clase SOLibraryLoader
        *   4.1.2. Carga Dinámica de Librerías
        *   4.1.3. Manejo de Dependencias
    *   4.2. Ejecución de Pipelines (pipelines/pipeline_executor.py)
        *   4.2.1. Descripción de la Clase PipelineExecutor
        *   4.2.2. Creación y Ejecución de Pipelines
        *   4.2.3. Planificación de Pipelines (pipe_plan.json)
    *   4.3. Modelos y Transformaciones (models_warehouse/)
        *   4.3.1. Descripción General de los Modelos y Transformaciones
        *   4.3.2. Ejemplos Específicos (TemperatureReachTransform, RoomTemperatureTransform, DataQualityAnalysis)
        *   4.3.3. Metodología de Desarrollo y Pruebas
    *   4.4. Utilidades (utils/)
        *   4.4.1. Descripción de las Utilidades (file_utils.py, logger_config.py, pipe_utils.py, so_utils.py)
        *   4.4.2. Funciones Clave y su Propósito

5.  Proceso de Testing
    *   5.1. Descripción del Proceso de Testing
    *   5.2. Generación de Datos de Prueba (testing_tools/testing_influxdb.py)
    *   5.3. Ejecución de Tests (testing_tools/testing_app.py, testing_tools/testing_demo.py)
    *   5.4. Pruebas Unitarias e Integración
    *   5.5. Resultados de las Pruebas y Métricas de Rendimiento

6.  Interfaz de Usuario (display/display.py)
    *   6.1. Descripción de la Interfaz de Usuario
    *   6.2. Componentes Principales (Streamlit)
    *   6.3. Funcionalidades (Selección de Librería, Configuración de Fechas, Visualización de Resultados)
    *   6.4. Guía de Uso

7.  Conclusiones
    *   7.1. Logros del Proyecto
    *   7.2. Posibles Mejoras y Extensiones Futuras

  



## 1.1. Propósito del Documento

Este documento tiene como objetivo proporcionar una descripción exhaustiva y detallada del proyecto "AItea Building Lab", desarrollado por Aerin S.L. El propósito principal es documentar la arquitectura, los componentes clave, los procesos de desarrollo y las pruebas realizadas en el proyecto. 

El documento servirá como referencia técnica para futuras mejoras, extensiones y mantenimiento del proyecto. Además, facilitará la comprensión del proyecto por parte de terceros, incluyendo auditores, nuevos miembros del equipo y colaboradores externos. Se detallarán los aspectos innovadores del proyecto, las metodologías empleadas y los resultados obtenidos, con el fin de evidenciar el valor añadido y el impacto potencial del "AItea Building Lab" en el ámbito de la eficiencia energética y la gestión inteligente de edificios.


## 1.2. Resumen del Proyecto AItea Building Lab

El proyecto "AItea Building Lab" es una iniciativa de I+D+i desarrollada por Aerin S.L. que tiene como objetivo principal la creación de una plataforma cuyo fin es elaborar y probar analíticas y modelos encaminados a analizar los datos generados por un edificio inteligente, optimizar consumos y detectar anómalias de funcionamiento. La plataforma se basa en el análisis de datos provenientes de diversas fuentes, como sensores IoT, sistemas de gestión de edificios (BMS) y datos meteorológicos. Aitea Builging Lab proporciona todo lo necesario para desarrollar algoritmos, de cualquier complejidad, incluidas las técnicas de deep learning más avanzadas. El resultado final es una librería .so, con el código ofuscado para mayor seguridad y preparado para ser ejecutada en producción. El proposito de la herramienta se centra en el desarrollo de modelos analíticos avanzados, utilizando técnicas de inteligencia artificial y machine learning, para predecir el comportamiento energético de los edificios, detectar anomalías y optimizar el funcionamiento de los sistemas de climatización, iluminación y ventilación. Además, se ha desarrollado una interfaz de usuario con el fin de tyestear los resultados con el fín de dejarlos listos para llevarlos a producción.


Una característica clave de "AItea Building Lab" es su diseño modular, que permite la fácil integración de nuevos modelos y analíticas, así como su adaptación a diferentes tipos de edificios y entornos. Además, la plataforma permite automatizar los entrenamientos de los modelos utilizando computación paralela, lo que reduce significativamente los tiempos de desarrollo y mejora la eficiencia en la generación de soluciones personalizadas.


## 2.1. Requisitos del Sistema

Para la correcta ejecución y funcionamiento del proyecto "AItea Building Lab", se requiere cumplir con los siguientes requisitos de sistema, tanto a nivel de hardware como de software.

### 2.1.1. Hardware

Los reausitos de hardware no son excesivamente elevados para realizar algoritmos que no requieran grandes cantidades de datos y que seran sencillos, no obstante para no encontrar problemas a la hora de implementar algoritmos complejos se recomienda:

*   **Procesador:** Se recomienda un procesador multi-core (mínimo 4 núcleos) para soportar la ejecución de modelos de machine learning y la computación paralela. Intel Core i5 o equivalente de AMD.Si se desea desarrollar modelos de deep learning se recomienda una GPU capaz de ejecutar instrucciones CUDA. 
*   **Memoria RAM:** Se requiere un mínimo de 8 GB de RAM para garantizar un rendimiento adecuado para algoritmos sencillos, 16 Gb son recomendados para entrenar modelos que requieran grandes dataframes de datos. Se recomienda 32 GB para cargas de trabajo más intensivas.
*   **Almacenamiento:** Se necesita un espacio de almacenamiento de al menos 500 GB para alojar el sistema operativo, las dependencias del proyecto, los datos de entrenamiento y los modelos generados. Se recomienda el uso de un SSD para mejorar la velocidad de acceso a los datos.
*   **Conexión a Internet:** Se requiere una conexión a Internet estable para la descarga de dependencias, la conexión a la base de datos InfluxDB (si está alojada en la nube) y el acceso a recursos externos.

### 2.1.2. Software

*   **Sistema Operativo:** Debian GNU/Linux (o una distribución similar) es el sistema operativo recomendado para el despliegue del proyecto.
*   **Python:** Se requiere Python 3.11 o superior.
*   **InfluxDB:** Aúnque no es necesario, los algoritmos pueden simplemente optener los datos de un fichero, se recomienda una instancia de InfluxDB para el almacenamiento y la gestión de los datos.
*   **Dependencias de Python:** Las dependencias del proyecto se gestionan a través de `pip` y se listan en el archivo [config/requirements.txt](config/requirements.txt). Entre las dependencias más importantes se encuentran:
    *   `dill==0.3.8`
    *   `dotenv==0.9.9`
    *   `influxdb-client==1.44.0`
    *   `joblib==1.4.2`
    *   `loguru==0.7.2`
    *   `numpy==2.0.1`
    *   `onnx==1.16.2`
    *   `onnxconverter-common==1.14.0`
    *   `onnxruntime==1.19.2`
    *   `pandas==2.2.2`
    *   `pillow==11.2.1`
    *   `plotly==6.0.1`
    *   `pythermalcomfort==3.2.0`
    *   `python-dotenv==1.0.1`
    *   `pytz==2024.1`
    *   `scikit-learn==1.5.1`
    *   `scipy==1.14.1`
    *   `skl2onnx==1.17.0`
    *   `streamlit==1.45.1`
    *   `Nuitka`
*   **Nuitka:** Se utiliza Nuitka para la compilación y ofuscación del código Python.
*   
Entre estas dependencias no se citan librerías para el desarrollo de algoritmos de deep learning, ya que el desarrollador puede elegir la que más de agrade (Torch, TensorFlow, etc.). 



## 2.2. Configuración del Entorno Python

Para asegurar la correcta ejecución del proyecto, es fundamental configurar un entorno Python aislado que gestione las dependencias de manera eficiente. A continuación, se describen los pasos para crear y configurar el entorno Python.

### 2.2.1. Creación y Activación del Entorno Virtual

Se recomienda utilizar `venv` para crear un entorno virtual. Los siguientes comandos crean y activan un entorno virtual en el directorio del proyecto:

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Arquitectura del Proyecto

La arquitectura del proyecto "AItea Building Lab" se ha diseñado siguiendo un enfoque modular y flexible, permitiendo la fácil integración de nuevos componentes y la adaptación a diferentes entornos y requisitos. A continuación, se describe la arquitectura general del proyecto, sus componentes principales y el flujo de datos.

### 3.1. Descripción General de la Arquitectura

El proyecto "AItea Building Lab" se estructura en varias capas lógicas, cada una con responsabilidades específicas:

1.  **Capa de Adquisición de Datos:** Se presupone la exitencia de  uan capa de adquisición cuyo fin es la recolección de datos provenientes de diversas fuentes, como sensores IoT, sistemas de gestión de edificios (BMS) y APIs externas (datos meteorológicos, etc.). Se proporciona un conector de Influx por lo que el software podría usarse de forma inmediata si los datos se disponene en Influx, en otro caso el usuario puede usar un archivo con formato csv. 
2.  **Capa de Procesamiento de Datos:** Esta capa se encarga de la fusión de los datos, de la limpieza y si se considera, de la transformación y enriquecimiento de los datos. El desarrollador puede incoporar la técnica que precise para incorporarla en la tubería como una o como varias transformaciones. 
3.  **Capa de Modelado y Analítica:** Esta capa contiene los modelos de machine learning y las analíticas que se utilizan para predecir el comportamiento energético de los edificios, detectar anomalías y optimizar el funcionamiento de los sistemas o cualquier otra función que se quiere añadir.
4.  **Capa de Ejecución:** Esta capa se encarga de la compilación y ofuscación del código Python mediante Nuitka, generando librerías .so que pueden ser ejecutadas en producción.
5.  **Capa de Visualización:** Esta capa proporciona una interfaz de usuario (desarrollada con Streamlit) que permite a los usuarios monitorizar el coportamiento de las analíticas como si estuviesen en producción.

### 3.2. Componentes Principales

El proyecto "AItea Building Lab" se compone de varios módulos y paquetes, cada uno con una función específica:

*   **database_tools:** Contiene el módulo [`influxdb_connector.py`](database_tools/influxdb_connector.py), que proporciona la funcionalidad para conectarse a la base de datos InfluxDB, realizar consultas y almacenar datos.
*   **library_loader:** Contiene el módulo [`so_library_loader.py`](library_loader/so_library_loader.py), que se encarga de cargar dinámicamente las librerías .so generadas por Nuitka.
*   **pipelines:** Contiene el módulo [`pipeline_executor.py`](pipelines/pipeline_executor.py), que permite crear y ejecutar pipelines de procesamiento de datos y modelado.
*   **models_warehouse:** Contiene los modelos de machine learning y las transformaciones que se utilizan para analizar los datos. Ejemplos: [`TemperatureReachTransform`](models_warehouse/TemperatureReachTransform.py), [`RoomTemperatureTransform`](models_warehouse/RoomTemperatureTransform.py), [`DataQualityAnalysis`](models_warehouse/data_quality_analysis.py).
*   **utils:** Contiene utilidades generales, como funciones para la gestión de archivos ([`file_utils.py`](utils/file_utils.py)), la configuración del logger ([`logger_config.py`](utils/logger_config.py)), la gestión de pipelines ([`pipe_utils.py`](utils/pipe_utils.py)) y la creación de librerías .so ([`so_utils.py`](utils/so_utils.py)).
*   **display:** Contiene el código de la interfaz de usuario ([`display.py`](display/display.py)), desarrollada con Streamlit.

### 3.3. Flujo de Datos

El flujo de datos en el proyecto "AItea Building Lab" sigue los siguientes pasos:

1.  **Adquisición de Datos:** Los datos se recolectan de diversas fuentes y se envían a la capa de almacenamiento.
2.  **Almacenamiento de Datos:** Los datos se almacenan en InfluxDB como series temporales.
3.  **Procesamiento de Datos:** Los datos se extraen de InfluxDB, se limpian, se transforman y se enriquecen.
4.  **Modelado y Analítica:** Se aplican modelos de machine learning y analíticas para predecir el comportamiento energético, detectar anomalías y optimizar el funcionamiento de los sistemas.
5.  **Visualización:** Los resultados de los modelos y las analíticas se visualizan en la interfaz de usuario.

### 3.4. Diagrama de Arquitectura

A continuación se presenta un diagrama que ilustra la arquitectura y el flujo de trabajo del "AItea Building Lab". El diagrama se divide en dos flujos principales: el **Flujo de Desarrollo y Compilación** (cómo un desarrollador crea una nueva analítica) y el **Flujo de Testing y Visualización** (cómo se prueba y se visualiza el resultado de esa analítica).

El resultado final es una librería (.so) que una vez testeada puede exportarse al entorno Aitea Braín Lite (o a cualquier otro entorno que el desarrollador personalice) capaz de analizar los datos de un edificio según la lógica dada por data science. 

![alt text](image.png)

## 4. Componentes Clave

Esta sección describe en detalle los módulos y paquetes más importantes que conforman el núcleo funcional del proyecto "AItea Building Lab".

### 4.1. Carga de Librerías (`library_loader/so_library_loader.py`)

Este componente es fundamental para la modularidad del sistema, ya que permite cargar y ejecutar las analíticas compiladas de forma dinámica.

#### 4.1.1. Descripción de la Clase `SOLibraryLoader`

La clase [`SOLibraryLoader`](library_loader/so_library_loader.py) es la responsable de interactuar con las librerías `.so` generadas. Sus funciones principales son:

*   **Cargar dinámicamente** una librería `.so` específica basándose en su nombre.
*   Actuar como **puente entre la interfaz de usuario y la analítica compilada**, permitiendo invocar sus métodos.
*   Gestionar la **obtención de datos** desde múltiples fuentes (archivos locales, InfluxDB, PostgreSQL) para alimentar las pruebas.
*   Extraer **información y metadatos** de la librería cargada, como su versión, fecha de creación y parámetros internos.

#### 4.1.2. Carga Dinámica de Librerías

El sistema utiliza la biblioteca `importlib` de Python para la carga dinámica. Cuando el usuario selecciona una analítica en la interfaz, `SOLibraryLoader` importa el módulo `.so` correspondiente. Este mecanismo desacopla la aplicación principal de las analíticas, permitiendo añadir, eliminar o actualizar modelos sin necesidad de modificar el código del cargador o de la interfaz. Simplemente con compilar un nuevo modelo y colocar el archivo `.so` en el directorio `libs/`, este estará disponible automáticamente para su uso.

#### 4.1.3. Manejo de Dependencias

El cargador está diseñado para ser robusto frente a la ausencia de dependencias opcionales. Como se observa en el código de [`so_library_loader.py`](library_loader/so_library_loader.py), los conectores de bases de datos (`AITEA_CONNECTORS`) se importan dentro de un bloque `try-except`. Si estos conectores no están instalados en el entorno, el sistema no falla; en su lugar, deshabilita la funcionalidad de conexión a bases de datos y opera en un modo limitado que solo permite cargar datos desde archivos locales.

### 4.2. Ejecución de Pipelines (`pipelines/pipeline_executor.py`)

Este componente está diseñado para orquestar flujos de trabajo complejos de procesamiento y entrenamiento de modelos, aunque su uso principal en el "Lab" es dentro de las propias analíticas.

#### 4.2.1. Descripción de la Clase `PipelineExecutor`

El [`PipelineExecutor`](pipelines/pipeline_executor.py) está basado en la clase `Pipeline` de `scikit-learn`. Su propósito es encadenar una secuencia de pasos de transformación de datos y un estimador final (modelo) en un único objeto. Esto simplifica el flujo de trabajo, ya que todo el proceso (desde la limpieza de datos hasta la predicción) puede ser ejecutado con una sola llamada.

#### 4.2.2. Creación y Ejecución de Pipelines

Un pipeline se define como una lista de tuplas, donde cada tupla contiene un nombre para el paso y una instancia del transformador o modelo. El `PipelineExecutor` toma esta lista y la configura. Al ejecutar el pipeline, los datos pasan secuencialmente a través de cada paso de transformación antes de llegar al modelo final para el entrenamiento o la predicción.

#### 4.2.3. Planificación de Pipelines (`pipe_plan.json`)

El archivo [`pipes_schedules/pipe_plan.json`](pipes_schedules/pipe_plan.json) es un ejemplo de cómo se puede definir declarativamente un pipeline. Este archivo JSON especifica:
*   **Los pasos del pipeline:** Qué clases de transformación o modelo se deben usar.
*   **Los parámetros para cada paso:** La configuración específica para cada modelo o transformador.
*   **La consulta de datos para el entrenamiento:** Define de dónde y cómo obtener los datos necesarios (qué `buckets` de InfluxDB, rangos de fechas, filtros, etc.).

Este enfoque permite modificar los flujos de trabajo sin cambiar el código, simplemente editando el archivo de planificación.

### 4.3. Modelos y Transformaciones (`models_warehouse/`)

Este directorio es el corazón del proyecto, donde reside toda la lógica de negocio y la inteligencia analítica.

#### 4.3.1. Descripción General de los Modelos y Transformaciones

El directorio [`models_warehouse/`](models_warehouse/) contiene todos los algoritmos y analíticas desarrollados. Cada archivo `.py` representa una capacidad específica (ej. análisis de calidad de datos, predicción de consumo). La práctica estándar es definir una clase por archivo que hereda de una clase base (`MetaModel`), implementando una interfaz común con métodos como `fit`, `predict` y/o `transform`. Esto asegura la consistencia y la interoperabilidad entre los diferentes componentes.

#### 4.3.2. Ejemplos Específicos

*   [`DataQualityAnalysis`](models_warehouse/data_quality_analysis.py): Implementa algoritmos para detectar anomalías en los datos, como valores atípicos (outliers), datos faltantes o problemas de frecuencia en la recepción de datos.
*   **TemperatureReachTransform**: (Ejemplo) Una transformación que podría calcular el tiempo que tarda una zona en alcanzar una temperatura de consigna.
*   **RoomTemperatureTransform**: (Ejemplo) Un modelo que podría predecir la temperatura futura de una sala basándose en datos históricos y variables externas.

#### 4.3.3. Metodología de Desarrollo y Pruebas

El ciclo de vida de una nueva analítica es el siguiente:
1.  **Desarrollo:** Un desarrollador crea un nuevo archivo Python en `models_warehouse/`, definiendo la clase y la lógica de la analítica.
2.  **Compilación:** Se utiliza el script `utils/so_utils.py` para compilar el nuevo archivo Python en una librería `.so` ofuscada usando Nuitka.
3.  **Prueba:** Se inicia la interfaz de Streamlit (`display/display.py`), se selecciona la nueva librería y se ejecuta contra datos reales o de prueba para validar su comportamiento y resultados.
4.  **Iteración:** Se repiten los pasos anteriores hasta que la analítica funcione como se espera y esté lista para ser desplegada en un entorno de producción.

### 4.4. Utilidades (`utils/`)

Este paquete contiene módulos de soporte que son utilizados transversalmente en todo el proyecto.

#### 4.4.1. Descripción de las Utilidades

*   [`file_utils.py`](utils/file_utils.py): Proporciona funciones para leer y escribir archivos, como configuraciones JSON.
*   [`logger_config.py`](utils/logger_config.py): Configura un sistema de logging centralizado usando la librería `Loguru`, que permite un registro de eventos detallado y con formato en toda la aplicación.
*   [`pipe_utils.py`](utils/pipe_utils.py): Contiene funciones auxiliares para trabajar con los pipelines.
*   [`so_utils.py`](utils/so_utils.py): Es el script clave que orquesta el proceso de compilación con Nuitka, tomando un archivo `.py` de `models_warehouse` y convirtiéndolo en una librería `.so` en el directorio `libs`.

#### 4.4.2. Funciones Clave y su Propósito

La función más crítica en este paquete es `create_so` dentro de [`so_utils.py`](utils/so_utils.py). Esta función automatiza la invocación de Nuitka con todos los parámetros necesarios, como incluir las dependencias correctas, especificar el directorio de salida y gestionar los archivos temporales. Esto simplifica enormemente el proceso de compilación para el desarrollador.

## 5. Proceso de Testing

El proceso de testing en "AItea Building Lab" es una fase crucial que garantiza la calidad, robustez y fiabilidad de cada analítica antes de su despliegue. La plataforma proporciona herramientas tanto para pruebas interactivas y visuales como para tests de integración automatizados.

### 5.1. Descripción del Proceso de Testing

El testing se aborda desde dos perspectivas complementarias:

1.  **Testing Interactivo y Visual:** Realizado principalmente a través de la interfaz de usuario de Streamlit ([`display/display.py`](display/display.py)). Este enfoque está pensado para el científico de datos durante la fase de desarrollo. Permite cargar una librería `.so` recién compilada, ejecutarla contra un conjunto de datos (real o de prueba) y visualizar los resultados (tablas, gráficos) de forma inmediata. Es fundamental para la depuración, el ajuste fino de parámetros y la validación cualitativa del comportamiento del modelo.

2.  **Testing de Integración:** Se lleva a cabo mediante scripts ubicados en el directorio [`testing_tools/`](testing_tools/). Estos scripts, como [`testing_app.py`](testing_tools/testing_app.py), están diseñados para ejecutar el flujo de trabajo completo de principio a fin: desde la lectura de la configuración y el plan del pipeline, pasando por la obtención de datos, el entrenamiento del modelo y la compilación final a una librería `.so`. Su objetivo es asegurar que todos los componentes del sistema interactúan correctamente y que el proceso es reproducible.

### 5.2. Herramientas de Soporte para Testing (`testing_tools/testing_databases.py`)

Para facilitar la verificación y preparación de los datos de prueba, el proyecto incluye scripts de soporte. El módulo [`testing_tools/testing_databases.py`](testing_tools/testing_databases.py) proporciona funciones para conectar y ejecutar consultas directamente contra las bases de datos configuradas (InfluxDB y PostgreSQL).

Estas funciones, como `influx_query_test`, permiten al desarrollador inspeccionar rápidamente el estado de los datos, verificar los resultados de una inserción o preparar un escenario específico antes de lanzar una prueba completa, sin necesidad de utilizar una herramienta de base de datos externa.

### 5.3. Ejecución de Tests (`testing_tools/testing_app.py`, `testing_tools/testing_demo.py`)

El directorio `testing_tools/` contiene los scripts para la ejecución de pruebas de integración.

*   [`testing_app.py`](testing_tools/testing_app.py): Es el script principal de prueba de integración. Su flujo de trabajo consiste en:
    1.  Cargar la configuración del entorno y la ruta al plan de pipelines (`pipe_plan.json`).
    2.  Instanciar el [`PipelineExecutor`](pipelines/pipeline_executor.py).
    3.  Ejecutar el método `pipes_executor`, que orquesta todo el proceso de entrenamiento y compilación para las analíticas definidas en el plan.
    El propósito de este script es verificar que el pipeline completo se ejecuta sin errores, sirviendo como una prueba de humo ("smoke test") para la configuración y el código base.

*   [`testing_demo.py`](testing_tools/testing_demo.py): Es un script similar a `testing_app.py`, pero puede estar configurado para ejecutar un subconjunto específico de pruebas o utilizar una configuración de demostración. Sirve como un ejemplo práctico y un punto de partida para crear nuevos scripts de prueba personalizados.

### 5.4. Pruebas Unitarias e Integración

*   **Pruebas de Integración:** Como se ha descrito, este es el enfoque principal de las pruebas automatizadas en el proyecto. Los scripts en `testing_tools/` validan que los diferentes módulos (`PipelineExecutor`, `SOLibraryLoader`, los conectores de datos y los propios modelos) funcionan correctamente en conjunto.

*   **Pruebas Unitarias:** Aunque el proyecto no incluye un framework formal de pruebas unitarias (como Pytest o Unittest) en su estado actual, la arquitectura modular está diseñada para facilitarlas. Las funciones y métodos dentro de los modelos en `models_warehouse/` y las utilidades en `utils/` son, en su mayoría, autocontenidos y pueden ser instanciados y probados de forma aislada. La implementación de pruebas unitarias formales se considera una posible mejora futura.

### 5.5. Resultados de las Pruebas y Métricas de Rendimiento

La validación de los resultados se realiza en dos niveles:

1.  **Validación Cualitativa:** Se realiza a través de la interfaz de Streamlit. El desarrollador analiza visualmente los gráficos y tablas generados para confirmar que la salida del modelo es coherente y se alinea con las expectativas.

2.  **Métricas de Rendimiento:** Durante la ejecución en la interfaz, se mide y muestra el **tiempo de ejecución** de la analítica. Esta métrica es un indicador clave del rendimiento y la eficiencia del código compilado, permitiendo al desarrollador identificar cuellos de botella y optimizar el rendimiento antes del despliegue en producción. Métricas más específicas del modelo (como precisión, F1-score, etc.) pueden ser implementadas dentro de la lógica de cada analítica y mostradas en la interfaz según sea necesario.
3.  

## 6. Interfaz de Usuario (`display/display.py`)

La interfaz de usuario es el componente central para la fase de pruebas y validación del "AItea Building Lab". Proporciona un entorno interactivo y visual que permite a los desarrolladores y científicos de datos interactuar con las analíticas compiladas de una manera ágil y eficiente.

### 6.1. Descripción de la Interfaz de Usuario

La interfaz es una aplicación web desarrollada con la biblioteca **Streamlit**. Su diseño está enfocado en la simplicidad y la funcionalidad, permitiendo al usuario realizar un ciclo completo de prueba (selección de analítica, configuración de datos, ejecución y visualización de resultados) en una única pantalla.

El propósito principal de esta interfaz no es ser un producto final para un cliente, sino una herramienta de laboratorio (`Lab`) para el equipo de desarrollo. Permite simular la ejecución de una analítica en un entorno controlado, utilizando datos históricos para verificar su comportamiento antes de que sea desplegada en un sistema de producción como Aitea Brain Lite.

### 6.2. Componentes Principales (Streamlit)

La interfaz se construye utilizando una combinación de los siguientes componentes de Streamlit:

*   **`st.title` y `st.markdown`**: Para estructurar la página y mostrar texto informativo, como los títulos, descripciones y los metadatos de la librería seleccionada.
*   **`st.selectbox`**: Es el elemento clave que permite al usuario seleccionar la librería (`.so`) que desea probar de una lista desplegable. Esta lista se genera dinámicamente escaneando el directorio `libs/`.
*   **`st.radio`**: Se utiliza para ofrecer opciones claras y excluyentes, como la selección de la fuente de datos (local, InfluxDB, etc.) o el formato de visualización de los resultados (Tabla o Gráfico).
*   **`st.date_input` y `st.time_input`**: Permiten al usuario definir con precisión el rango temporal (fecha y hora de inicio y fin) para la consulta de datos de prueba.
*   **`st.button`**: El botón "Execute Testing" que inicia todo el proceso de prueba una vez que se ha configurado.
*   **`st.plotly_chart`**: Para renderizar los resultados en forma de gráficos interactivos, utilizando la biblioteca Plotly. Esto permite al usuario hacer zoom, desplazarse y examinar en detalle los datos de salida.
*   **`st.dataframe`**: Para mostrar los resultados en un formato tabular.
*   **`st.session_state`**: Se utiliza internamente para gestionar el estado de la aplicación entre las interacciones del usuario, evitando que la aplicación se recargue o ejecute código innecesariamente al cambiar un widget.

### 6.3. Funcionalidades

La interfaz de usuario ofrece las siguientes funcionalidades clave:

1.  **Descubrimiento y Selección de Analíticas**: La aplicación detecta automáticamente todas las librerías `.so` disponibles en el directorio `libs` y las presenta en un menú desplegable.
2.  **Configuración de la Fuente de Datos**: Permite elegir entre diferentes fuentes de datos para la prueba, como archivos locales (CSV, Parquet) o bases de datos (InfluxDB, PostgreSQL), siempre que los conectores estén disponibles.
3.  **Selección de Rango Temporal**: El usuario tiene control total sobre el período de datos que se utilizará para la prueba, lo que es esencial para validar el comportamiento del modelo en diferentes escenarios (ej. verano vs. invierno, día vs. noche).
4.  **Ejecución bajo demanda**: La analítica solo se ejecuta cuando el usuario pulsa el botón, lo que permite configurar todos los parámetros tranquilamente antes de iniciar el proceso.
5.  **Visualización Dual (Tabla/Gráfico)**: El usuario puede cambiar fácilmente entre una vista de tabla, útil para inspeccionar valores exactos, y una vista de gráfico, ideal para identificar tendencias, patrones y anomalías visualmente.
6.  **Información de Metadatos y Rendimiento**: Antes de la ejecución, la interfaz muestra información relevante sobre la librería seleccionada. Después de la ejecución, informa del tiempo total que ha tardado el proceso, una métrica clave para evaluar la eficiencia de la analítica.

### 6.4. Guía de Uso

El flujo de trabajo típico para un desarrollador que utiliza la interfaz es el siguiente:

1.  **Lanzar la Aplicación**: Ejecutar el comando `streamlit run display/display.py` en la terminal desde el directorio raíz del proyecto.
2.  **Seleccionar la Analítica**: Elegir la librería `.so` que se acaba de compilar o que se desea probar en el menú desplegable "Select a library".
3.  **Configurar los Datos**:
    *   Seleccionar la fuente de datos (ej. "influxdb").
    *   Establecer las fechas y horas de inicio y fin para el conjunto de datos de prueba.
4.  **Elegir Visualización**: Seleccionar si los resultados se mostrarán como "Table" o "Graph".
5.  **Ejecutar la Prueba**: Hacer clic en el botón "Execute Testing".
6.  **Analizar los Resultados**: Observar la salida que aparece en la parte inferior de la página. Inspeccionar los gráficos en busca de comportamientos esperados o anómalos, o revisar los datos brutos en la tabla. Medir el tiempo de ejecución para asegurar que cumple los requisitos de rendimiento.


## 7. Conclusiones

El proyecto "AItea Building Lab" ha culminado con éxito, estableciendo una plataforma robusta y flexible para el desarrollo, prueba y compilación de analíticas (que pueden incluir cualquier tipo de algoritmo de ML) de datos orientadas a la gestión inteligente de edificios. Esta sección final resume los principales logros, su alineación con los objetivos de I+D y las posibles vías de trabajo futuro.

### 7.1. Logros del Proyecto

Los logros más significativos del proyecto se pueden resumir en los siguientes puntos:

1.  **Creación de un Entorno de Desarrollo Integral:** Se ha desarrollado un ecosistema completo que cubre todo el ciclo de vida de una analítica, desde la escritura del código en Python hasta la generación de una librería binaria (`.so`) lista para producción. Esto estandariza y acelera significativamente el proceso de desarrollo.

2.  **Arquitectura Modular y Extensible:** Gracias a su diseño modular, la plataforma permite la fácil incorporación de nuevos modelos, transformaciones y fuentes de datos. La carga dinámica de librerías es un pilar de esta flexibilidad, permitiendo que el sistema evolucione sin necesidad de modificar sus componentes centrales.

3.  **Seguridad y Optimización mediante Compilación:** La integración de Nuitka para compilar el código Python a C no solo ofrece una capa de ofuscación que protege la propiedad intelectual de los algoritmos, sino que también abre la puerta a optimizaciones de rendimiento, haciendo las analíticas más eficientes para su ejecución en entornos con recursos limitados.

4.  **Plataforma de Prototipado y Testing Rápido:** La interfaz de usuario desarrollada con Streamlit ha demostrado ser una herramienta de laboratorio (Lab) de incalculable valor. Permite a los científicos de datos validar visual e interactivamente sus hipótesis y modelos contra datos reales, reduciendo drásticamente los ciclos de depuración y ajuste.

5.  **Automatización del Flujo de Trabajo:** Se han sentado las bases para la automatización de procesos complejos, como el entrenamiento de modelos a partir de planes declarativos (ej. `pipe_plan.json`) y la compilación automatizada, minimizando la intervención manual y el riesgo de errores.

### 7.2. Cumplimiento de los Objetivos de I+D

El proyecto "AItea Building Lab" cumple los siguientes objetivos de I+D:

*   **Innovación de Proceso:** La plataforma representa una innovación fundamental en el *proceso* de creación y despliegue de soluciones de analíticas. Se ha creado una metodología y un conjunto de herramientas que optimizan un flujo de trabajo que tradicionalmente era artesanal y propenso a errores y que se realizan por lo general en ad-hoc.

*   **Investigación Aplicada:** "Aitea Building Lab es, en esencia, una herramienta de investigación aplicada. Facilita la experimentación con algoritmos avanzados (incluyendo machine learning y deep learning) en un entorno controlado pero realista, acortando la brecha entre la investigación teórica y la aplicación práctica.

*   **Generación de Conocimiento y Activos Tecnológicos:** El resultado del proyecto no es solo un software, sino un activo tecnológico estratégico. Las librerías `.so` generadas son activos reutilizables y desplegables. Además, el conocimiento adquirido sobre la compilación de modelos de IA y la creación de flujos de trabajo MLOps es un valor intrínseco para la empresa. 

*   **Mejora de la Competitividad:** Al acelerar el desarrollo y aumentar la fiabilidad de las analíticas, la plataforma permite a Aerin S.L. responder más rápidamente a las necesidades del mercado y ofrecer soluciones más sofisticadas y robustas, mejorando así su posición competitiva.

### 7.3. Posibles Mejoras y Extensiones Futuras

El "AItea Building Lab" es una base sólida sobre la cual se pueden construir numerosas mejoras y extensiones. Las líneas de trabajo futuro más prometedoras incluyen:

*   **Formalización de Pruebas Unitarias:** Integrar un framework como `Pytest` para añadir una capa de pruebas unitarias automatizadas que complementen las actuales pruebas de integración y visuales, aumentando aún más la fiabilidad del código.

*   **Registro y Versionado de Modelos:** Incorporar un sistema de registro de modelos (similar a MLflow Registry) para gestionar las diferentes versiones de las librerías `.so` generadas, sus parámetros y sus métricas de rendimiento, facilitando la trazabilidad y la gobernanza.

*   **Expansión de la Biblioteca de Modelos:** Continuar desarrollando y añadiendo nuevas analíticas pre-construidas en `models_warehouse` para cubrir un espectro más amplio de casos de uso en la gestión de edificios (ej. predicción de ocupación, optimización de la carga de baterías, etc.).

*   **Visualizaciones Avanzadas:** Mejorar la interfaz de usuario con opciones de visualización más avanzadas, como mapas de calor, análisis de correlación interactivos o comparativas automáticas entre diferentes ejecuciones de un modelo.

*   **Integración con Orquestadores de Flujo de Trabajo:** Explorar la integración con herramientas como Airflow o Prefect para gestionar la ejecución programada y condicional de los pipelines de entrenamiento y compilación a gran escala.
  
*   **Añadir la capacidad de usar dash para sustentar los datos:** En un futuro pueden requerirse calculos con cantidades de datos excesivamente grandes, lo que puede resultar problemático para ser ejecutado de forma no distribuida. El empleo de Dash como sustrato para los datos, en lugar de pandas o numpy, podría ser una solución.     