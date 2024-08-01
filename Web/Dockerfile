# Usa una imagen base de Python
FROM python:3.12.3-slim

# Establecer el directorio de trabajo al directorio raíz
WORKDIR /

# Copiar los archivos de la aplicación al directorio raíz de la imagen
COPY . .

# Actualizar pip a la última versión
RUN pip install --upgrade pip

# Instalar virtualenv
RUN pip install virtualenv

# Crear un entorno virtual
RUN virtualenv venv

# Activar el entorno virtual e instalar las dependencias
RUN . venv/bin/activate && pip install --no-cache-dir Flask scikit-learn pandas xgboost

# Correr el script de procesamiento
RUN . venv/bin/activate && python processing.py

# Exponer el puerto en el que correrá la aplicación Flask
EXPOSE 5000

# Comando para correr la aplicación Flask
CMD ["/bin/bash", "-c", ". venv/bin/activate && python app.py"]
