# 1. Imagen base (por ejemplo, Python oficial)
FROM python:3.11-slim

# 2. Establece un directorio dentro del contenedor
WORKDIR /app

# 3. Copia los archivos Python al contenedor
COPY . /app

# 4. Instala dependencias
RUN pip install --no-cache-dir -r requirements.txt

# 5. Comando por defecto al ejecutar el contenedor
CMD ["python", "bot.py"]