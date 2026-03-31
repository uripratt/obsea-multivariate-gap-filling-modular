# 🚀 Guía: Copiar Datos desde JupyterHub Server

## 📋 Información de tu Servidor

**Servidor:** JupyterHub (`jupyter-opratibayarri`)  
**Usuario JupyterHub:** `jovyan`  
**Tu usuario SSH:** Probablemente `opratibayarri` (tu nombre de usuario institucional)

---

## ✅ **Método Recomendado: Script Automatizado**

### **Paso 1: Obtener información del servidor**

Desde tu terminal **EN EL SERVIDOR JupyterHub**, ejecuta:

```bash
# Ver tu directorio actual
pwd

# Buscar dónde está output_lup
find ~ -name "output_lup" -type d

# Obtener hostname completo
hostname -f

# Obtener IP del servidor (por si hostname no funciona)
hostname -I
```

**Guarda esta información:**
- 📂 Ruta a output_lup: `_______________`
- 🌐 Hostname/IP: `_______________`
- 👤 Tu usuario SSH: `_______________`

---

### **Paso 2: Ejecutar script de copia**

Desde **TU MÁQUINA LOCAL** (no el servidor), ejecuta:

```bash
cd ~/Documents/PhD/OBSEA_data/CTD/scripts

# Ejecutar el script
./copy_from_jupyter.sh [HOSTNAME_DEL_SERVIDOR]
```

**Ejemplos:**

```bash
# Si tienes el hostname completo:
./copy_from_jupyter.sh jupyter-opratibayarri.upc.edu

# Si tienes la IP:
./copy_from_jupyter.sh 192.168.1.100

# Si no estás seguro, solo ejecuta:
./copy_from_jupyter.sh
# (el script te pedirá la información)
```

El script:
- ✅ Te pedirá tu usuario SSH
- ✅ Buscará automáticamente el directorio output_lup
- ✅ Copiará todos los archivos necesarios
- ✅ Verificará que todo esté correcto

---

## 🔧 **Método Alternativo: Copia Manual**

Si prefieres hacerlo manualmente o el script no funciona:

### **Opción A: Usando rsync (recomendado)**

```bash
# Variables - EDITA ESTOS VALORES
SERVIDOR="opratibayarri@jupyter-opratibayarri.upc.edu"  # O la IP
RUTA_REMOTA="~/output_lup"  # O la ruta que encontraste en Paso 1

# Copiar data/
rsync -avz --progress \
  ${SERVIDOR}:${RUTA_REMOTA}/data/ \
  ~/Documents/PhD/OBSEA_data/CTD/scripts/webapp/data/

# Copiar tables/
rsync -avz --progress \
  ${SERVIDOR}:${RUTA_REMOTA}/tables/ \
  ~/Documents/PhD/OBSEA_data/CTD/scripts/webapp/tables/
```

### **Opción B: Usando scp**

```bash
# Variables - EDITA ESTOS VALORES
SERVIDOR="opratibayarri@jupyter-opratibayarri.upc.edu"
RUTA_REMOTA="~/output_lup"

# Copiar archivos de data
scp ${SERVIDOR}:${RUTA_REMOTA}/data/OBSEA_multivariate_30min*.csv \
  ~/Documents/PhD/OBSEA_data/CTD/scripts/webapp/data/

# Copiar archivos de tables
scp ${SERVIDOR}:${RUTA_REMOTA}/tables/*.csv \
  ~/Documents/PhD/OBSEA_data/CTD/scripts/webapp/tables/
```

---

## 🔑 **Configurar SSH sin Contraseña (Opcional pero Recomendado)**

Para evitar escribir la contraseña cada vez:

```bash
# Desde tu máquina local:
./setup_ssh_key.sh opratibayarri@jupyter-opratibayarri.upc.edu
```

Esto:
1. Genera una clave SSH (si no existe)
2. La copia al servidor
3. Te permite conectar sin contraseña en el futuro

**Solo necesitas hacerlo UNA VEZ.**

---

## 🌐 **Método Gráfico: Usar JupyterHub File Browser**

Si JupyterHub tiene interfaz web, puedes descargar archivos directamente:

1. **Abre JupyterHub en tu navegador**
   - URL: probablemente `https://jupyter-opratibayarri.upc.edu` o similar

2. **Navega al directorio `output_lup/`**

3. **Descarga los archivos necesarios:**
   - Selecciona archivos en `data/` y `tables/`
   - Click derecho → Download
   - O usa el botón de descarga de Jupyter

4. **Mueve los archivos descargados a:**
   ```bash
   ~/Documents/PhD/OBSEA_data/CTD/scripts/webapp/data/
   ~/Documents/PhD/OBSEA_data/CTD/scripts/webapp/tables/
   ```

---

## 📦 **Método por Compresión (Para Archivos Grandes)**

Si los archivos son muy grandes o la conexión es lenta:

### **En el servidor JupyterHub:**

```bash
# Comprimir archivos
cd ~/output_lup
tar -czf obsea_webapp_data.tar.gz data/ tables/
```

### **En tu máquina local:**

```bash
# Copiar el archivo comprimido
scp opratibayarri@jupyter-opratibayarri.upc.edu:~/output_lup/obsea_webapp_data.tar.gz ~/Downloads/

# Extraer en la webapp
cd ~/Documents/PhD/OBSEA_data/CTD/scripts/webapp
tar -xzf ~/Downloads/obsea_webapp_data.tar.gz
```

**Ventajas:**
- ✅ Una sola transferencia (menos contraseñas)
- ✅ Más rápido (comprimido)
- ✅ Puedes reanudar si se interrumpe

---

## ✅ **Verificación Post-Copia**

Después de copiar, verifica que todo esté bien:

```bash
cd ~/Documents/PhD/OBSEA_data/CTD/scripts

# Verificar archivos
./check_webapp_files.sh

# O manualmente:
ls -lh webapp/data/
ls -lh webapp/tables/
```

**Deberías ver:**
```
✓ data/OBSEA_multivariate_30min.csv (59M)
✓ data/OBSEA_multivariate_30min_interpolated.csv (59M)
✓ tables/descriptive_statistics.csv (8.0K)
✓ tables/gap_summary.csv (8.0K)
✓ tables/interpolation_comparison.csv (4.0K)
✓ tables/correlation_matrix.csv (8.0K)
```

---

## 🐛 **Troubleshooting JupyterHub**

### ❌ "Permission denied" al conectar

**Problema:** No puedes conectar por SSH al servidor.

**Soluciones:**
1. Verifica que SSH esté habilitado en JupyterHub
2. Usa tu usuario institucional, no `jovyan`
3. Pide ayuda al administrador del servidor
4. Usa el método gráfico de descarga por web

### ❌ "Host key verification failed"

**Solución:**
```bash
ssh-keygen -R jupyter-opratibayarri.upc.edu
# Luego intenta de nuevo
```

### ❌ "Output_lup not found"

**Problema:** El directorio no está donde esperabas.

**Solución:**
En el servidor ejecuta:
```bash
find / -name "lup_data_obsea_analysis.py" 2>/dev/null
# Esto te dirá dónde ejecutaste el script
```

---

## 📊 **Resumen de Comandos Rápidos**

```bash
# TODO EN UNO - Desde tu máquina local:

# 1. Copiar archivos
cd ~/Documents/PhD/OBSEA_data/CTD/scripts
./copy_from_jupyter.sh jupyter-opratibayarri.upc.edu

# 2. Verificar
./check_webapp_files.sh

# 3. Listo! La webapp ya tiene los datos
```

---

## ℹ️ **Información Adicional**

- **JupyterHub usa directorios diferentes** que pueden variar según la instalación
- Rutas comunes en JupyterHub:
  - `~/` - Tu home directory
  - `~/work/` - Directorio de trabajo (común en algunos JupyterHub)
  - `/home/jovyan/` - Home del usuario Jupyter

- **Para conectar por SSH a JupyterHub:**
  - Normalmente necesitas puerto 22 (SSH)
  - Algunos JupyterHub NO permiten SSH, solo interfaz web
  - Si no funciona SSH, contacta al administrador del sistema

---

**¿Necesitas ayuda específica con tu servidor JupyterHub?**
Comparte el hostname/IP y te ayudo a configurar la conexión.
