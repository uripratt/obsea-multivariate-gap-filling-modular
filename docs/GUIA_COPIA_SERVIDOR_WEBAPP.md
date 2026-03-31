# 🚀 Guía de Copia de Datos del Servidor a la Webapp

## 📋 Resumen de Archivos Generados

Has ejecutado el script `lup_data_obsea_analysis.py` en el servidor y tienes los siguientes archivos:

### ✅ En `output_lup/data/`:
- `OBSEA_multivariate_30min.csv` ✓
- `OBSEA_multivariate_30min_interpolated.csv` ✓

### ✅ En `output_lup/tables/`:
- `correlation_matrix.csv` ✓
- `descriptive_statistics.csv` ✓
- `gap_summary.csv` ✓
- `gaps_detailed.csv` ✓
- `high_quality_variable_names.txt` ✓
- `high_quality_variables.csv` ✓
- `interpolation_comparison.csv` ✓
- `interpolation_tracking.csv` ✓
- `variable_quality_analysis.csv` ✓

---

## 🎯 Archivos que Necesita la Webapp

La webapp requiere estos archivos **esenciales**:

| Archivo | Ubicación Webapp | Estado |
|---------|------------------|--------|
| `OBSEA_multivariate_30min.csv` | `webapp/data/` | ✓ Disponible |
| `OBSEA_multivariate_30min_interpolated.csv` | `webapp/data/` | ✓ Disponible |
| `descriptive_statistics.csv` | `webapp/tables/` | ✓ Disponible |
| `gap_summary.csv` | `webapp/tables/` | ✓ Disponible |
| `interpolation_comparison.csv` | `webapp/tables/` | ✓ Disponible |
| `correlation_matrix.csv` | `webapp/tables/` | ✓ Disponible |

### Archivos Opcionales:
- `comparison_case.json` (data/) - Solo se genera si benchmarks están activos
- `gaps_detailed.csv` (tables/) - Detalles completos de gaps
- `interpolation_tracking.csv` (tables/) - Tracking de interpolación por gap

---

## 🔧 Métodos de Copia

### **Opción 1: Script Automático Local** (Si tienes acceso directo al servidor)

Si estás **en el servidor** o puedes acceder a los archivos directamente:

```bash
cd ~/Documents/PhD/OBSEA_data/CTD/scripts
./copy_server_outputs_to_webapp.sh
```

Este script:
- ✅ Copia todos los archivos esenciales
- ✅ Crea los directorios necesarios
- ✅ Verifica qué archivos faltan
- ✅ Muestra un resumen de lo copiado

---

### **Opción 2: Script Rsync Remoto** (Si el servidor es remoto)

Si el servidor es **remoto** y necesitas sincronizar por SSH:

```bash
cd ~/Documents/PhD/OBSEA_data/CTD/scripts
./sync_from_remote_server.sh usuario@servidor
```

**Ejemplo:**
```bash
./sync_from_remote_server.sh uripratt@obsea-server.upc.edu
```

Este script:
- ✅ Sincroniza solo los archivos necesarios
- ✅ Muestra progreso de transferencia
- ✅ Verifica que todos los archivos esenciales estén presentes
- ✅ Opcionalmente copia figuras

---

### **Opción 3: Copia Manual con rsync**

Si prefieres control total:

```bash
# Variables de configuración
SERVER="usuario@servidor"
REMOTE_PATH="~/Documents/PhD/OBSEA_data/CTD/scripts/output_lup"
LOCAL_WEBAPP="~/Documents/PhD/OBSEA_data/CTD/scripts/webapp"

# Copiar data/
rsync -avz --progress \
  ${SERVER}:${REMOTE_PATH}/data/OBSEA_multivariate_30min.csv \
  ${SERVER}:${REMOTE_PATH}/data/OBSEA_multivariate_30min_interpolated.csv \
  ${LOCAL_WEBAPP}/data/

# Copiar tables/
rsync -avz --progress \
  ${SERVER}:${REMOTE_PATH}/tables/descriptive_statistics.csv \
  ${SERVER}:${REMOTE_PATH}/tables/gap_summary.csv \
  ${SERVER}:${REMOTE_PATH}/tables/interpolation_comparison.csv \
  ${SERVER}:${REMOTE_PATH}/tables/correlation_matrix.csv \
  ${LOCAL_WEBAPP}/tables/
```

---

### **Opción 4: Copia Manual con SCP**

Si prefieres scp:

```bash
# Copiar archivos de data
scp usuario@servidor:~/Documents/PhD/OBSEA_data/CTD/scripts/output_lup/data/OBSEA_multivariate_30min*.csv \
  ~/Documents/PhD/OBSEA_data/CTD/scripts/webapp/data/

# Copiar archivos de tables (esenciales)
scp usuario@servidor:~/Documents/PhD/OBSEA_data/CTD/scripts/output_lup/tables/{descriptive_statistics,gap_summary,interpolation_comparison,correlation_matrix}.csv \
  ~/Documents/PhD/OBSEA_data/CTD/scripts/webapp/tables/
```

---

## ✅ Verificación Post-Copia

Después de copiar, verifica que todo esté en su lugar:

```bash
cd ~/Documents/PhD/OBSEA_data/CTD/scripts/webapp

# Verificar data/
echo "=== DATA ==="
ls -lh data/*.csv

# Verificar tables/
echo "=== TABLES ==="
ls -lh tables/*.csv

# Verificar tamaños
echo "=== TAMAÑOS ==="
du -sh data/
du -sh tables/
```

**Salida esperada:**

```
=== DATA ===
-rw-r--r-- 1 user user  59M OBSEA_multivariate_30min.csv
-rw-r--r-- 1 user user  59M OBSEA_multivariate_30min_interpolated.csv

=== TABLES ===
-rw-r--r-- 1 user user 4.5K correlation_matrix.csv
-rw-r--r-- 1 user user 7.4K descriptive_statistics.csv
-rw-r--r-- 1 user user 7.2K gap_summary.csv
-rw-r--r-- 1 user user 1.3K interpolation_comparison.csv

=== TAMAÑOS ===
120M    data/
30M     tables/
```

---

## 🌐 Iniciar la Webapp

Una vez copiados los archivos, inicia la webapp:

```bash
cd ~/Documents/PhD/OBSEA_data/CTD/scripts/webapp

# Opción 1: Python HTTP Server
python3 -m http.server 8080

# Opción 2: PHP Built-in Server (si disponible)
php -S localhost:8080

# Opción 3: Node.js HTTP Server (si tienes npx)
npx http-server -p 8080
```

Luego abre tu navegador en:
```
http://localhost:8080
```

---

## 🐛 Troubleshooting

### ⚠️ "File not found" en la webapp

**Problema:** La webapp muestra errores de archivos no encontrados.

**Solución:**
1. Verifica que los archivos estén en las rutas correctas
2. Comprueba que el servidor web esté sirviendo desde el directorio correcto
3. Revisa la consola del navegador (F12) para ver qué archivo falta

### ⚠️ "comparison_case.json not found"

**Problema:** Este archivo opcional no existe.

**Solución:** 
- Este archivo solo se genera si los benchmarks están activos
- La webapp debería funcionar sin él (solo afecta el análisis de métodos)
- Si lo necesitas, ejecuta el script con benchmarks habilitados

### ⚠️ Archivos muy grandes (timeout en carga)

**Problema:** Los archivos CSV son muy grandes y tardan en cargar.

**Solución:**
1. Verifica que solo copies los archivos necesarios
2. Considera comprimir archivos: `gzip *.csv`
3. Implementa carga progresiva en la webapp

---

## 📊 Próximos Pasos

1. ✅ Copiar archivos del servidor a webapp
2. ✅ Verificar que todos los archivos esenciales estén presentes
3. 🚀 Iniciar servidor web local
4. 🌐 Abrir webapp en navegador
5. 📈 Visualizar datos procesados

---

**¿Necesitas ayuda?**
- Script no funciona: Revisa permisos de ejecución con `chmod +x script.sh`
- Errores de conexión SSH: Verifica credenciales y acceso al servidor
- Webapp no carga datos: Revisa rutas de archivos y consola del navegador
