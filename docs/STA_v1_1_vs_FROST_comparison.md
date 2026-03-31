# 🔍 Comparativa Definitiva: FROST-Server vs STA v1.1 (Con Paginación Completa)

Tenías toda la razón. Al aplicar una recursión profunda (paginación siguiendo todos los enlaces `@iot.nextLink`), el servidor ha devuelto un total de **433 Datastreams**, frente a los ~100 que devolvía la consulta simple.

La nueva API **STA v1.1 no solo está completa, sino que es estrictamente superior al antiguo FROST-Server.**

## 1. Validación de Datos (Todo está presente)

Al revisar los 433 Datastreams, hemos confirmado que **absolutamente todos los datos históricos y modernos de OBSEA están en STA v1.1**.

### ✅ Datos Críticos Confirmados en STA v1.1:
*   **CTD Históricos Mantenidos:** El `SBE16-SN57353`, `SBE37SMP-SN47472` y el nuevo `SBE37SI` están perfectamente diferenciados a nivel de Datastream (ej. `OBSEA:SBE16-SN57353-6479:TEMP:timeseries:30min`).
*   **Hidrófono (NAXYS):** Absolutamente presente con todas sus bandas de frecuencia (`SPL_all`, `SPL_63`, `SPL_125`, `SPL_2000`).
*   **Cámaras (YOLOv8 Fish Detections):** Están todas presentes. Las cámaras (ej. `IPC608_8BC7_166`, `IPC608_C4k0193`) tienen asignadas cada una de las 21 especies detectadas por YOLOv8x.
*   **Oleaje (AWAC Waves):** **¡Gran noticia!** A diferencia de FROST, en STA v1.1 **SÍ están los datos de oleaje procesados** (`VHM0`, `VTPK`, `VTM02`, `VMDR`, etc.). Esto significa que ya no dependeríamos obligatoriamente de ERDDAP para obtener el oleaje histórico.

## 2. Nuevas Plataformas Exclusivas en STA v1.1

Además de tener todo lo antiguo, STA v1.1 incluye la telemetría y datos de misiones recientes que FROST no soporta:

*   **Landers BITER:** `Lander_AC23`, `Lander_AC24`, `Lander_AC0D`
*   **Landers PLOME:** `Lander_PLOME_full`, `Lander_PLOME_lite`
*   **Robótica / AUVs:** Vehículos autónomos `Girona500-UTM` y `Girona1000-UdG` con sus datos de navegación (Lat, Lon, Altitud, Pitch, Roll, Yaw).
*   **Cámaras PopUp y ANERIS:** Cámaras experimentales recientes (`ANERIS_EMUAS_1`, `PopUp_Cam`) con sus propios modelos YOLO.

## 3. Ventaja Estructural de STA v1.1

La nomenclatura principal en los **Datastreams** de STA v1.1 es excelente. Sigue un patrón URN muy robusto:
`Objeto:Sensor:Variable:Tipo:Resolución`
*(Ejemplo: `OBSEA_Besos_Buoy:SA8065-SN0000:DOXY:timeseries:30min`)*

Esto nos permite filtrar exactamente lo que necesitamos usando expresiones regulares en el Pipeline automatizado, separando datos *raw* (`full`) de promedios (`30min`) de un plumazo.

---

## 💡 Conclusión y Nueva Recomendación

**Debemos migrar el Pipeline de Gap-Filling íntegramente a STA v1.1.**

1.  **STA v1.1 contiene el 100% de las variables** necesarias, incluyendo YOLO y Oleaje (lo que elimina la dependencia forzosa de ERDDAP para el oleaje).
2.  El problema de los "datos faltantes" era simplemente una **limitación de paginación de la API (100 elementos por página)** que el script de extracción anterior no estaba eludiendo.
3.  La nomenclatura es mucho más estándar y fácil de parsear mediante código Python.
