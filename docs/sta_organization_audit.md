# Auditoría Estructural: OBSEA OGC SensorThings API v1.1

Esta auditoría detalla cómo está organizada jerárquicamente la base de datos de OBSEA siguiendo el estándar OGC SensorThings API.

## 1. Topología del Servidor OGC STA
La API organiza la información en 4 entidades jerárquicas relacionadas entre sí:
1. **Thing (Nodo)**: La plataforma física de observación (ej. la boya *OBSEA*, la torre *CTVG*, un Lander submarino).
2. **Sensor (Instrumento)**: El aparato de medición física concreto fijado a un Thing (ej. un CTD *SBE16*, o un medidor de corrientes *AWAC*).
3. **ObservedProperty (Variable Física)**: La magnitud que se desea medir, agnóstica al sensor (ej. Temperatura *TEMP*, Salinidad *PSAL*).
4. **Datastream (Serie Temporal)**: Es la conjunción de las 3 anteriores. Un Datastream agrupa las observaciones concretas de un *Sensor* situado en un *Thing*, midiendo una *Property* a un intervalo temporal específico (ej. a `:30min` o `:full`).

## 2. Resumen Cuantitativo del Servidor
- **Total de Nodos/Things:** 13
- **Total de Instrumentos/Sensors:** 47
- **Total de Variables/ObservedProperties:** 97
- **Total de Flujos de Datos/Datastreams:** 433

---

## 3. Mapa Estructural de Plataformas e Instrumentos

### 📍 Plataforma (Thing): `CTVG`
#### 🔬 Instrumento (Sensor): `Vantage_Pro2-SN6150CEU`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| **200** | `CAPH` | CAPH:timeseries:30min | **30min** | 2010-04-08 -> 2025-05-15 |
| **201** | `RELH` | RELH:timeseries:30min | **30min** | 2010-04-08 -> 2025-02-12 |
| **202** | `AIRT` | AIRT:timeseries:30min | **30min** | 2010-04-08 -> 2024-11-11 |
| **203** | `WDIR` | WDIR:timeseries:30min | **30min** | 2010-04-08 -> 2025-02-12 |
| **204** | `WSPD` | WSPD:timeseries:30min | **30min** | 2010-04-08 -> 2025-02-12 |
| 194 | `CAPH` | CAPH:timeseries:full | full | N -> A |
| 195 | `RELH` | RELH:timeseries:full | full | N -> A |
| 196 | `AIRT` | AIRT:timeseries:full | full | N -> A |
| 197 | `WDIR` | WDIR:timeseries:full | full | N -> A |
| 198 | `WSPD` | WSPD:timeseries:full | full | N -> A |
| 199 | `RAIN` | RAIN:timeseries:full | full | N -> A |

#### 🔬 Instrumento (Sensor): `Vantage_Pro2-SNBF241204107`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| **462** | `CAPH` | CAPH:timeseries:30min | **30min** | 2026-01-13 -> 2026-03-11 |
| **463** | `RELH` | RELH:timeseries:30min | **30min** | 2026-01-13 -> 2026-03-11 |
| **464** | `AIRT` | AIRT:timeseries:30min | **30min** | 2026-01-13 -> 2026-03-11 |
| **465** | `WDIR` | WDIR:timeseries:30min | **30min** | 2026-01-13 -> 2026-03-11 |
| **466** | `WSPD` | WSPD:timeseries:30min | **30min** | 2026-01-13 -> 2026-03-11 |
| **468** | `WSPD10` | WSPD10:timeseries:30min | **30min** | N -> A |
| 456 | `CAPH` | CAPH:timeseries:full | full | N -> A |
| 457 | `RELH` | RELH:timeseries:full | full | N -> A |
| 458 | `AIRT` | AIRT:timeseries:full | full | N -> A |
| 459 | `WDIR` | WDIR:timeseries:full | full | N -> A |
| 460 | `WSPD` | WSPD:timeseries:full | full | N -> A |
| 461 | `RAIN` | RAIN:timeseries:full | full | N -> A |
| 467 | `WSPD10` | WSPD10:timeseries:full | full | N -> A |

### 📍 Plataforma (Thing): `Girona1000-UdG`
#### 🔬 Instrumento (Sensor): `FLIR_Blackfly-SN21115515`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 455 | `underwater_photography` | underwater_photography:files | files | 2025-07-16 -> 2025-07-23 |

#### 🔬 Instrumento (Sensor): `FLIR_Blackfly-SN22240994`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 384 | `underwater_photography` | underwater_photography:files | files | 2023-12-19 -> 2024-07-25 |

#### 🔬 Instrumento (Sensor): `Girona1000-UdG-navigation`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 377 | `LATITUDE` | LATITUDE:timeseries:full | full | N -> A |
| 378 | `LONGITUDE` | LONGITUDE:timeseries:full | full | N -> A |
| 379 | `ALTITUDE` | ALTITUDE:timeseries:full | full | N -> A |
| 380 | `DEPTH` | DEPTH:timeseries:full | full | N -> A |
| 381 | `PTCH` | PTCH:timeseries:full | full | N -> A |
| 382 | `YAW` | YAW:timeseries:full | full | N -> A |
| 383 | `ROLL` | ROLL:timeseries:full | full | N -> A |

### 📍 Plataforma (Thing): `Girona500-UTM`
#### 🔬 Instrumento (Sensor): `FLIR_Blackfly-SN21115514`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 368 | `underwater_photography` | underwater_photography:files | files | 2023-12-15 -> 2025-07-23 |

#### 🔬 Instrumento (Sensor): `Girona500-UTM-navigation`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 369 | `LATITUDE` | LATITUDE:timeseries:full | full | N -> A |
| 370 | `LONGITUDE` | LONGITUDE:timeseries:full | full | N -> A |
| 371 | `ALTITUDE` | ALTITUDE:timeseries:full | full | N -> A |
| 372 | `DEPTH` | DEPTH:timeseries:full | full | N -> A |
| 373 | `PTCH` | PTCH:timeseries:full | full | N -> A |
| 374 | `YAW` | YAW:timeseries:full | full | N -> A |
| 375 | `ROLL` | ROLL:timeseries:full | full | N -> A |

### 📍 Plataforma (Thing): `Lander_AC0D`
#### 🔬 Instrumento (Sensor): `Guard1-SN0001`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 142 | `underwater_photography` | underwater_photography:files | files | 2022-09-24 -> 2024-02-06 |

#### 🔬 Instrumento (Sensor): `Guard1-SN0002`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 478 | `underwater_photography` | underwater_photography:files | files | 2024-12-01 -> 2025-12-04 |

#### 🔬 Instrumento (Sensor): `SBE37SMP-SN3724510`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 143 | `TEMP` | TEMP:timeseries:full | full | N -> A |
| 144 | `PRES` | PRES:timeseries:full | full | N -> A |
| 145 | `CNDC` | CNDC:timeseries:full | full | N -> A |
| 146 | `PSAL` | PSAL:timeseries:full | full | N -> A |
| 147 | `SVEL` | SVEL:timeseries:full | full | N -> A |
| 148 | `POTM` | POTM:timeseries:full | full | N -> A |
| 149 | `DEPTH` | DEPTH:timeseries:full | full | N -> A |
| 150 | `SIGT` | SIGT:timeseries:full | full | N -> A |

#### 🔬 Instrumento (Sensor): `SBE37SMP-SN3727115`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 481 | `TEMP` | TEMP:timeseries:full | full | N -> A |
| 482 | `PRES` | PRES:timeseries:full | full | N -> A |
| 483 | `CNDC` | CNDC:timeseries:full | full | N -> A |
| 484 | `PSAL` | PSAL:timeseries:full | full | N -> A |
| 485 | `SVEL` | SVEL:timeseries:full | full | N -> A |
| 486 | `POTM` | POTM:timeseries:full | full | N -> A |
| 487 | `DEPTH` | DEPTH:timeseries:full | full | N -> A |
| 488 | `SIGT` | SIGT:timeseries:full | full | N -> A |

### 📍 Plataforma (Thing): `Lander_AC23`
#### 🔬 Instrumento (Sensor): `Guard1-SN0003`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 51 | `underwater_photography` | underwater_photography:files | files | 2023-06-03 -> 2025-12-04 |

#### 🔬 Instrumento (Sensor): `INFINITY-EM-SN1965`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 81 | `CSPD` | CSPD:timeseries:full | full | N -> A |
| 82 | `CDIR` | CDIR:timeseries:full | full | N -> A |
| 83 | `VCUR` | VCUR:timeseries:full | full | N -> A |
| 84 | `UCUR` | UCUR:timeseries:full | full | N -> A |
| 85 | `TEMP` | TEMP:timeseries:full | full | N -> A |
| 86 | `HEAD` | HEAD:timeseries:full | full | N -> A |
| 87 | `BATT` | BATT:timeseries:full | full | N -> A |

#### 🔬 Instrumento (Sensor): `SBE37SMP-SN3724510`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 151 | `TEMP` | TEMP:timeseries:full | full | N -> A |
| 152 | `PRES` | PRES:timeseries:full | full | N -> A |
| 153 | `CNDC` | CNDC:timeseries:full | full | N -> A |
| 154 | `PSAL` | PSAL:timeseries:full | full | N -> A |
| 155 | `SVEL` | SVEL:timeseries:full | full | N -> A |
| 156 | `POTM` | POTM:timeseries:full | full | N -> A |
| 157 | `DEPTH` | DEPTH:timeseries:full | full | N -> A |
| 158 | `SIGT` | SIGT:timeseries:full | full | N -> A |

### 📍 Plataforma (Thing): `Lander_AC24`
#### 🔬 Instrumento (Sensor): `Guard1-SN0001`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 479 | `underwater_photography` | underwater_photography:files | files | N -> A |

#### 🔬 Instrumento (Sensor): `Guard1-SN0002`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 96 | `underwater_photography` | underwater_photography:files | files | 2023-06-03 -> 2024-11-05 |

#### 🔬 Instrumento (Sensor): `SBE37SMP-SN3727117`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 489 | `TEMP` | TEMP:timeseries:full | full | N -> A |
| 490 | `PRES` | PRES:timeseries:full | full | N -> A |
| 491 | `CNDC` | CNDC:timeseries:full | full | N -> A |
| 492 | `PSAL` | PSAL:timeseries:full | full | N -> A |
| 493 | `SVEL` | SVEL:timeseries:full | full | N -> A |
| 494 | `POTM` | POTM:timeseries:full | full | N -> A |
| 495 | `DEPTH` | DEPTH:timeseries:full | full | N -> A |
| 496 | `SIGT` | SIGT:timeseries:full | full | N -> A |

### 📍 Plataforma (Thing): `Lander_PLOME_full`
#### 🔬 Instrumento (Sensor): `Cyclops_7F-SN21102107`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 419 | `CPWC` | CPWC:timeseries:full | full | N -> A |

#### 🔬 Instrumento (Sensor): `RBRcoda3PAR-SN211667`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 420 | `PAR` | PAR:timeseries:full | full | N -> A |

#### 🔬 Instrumento (Sensor): `RBRconcerto-SN210972`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 421 | `TEMP` | TEMP:timeseries:full | full | N -> A |
| 422 | `PRES` | PRES:timeseries:full | full | N -> A |
| 423 | `DEPTH` | DEPTH:timeseries:full | full | N -> A |
| 424 | `CNDC` | CNDC:timeseries:full | full | N -> A |
| 425 | `PSAL` | PSAL:timeseries:full | full | N -> A |
| 426 | `SVEL` | SVEL:timeseries:full | full | N -> A |
| 427 | `SIGMAT` | SIGMAT:timeseries:full | full | N -> A |

### 📍 Plataforma (Thing): `Lander_Palamos_01`
#### 🔬 Instrumento (Sensor): `IPC608_4396_100`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 376 | `underwater_photography` | underwater_photography:files | files | 2024-12-18 -> 2025-10-31 |

### 📍 Plataforma (Thing): `OBSEA`
#### 🔬 Instrumento (Sensor): `AIPC608UW_10_167`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 243 | `aetomylaeus_bovinus` | aetomylaeus_bovinus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 244 | `chromis_chromis` | chromis_chromis:yolov8x_21sp_5364img:detections | detections | N -> A |
| 245 | `coris_julis` | coris_julis:yolov8x_21sp_5364img:detections | detections | N -> A |
| 246 | `dentex_dentex` | dentex_dentex:yolov8x_21sp_5364img:detections | detections | N -> A |
| 247 | `diplodus_cervinus` | diplodus_cervinus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 248 | `diplodus_puntazzo` | diplodus_puntazzo:yolov8x_21sp_5364img:detections | detections | N -> A |
| 249 | `diplodus_sargus` | diplodus_sargus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 250 | `diplodus_vulgaris` | diplodus_vulgaris:yolov8x_21sp_5364img:detections | detections | N -> A |
| 251 | `epinephelus_costae` | epinephelus_costae:yolov8x_21sp_5364img:detections | detections | N -> A |
| 252 | `epinephelus_marginatus` | epinephelus_marginatus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 253 | `mullus_surmuletus` | mullus_surmuletus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 254 | `muraena_helena` | muraena_helena:yolov8x_21sp_5364img:detections | detections | N -> A |
| 255 | `oblada_melanurus` | oblada_melanurus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 256 | `parablennius_gattorugine` | parablennius_gattorugine:yolov8x_21sp_5364img:detections | detections | N -> A |
| 257 | `sarpa_salpa` | sarpa_salpa:yolov8x_21sp_5364img:detections | detections | N -> A |
| 258 | `seriola_dumerili` | seriola_dumerili:yolov8x_21sp_5364img:detections | detections | N -> A |
| 259 | `serranus_cabrilla` | serranus_cabrilla:yolov8x_21sp_5364img:detections | detections | N -> A |
| 260 | `sparus_aurata` | sparus_aurata:yolov8x_21sp_5364img:detections | detections | N -> A |
| 261 | `symphodus_mediterraneus` | symphodus_mediterraneus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 10 | `underwater_photography` | underwater_photography:files | files | 2024-03-19 -> 2024-10-24 |

#### 🔬 Instrumento (Sensor): `ANB_AQ50-SN888888`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| **221** | `ALKY` | ALKY:timeseries:30min | **30min** | 2024-10-29 -> 2025-07-26 |
| 214 | `ALKY` | ALKY:timeseries:full | full | N -> A |
| 215 | `TEMP` | TEMP:timeseries:full | full | N -> A |
| 216 | `PSAL` | PSAL:timeseries:full | full | N -> A |
| 217 | `CNDC` | CNDC:timeseries:full | full | N -> A |
| 218 | `transducer_health` | transducer_health:timeseries:full | full | N -> A |
| 219 | `sensor_diagnostics` | sensor_diagnostics:timeseries:full | full | N -> A |
| 220 | `file_number` | file_number:timeseries:full | full | N -> A |

#### 🔬 Instrumento (Sensor): `ANERIS_EMUAS_1`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 400 | `aetomylaeus_bovinus` | aetomylaeus_bovinus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 401 | `chromis_chromis` | chromis_chromis:yolov8x_21sp_5364img:detections | detections | N -> A |
| 402 | `coris_julis` | coris_julis:yolov8x_21sp_5364img:detections | detections | N -> A |
| 403 | `dentex_dentex` | dentex_dentex:yolov8x_21sp_5364img:detections | detections | N -> A |
| 404 | `diplodus_cervinus` | diplodus_cervinus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 405 | `diplodus_puntazzo` | diplodus_puntazzo:yolov8x_21sp_5364img:detections | detections | N -> A |
| 406 | `diplodus_sargus` | diplodus_sargus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 407 | `diplodus_vulgaris` | diplodus_vulgaris:yolov8x_21sp_5364img:detections | detections | N -> A |
| 408 | `epinephelus_costae` | epinephelus_costae:yolov8x_21sp_5364img:detections | detections | N -> A |
| 409 | `epinephelus_marginatus` | epinephelus_marginatus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 410 | `mullus_surmuletus` | mullus_surmuletus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 411 | `muraena_helena` | muraena_helena:yolov8x_21sp_5364img:detections | detections | N -> A |
| 412 | `oblada_melanurus` | oblada_melanurus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 413 | `parablennius_gattorugine` | parablennius_gattorugine:yolov8x_21sp_5364img:detections | detections | N -> A |
| 414 | `sarpa_salpa` | sarpa_salpa:yolov8x_21sp_5364img:detections | detections | N -> A |
| 415 | `seriola_dumerili` | seriola_dumerili:yolov8x_21sp_5364img:detections | detections | N -> A |
| 416 | `serranus_cabrilla` | serranus_cabrilla:yolov8x_21sp_5364img:detections | detections | N -> A |
| 417 | `sparus_aurata` | sparus_aurata:yolov8x_21sp_5364img:detections | detections | N -> A |
| 418 | `symphodus_mediterraneus` | symphodus_mediterraneus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 345 | `underwater_photography` | underwater_photography:files | files | 2024-11-15 -> 2025-11-14 |

#### 🔬 Instrumento (Sensor): `ANERIS_EMUAS_2`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 344 | `underwater_photography` | underwater_photography:files | files | 2024-11-15 -> 2025-11-14 |

#### 🔬 Instrumento (Sensor): `AWAC-SN5931`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| **189** | `CSPD` | CSPD:profiles:30min | **30min** | 2010-04-08 -> 2025-04-17 |
| **190** | `CDIR` | CDIR:profiles:30min | **30min** | 2010-04-08 -> 2025-04-17 |
| **191** | `UCUR` | UCUR:profiles:30min | **30min** | 2010-04-08 -> 2025-04-17 |
| **192** | `VCUR` | VCUR:profiles:30min | **30min** | 2010-04-08 -> 2025-04-17 |
| **193** | `ZCUR` | ZCUR:profiles:30min | **30min** | 2010-04-08 -> 2025-04-17 |
| 159 | `CSPD` | CSPD:profiles:full | full | N -> A |
| 160 | `CDIR` | CDIR:profiles:full | full | N -> A |
| 161 | `UCUR` | UCUR:profiles:full | full | N -> A |
| 162 | `VCUR` | VCUR:profiles:full | full | N -> A |
| 163 | `ZCUR` | ZCUR:profiles:full | full | N -> A |
| 164 | `BEAM1` | BEAM1:profiles:full | full | N -> A |
| 165 | `BEAM2` | BEAM2:profiles:full | full | N -> A |
| 166 | `BEAM3` | BEAM3:profiles:full | full | N -> A |
| 167 | `VHM0` | VHM0:timeseries:full | full | N -> A |
| 168 | `VAVH` | VAVH:timeseries:full | full | N -> A |
| 169 | `VH110` | VH110:timeseries:full | full | N -> A |
| 170 | `VZMX` | VZMX:timeseries:full | full | N -> A |
| 171 | `VTM02` | VTM02:timeseries:full | full | N -> A |
| 172 | `VTPK` | VTPK:timeseries:full | full | N -> A |
| 173 | `VTZA` | VTZA:timeseries:full | full | N -> A |
| 174 | `VPED` | VPED:timeseries:full | full | N -> A |
| 175 | `VSPS` | VSPS:timeseries:full | full | N -> A |
| 176 | `VMDR` | VMDR:timeseries:full | full | N -> A |
| 177 | `UNDX` | UNDX:timeseries:full | full | N -> A |
| 178 | `ERRC` | ERRC:timeseries:full | full | N -> A |
| 179 | `STAT` | STAT:timeseries:full | full | N -> A |
| 180 | `BATT` | BATT:timeseries:full | full | N -> A |
| 181 | `SVEL` | SVEL:timeseries:full | full | N -> A |
| 182 | `HEAD` | HEAD:timeseries:full | full | N -> A |
| 183 | `PTCH` | PTCH:timeseries:full | full | N -> A |
| 184 | `ROLL` | ROLL:timeseries:full | full | N -> A |
| 185 | `PRES` | PRES:timeseries:full | full | N -> A |
| 186 | `TEMP` | TEMP:timeseries:full | full | N -> A |
| 187 | `AIN1` | AIN1:timeseries:full | full | N -> A |
| 188 | `AIN2` | AIN2:timeseries:full | full | N -> A |

#### 🔬 Instrumento (Sensor): `AXIS_P1346`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 480 | `underwater_photography` | underwater_photography:files | files | 2014-12-11 -> 2017-10-05 |

#### 🔬 Instrumento (Sensor): `DCS_7010L-SN8304`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 325 | `aetomylaeus_bovinus` | aetomylaeus_bovinus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 326 | `chromis_chromis` | chromis_chromis:yolov8x_21sp_5364img:detections | detections | N -> A |
| 327 | `coris_julis` | coris_julis:yolov8x_21sp_5364img:detections | detections | N -> A |
| 328 | `dentex_dentex` | dentex_dentex:yolov8x_21sp_5364img:detections | detections | N -> A |
| 329 | `diplodus_cervinus` | diplodus_cervinus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 330 | `diplodus_puntazzo` | diplodus_puntazzo:yolov8x_21sp_5364img:detections | detections | N -> A |
| 331 | `diplodus_sargus` | diplodus_sargus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 332 | `diplodus_vulgaris` | diplodus_vulgaris:yolov8x_21sp_5364img:detections | detections | N -> A |
| 333 | `epinephelus_costae` | epinephelus_costae:yolov8x_21sp_5364img:detections | detections | N -> A |
| 334 | `epinephelus_marginatus` | epinephelus_marginatus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 335 | `mullus_surmuletus` | mullus_surmuletus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 336 | `muraena_helena` | muraena_helena:yolov8x_21sp_5364img:detections | detections | N -> A |
| 337 | `oblada_melanurus` | oblada_melanurus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 338 | `parablennius_gattorugine` | parablennius_gattorugine:yolov8x_21sp_5364img:detections | detections | N -> A |
| 339 | `sarpa_salpa` | sarpa_salpa:yolov8x_21sp_5364img:detections | detections | N -> A |
| 340 | `seriola_dumerili` | seriola_dumerili:yolov8x_21sp_5364img:detections | detections | N -> A |
| 341 | `serranus_cabrilla` | serranus_cabrilla:yolov8x_21sp_5364img:detections | detections | N -> A |
| 342 | `sparus_aurata` | sparus_aurata:yolov8x_21sp_5364img:detections | detections | N -> A |
| 343 | `symphodus_mediterraneus` | symphodus_mediterraneus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 323 | `underwater_photography` | underwater_photography:files | files | 2020-07-20 -> 2023-02-28 |

#### 🔬 Instrumento (Sensor): `IPC608_73DF_220`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 397 | `underwater_photography` | underwater_photography:files | files | 2025-02-20 -> 2025-11-14 |

#### 🔬 Instrumento (Sensor): `IPC608_8B64_165`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 263 | `aetomylaeus_bovinus` | aetomylaeus_bovinus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 264 | `chromis_chromis` | chromis_chromis:yolov8x_21sp_5364img:detections | detections | N -> A |
| 265 | `coris_julis` | coris_julis:yolov8x_21sp_5364img:detections | detections | N -> A |
| 266 | `dentex_dentex` | dentex_dentex:yolov8x_21sp_5364img:detections | detections | N -> A |
| 267 | `diplodus_cervinus` | diplodus_cervinus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 268 | `diplodus_puntazzo` | diplodus_puntazzo:yolov8x_21sp_5364img:detections | detections | N -> A |
| 269 | `diplodus_sargus` | diplodus_sargus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 270 | `diplodus_vulgaris` | diplodus_vulgaris:yolov8x_21sp_5364img:detections | detections | N -> A |
| 271 | `epinephelus_costae` | epinephelus_costae:yolov8x_21sp_5364img:detections | detections | N -> A |
| 272 | `epinephelus_marginatus` | epinephelus_marginatus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 273 | `mullus_surmuletus` | mullus_surmuletus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 274 | `muraena_helena` | muraena_helena:yolov8x_21sp_5364img:detections | detections | N -> A |
| 275 | `oblada_melanurus` | oblada_melanurus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 276 | `parablennius_gattorugine` | parablennius_gattorugine:yolov8x_21sp_5364img:detections | detections | N -> A |
| 277 | `sarpa_salpa` | sarpa_salpa:yolov8x_21sp_5364img:detections | detections | N -> A |
| 278 | `seriola_dumerili` | seriola_dumerili:yolov8x_21sp_5364img:detections | detections | N -> A |
| 279 | `serranus_cabrilla` | serranus_cabrilla:yolov8x_21sp_5364img:detections | detections | N -> A |
| 280 | `sparus_aurata` | sparus_aurata:yolov8x_21sp_5364img:detections | detections | N -> A |
| 281 | `symphodus_mediterraneus` | symphodus_mediterraneus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 1 | `underwater_photography` | underwater_photography:files | files | 2023-07-22 -> 2024-09-06 |

#### 🔬 Instrumento (Sensor): `IPC608_8BC7_166`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 223 | `aetomylaeus_bovinus` | aetomylaeus_bovinus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 224 | `chromis_chromis` | chromis_chromis:yolov8x_21sp_5364img:detections | detections | N -> A |
| 225 | `coris_julis` | coris_julis:yolov8x_21sp_5364img:detections | detections | N -> A |
| 226 | `dentex_dentex` | dentex_dentex:yolov8x_21sp_5364img:detections | detections | N -> A |
| 227 | `diplodus_cervinus` | diplodus_cervinus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 228 | `diplodus_puntazzo` | diplodus_puntazzo:yolov8x_21sp_5364img:detections | detections | N -> A |
| 229 | `diplodus_sargus` | diplodus_sargus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 230 | `diplodus_vulgaris` | diplodus_vulgaris:yolov8x_21sp_5364img:detections | detections | N -> A |
| 231 | `epinephelus_costae` | epinephelus_costae:yolov8x_21sp_5364img:detections | detections | N -> A |
| 232 | `epinephelus_marginatus` | epinephelus_marginatus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 233 | `mullus_surmuletus` | mullus_surmuletus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 234 | `muraena_helena` | muraena_helena:yolov8x_21sp_5364img:detections | detections | N -> A |
| 235 | `oblada_melanurus` | oblada_melanurus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 236 | `parablennius_gattorugine` | parablennius_gattorugine:yolov8x_21sp_5364img:detections | detections | N -> A |
| 237 | `sarpa_salpa` | sarpa_salpa:yolov8x_21sp_5364img:detections | detections | N -> A |
| 238 | `seriola_dumerili` | seriola_dumerili:yolov8x_21sp_5364img:detections | detections | N -> A |
| 239 | `serranus_cabrilla` | serranus_cabrilla:yolov8x_21sp_5364img:detections | detections | N -> A |
| 240 | `sparus_aurata` | sparus_aurata:yolov8x_21sp_5364img:detections | detections | N -> A |
| 241 | `symphodus_mediterraneus` | symphodus_mediterraneus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 60 | `underwater_photography` | underwater_photography:files | files | 2023-07-16 -> 2025-11-14 |

#### 🔬 Instrumento (Sensor): `IPC608_C4k0193`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 283 | `aetomylaeus_bovinus` | aetomylaeus_bovinus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 284 | `chromis_chromis` | chromis_chromis:yolov8x_21sp_5364img:detections | detections | N -> A |
| 285 | `coris_julis` | coris_julis:yolov8x_21sp_5364img:detections | detections | N -> A |
| 286 | `dentex_dentex` | dentex_dentex:yolov8x_21sp_5364img:detections | detections | N -> A |
| 287 | `diplodus_cervinus` | diplodus_cervinus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 288 | `diplodus_puntazzo` | diplodus_puntazzo:yolov8x_21sp_5364img:detections | detections | N -> A |
| 289 | `diplodus_sargus` | diplodus_sargus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 290 | `diplodus_vulgaris` | diplodus_vulgaris:yolov8x_21sp_5364img:detections | detections | N -> A |
| 291 | `epinephelus_costae` | epinephelus_costae:yolov8x_21sp_5364img:detections | detections | N -> A |
| 292 | `epinephelus_marginatus` | epinephelus_marginatus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 293 | `mullus_surmuletus` | mullus_surmuletus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 294 | `muraena_helena` | muraena_helena:yolov8x_21sp_5364img:detections | detections | N -> A |
| 295 | `oblada_melanurus` | oblada_melanurus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 296 | `parablennius_gattorugine` | parablennius_gattorugine:yolov8x_21sp_5364img:detections | detections | N -> A |
| 297 | `sarpa_salpa` | sarpa_salpa:yolov8x_21sp_5364img:detections | detections | N -> A |
| 298 | `seriola_dumerili` | seriola_dumerili:yolov8x_21sp_5364img:detections | detections | N -> A |
| 299 | `serranus_cabrilla` | serranus_cabrilla:yolov8x_21sp_5364img:detections | detections | N -> A |
| 300 | `sparus_aurata` | sparus_aurata:yolov8x_21sp_5364img:detections | detections | N -> A |
| 301 | `symphodus_mediterraneus` | symphodus_mediterraneus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 112 | `underwater_photography` | underwater_photography:files | files | 2023-01-13 -> 2024-03-19 |

#### 🔬 Instrumento (Sensor): `IPC608_CC70_169`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 304 | `aetomylaeus_bovinus` | aetomylaeus_bovinus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 305 | `chromis_chromis` | chromis_chromis:yolov8x_21sp_5364img:detections | detections | N -> A |
| 306 | `coris_julis` | coris_julis:yolov8x_21sp_5364img:detections | detections | N -> A |
| 307 | `dentex_dentex` | dentex_dentex:yolov8x_21sp_5364img:detections | detections | N -> A |
| 308 | `diplodus_cervinus` | diplodus_cervinus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 309 | `diplodus_puntazzo` | diplodus_puntazzo:yolov8x_21sp_5364img:detections | detections | N -> A |
| 310 | `diplodus_sargus` | diplodus_sargus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 311 | `diplodus_vulgaris` | diplodus_vulgaris:yolov8x_21sp_5364img:detections | detections | N -> A |
| 312 | `epinephelus_costae` | epinephelus_costae:yolov8x_21sp_5364img:detections | detections | N -> A |
| 313 | `epinephelus_marginatus` | epinephelus_marginatus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 314 | `mullus_surmuletus` | mullus_surmuletus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 315 | `muraena_helena` | muraena_helena:yolov8x_21sp_5364img:detections | detections | N -> A |
| 316 | `oblada_melanurus` | oblada_melanurus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 317 | `parablennius_gattorugine` | parablennius_gattorugine:yolov8x_21sp_5364img:detections | detections | N -> A |
| 318 | `sarpa_salpa` | sarpa_salpa:yolov8x_21sp_5364img:detections | detections | N -> A |
| 319 | `seriola_dumerili` | seriola_dumerili:yolov8x_21sp_5364img:detections | detections | N -> A |
| 320 | `serranus_cabrilla` | serranus_cabrilla:yolov8x_21sp_5364img:detections | detections | N -> A |
| 321 | `sparus_aurata` | sparus_aurata:yolov8x_21sp_5364img:detections | detections | N -> A |
| 322 | `symphodus_mediterraneus` | symphodus_mediterraneus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 302 | `underwater_photography` | underwater_photography:files | files | 2024-05-29 -> 2024-07-29 |

#### 🔬 Instrumento (Sensor): `IPC608_CD3F_110`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 398 | `underwater_photography` | underwater_photography:files | files | 2025-02-20 -> 2025-02-27 |

#### 🔬 Instrumento (Sensor): `Lanty1-SN0000`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 430 | `aetomylaeus_bovinus` | aetomylaeus_bovinus:yolov8l_uib_plome_lanty_2024:detections | detections | N -> A |
| 431 | `chelon` | chelon:yolov8l_uib_plome_lanty_2024:detections | detections | N -> A |
| 432 | `chromis_chromis` | chromis_chromis:yolov8l_uib_plome_lanty_2024:detections | detections | N -> A |
| 433 | `coris_julis` | coris_julis:yolov8l_uib_plome_lanty_2024:detections | detections | N -> A |
| 434 | `dentex_dentex` | dentex_dentex:yolov8l_uib_plome_lanty_2024:detections | detections | N -> A |
| 435 | `diplodus_annularis` | diplodus_annularis:yolov8l_uib_plome_lanty_2024:detections | detections | N -> A |
| 436 | `diplodus_cervinus` | diplodus_cervinus:yolov8l_uib_plome_lanty_2024:detections | detections | N -> A |
| 437 | `diplodus_puntazzo` | diplodus_puntazzo:yolov8l_uib_plome_lanty_2024:detections | detections | N -> A |
| 438 | `diplodus_sargus` | diplodus_sargus:yolov8l_uib_plome_lanty_2024:detections | detections | N -> A |
| 439 | `diplodus_vulgaris` | diplodus_vulgaris:yolov8l_uib_plome_lanty_2024:detections | detections | N -> A |
| 440 | `epinephelus_marginatus` | epinephelus_marginatus:yolov8l_uib_plome_lanty_2024:detections | detections | N -> A |
| 441 | `lithognathus_mormyrus` | lithognathus_mormyrus:yolov8l_uib_plome_lanty_2024:detections | detections | N -> A |
| 442 | `muraena_helena` | muraena_helena:yolov8l_uib_plome_lanty_2024:detections | detections | N -> A |
| 443 | `myliobatidae` | myliobatidae:yolov8l_uib_plome_lanty_2024:detections | detections | N -> A |
| 444 | `oblada_melanura` | oblada_melanura:yolov8l_uib_plome_lanty_2024:detections | detections | N -> A |
| 445 | `pomatomus_saltatrix` | pomatomus_saltatrix:yolov8l_uib_plome_lanty_2024:detections | detections | N -> A |
| 446 | `sciaena_umbra` | sciaena_umbra:yolov8l_uib_plome_lanty_2024:detections | detections | N -> A |
| 447 | `seriola_dumerili` | seriola_dumerili:yolov8l_uib_plome_lanty_2024:detections | detections | N -> A |
| 448 | `serranus_cabrilla` | serranus_cabrilla:yolov8l_uib_plome_lanty_2024:detections | detections | N -> A |
| 449 | `serranus_scriba` | serranus_scriba:yolov8l_uib_plome_lanty_2024:detections | detections | N -> A |
| 450 | `sparus_aurata` | sparus_aurata:yolov8l_uib_plome_lanty_2024:detections | detections | N -> A |
| 451 | `spicara_maena` | spicara_maena:yolov8l_uib_plome_lanty_2024:detections | detections | N -> A |
| 452 | `spondyliosoma_cantharus` | spondyliosoma_cantharus:yolov8l_uib_plome_lanty_2024:detections | detections | N -> A |
| 453 | `symphodus_mediterraneus` | symphodus_mediterraneus:yolov8l_uib_plome_lanty_2024:detections | detections | N -> A |
| 454 | `unknown` | unknown:yolov8l_uib_plome_lanty_2024:detections | detections | N -> A |
| 428 | `underwater_photography` | underwater_photography:files | files | 2024-09-17 -> 2024-09-25 |

#### 🔬 Instrumento (Sensor): `NAXYS-SN0010`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| **210** | `SPL_all` | SPL_all:timeseries:30min | **30min** | 2020-04-06 -> 2026-01-15 |
| **211** | `SPL_63` | SPL_63:timeseries:30min | **30min** | 2020-04-06 -> 2026-01-15 |
| **212** | `SPL_125` | SPL_125:timeseries:30min | **30min** | 2020-04-06 -> 2026-01-15 |
| **213** | `SPL_2000` | SPL_2000:timeseries:30min | **30min** | 2020-04-06 -> 2026-01-15 |
| 209 | `acoustics` | acoustics:files | files | 2024-10-08 -> 2026-01-15 |
| 205 | `SPL_all` | SPL_all:timeseries:full | full | N -> A |
| 206 | `SPL_63` | SPL_63:timeseries:full | full | N -> A |
| 207 | `SPL_125` | SPL_125:timeseries:full | full | N -> A |
| 208 | `SPL_2000` | SPL_2000:timeseries:full | full | N -> A |

#### 🔬 Instrumento (Sensor): `PopUp_Cam-SN001`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 367 | `underwater_photography` | underwater_photography:files | files | 2024-12-10 -> 2025-06-25 |

#### 🔬 Instrumento (Sensor): `SBE16-SN57353-6479`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 107 | `TEMP` | TEMP:timeseries:1day | 1day | 2010-02-26 -> 2026-01-16 |
| 108 | `PRES` | PRES:timeseries:1day | 1day | 2010-02-26 -> 2026-01-16 |
| 109 | `CNDC` | CNDC:timeseries:1day | 1day | 2010-02-26 -> 2026-01-16 |
| 110 | `PSAL` | PSAL:timeseries:1day | 1day | 2010-02-26 -> 2026-01-16 |
| 111 | `SVEL` | SVEL:timeseries:1day | 1day | 2010-02-26 -> 2026-01-16 |
| **102** | `TEMP` | TEMP:timeseries:30min | **30min** | 2010-02-26 -> 2026-01-15 |
| **103** | `PRES` | PRES:timeseries:30min | **30min** | 2010-02-26 -> 2026-01-15 |
| **104** | `CNDC` | CNDC:timeseries:30min | **30min** | 2010-02-26 -> 2026-01-15 |
| **105** | `PSAL` | PSAL:timeseries:30min | **30min** | 2010-02-26 -> 2026-01-15 |
| **106** | `SVEL` | SVEL:timeseries:30min | **30min** | 2010-02-26 -> 2026-01-15 |
| 97 | `TEMP` | TEMP:timeseries:full | full | N -> A |
| 98 | `PRES` | PRES:timeseries:full | full | N -> A |
| 99 | `CNDC` | CNDC:timeseries:full | full | N -> A |
| 100 | `PSAL` | PSAL:timeseries:full | full | N -> A |
| 101 | `SVEL` | SVEL:timeseries:full | full | N -> A |

#### 🔬 Instrumento (Sensor): `SBE37SI-SN37-24580`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| **392** | `TEMP` | TEMP:timeseries:30min | **30min** | 2025-01-22 -> 2025-09-12 |
| **393** | `PRES` | PRES:timeseries:30min | **30min** | 2025-01-22 -> 2025-09-12 |
| **394** | `CNDC` | CNDC:timeseries:30min | **30min** | 2025-01-22 -> 2025-09-12 |
| **395** | `PSAL` | PSAL:timeseries:30min | **30min** | 2025-01-22 -> 2025-09-12 |
| **396** | `SVEL` | SVEL:timeseries:30min | **30min** | 2025-01-22 -> 2025-09-12 |
| 385 | `TEMP` | TEMP:timeseries:full | full | N -> A |
| 386 | `PRES` | PRES:timeseries:full | full | N -> A |
| 387 | `CNDC` | CNDC:timeseries:full | full | N -> A |
| 388 | `PSAL` | PSAL:timeseries:full | full | N -> A |
| 389 | `SVEL` | SVEL:timeseries:full | full | N -> A |

#### 🔬 Instrumento (Sensor): `SBE37SMP-SN47472-5496`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 137 | `TEMP` | TEMP:timeseries:1day | 1day | 2009-05-29 -> 2025-07-12 |
| 138 | `PRES` | PRES:timeseries:1day | 1day | 2009-05-29 -> 2025-07-12 |
| 139 | `CNDC` | CNDC:timeseries:1day | 1day | 2009-05-29 -> 2025-07-12 |
| 140 | `PSAL` | PSAL:timeseries:1day | 1day | 2009-05-29 -> 2025-07-12 |
| 141 | `SVEL` | SVEL:timeseries:1day | 1day | 2009-05-29 -> 2025-07-12 |
| **132** | `TEMP` | TEMP:timeseries:30min | **30min** | 2012-03-30 -> 2025-07-11 |
| **133** | `PRES` | PRES:timeseries:30min | **30min** | 2012-03-30 -> 2025-07-11 |
| **134** | `CNDC` | CNDC:timeseries:30min | **30min** | 2012-03-30 -> 2025-07-11 |
| **135** | `PSAL` | PSAL:timeseries:30min | **30min** | 2009-05-29 -> 2025-07-11 |
| **136** | `SVEL` | SVEL:timeseries:30min | **30min** | 2012-03-30 -> 2025-07-11 |
| 127 | `TEMP` | TEMP:timeseries:full | full | N -> A |
| 128 | `PRES` | PRES:timeseries:full | full | N -> A |
| 129 | `CNDC` | CNDC:timeseries:full | full | N -> A |
| 130 | `PSAL` | PSAL:timeseries:full | full | N -> A |
| 131 | `SVEL` | SVEL:timeseries:full | full | N -> A |

#### 🔬 Instrumento (Sensor): `Sony_SNC-RZ25N`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| 348 | `aetomylaeus_bovinus` | aetomylaeus_bovinus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 349 | `chromis_chromis` | chromis_chromis:yolov8x_21sp_5364img:detections | detections | N -> A |
| 350 | `coris_julis` | coris_julis:yolov8x_21sp_5364img:detections | detections | N -> A |
| 351 | `dentex_dentex` | dentex_dentex:yolov8x_21sp_5364img:detections | detections | N -> A |
| 352 | `diplodus_cervinus` | diplodus_cervinus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 353 | `diplodus_puntazzo` | diplodus_puntazzo:yolov8x_21sp_5364img:detections | detections | N -> A |
| 354 | `diplodus_sargus` | diplodus_sargus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 355 | `diplodus_vulgaris` | diplodus_vulgaris:yolov8x_21sp_5364img:detections | detections | N -> A |
| 356 | `epinephelus_costae` | epinephelus_costae:yolov8x_21sp_5364img:detections | detections | N -> A |
| 357 | `epinephelus_marginatus` | epinephelus_marginatus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 358 | `mullus_surmuletus` | mullus_surmuletus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 359 | `muraena_helena` | muraena_helena:yolov8x_21sp_5364img:detections | detections | N -> A |
| 360 | `oblada_melanurus` | oblada_melanurus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 361 | `parablennius_gattorugine` | parablennius_gattorugine:yolov8x_21sp_5364img:detections | detections | N -> A |
| 362 | `sarpa_salpa` | sarpa_salpa:yolov8x_21sp_5364img:detections | detections | N -> A |
| 363 | `seriola_dumerili` | seriola_dumerili:yolov8x_21sp_5364img:detections | detections | N -> A |
| 364 | `serranus_cabrilla` | serranus_cabrilla:yolov8x_21sp_5364img:detections | detections | N -> A |
| 365 | `sparus_aurata` | sparus_aurata:yolov8x_21sp_5364img:detections | detections | N -> A |
| 366 | `symphodus_mediterraneus` | symphodus_mediterraneus:yolov8x_21sp_5364img:detections | detections | N -> A |
| 346 | `underwater_photography` | underwater_photography:files | files | 2009-06-01 -> 2020-03-11 |

### 📍 Plataforma (Thing): `OBSEA_Besos_Buoy`
#### 🔬 Instrumento (Sensor): `Airmar_200WX-SN60390327`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| **92** | `CAPH` | CAPH:timeseries:30min | **30min** | 2022-08-02 -> 2026-03-11 |
| **93** | `AIRT` | AIRT:timeseries:30min | **30min** | 2022-08-02 -> 2026-03-11 |
| **94** | `WDIR` | WDIR:timeseries:30min | **30min** | 2022-08-02 -> 2026-03-11 |
| **95** | `WSPD` | WSPD:timeseries:30min | **30min** | 2022-08-02 -> 2026-03-11 |
| 88 | `CAPH` | CAPH:timeseries:full | full | N -> A |
| 89 | `AIRT` | AIRT:timeseries:full | full | N -> A |
| 90 | `WDIR` | WDIR:timeseries:full | full | N -> A |
| 91 | `WSPD` | WSPD:timeseries:full | full | N -> A |

#### 🔬 Instrumento (Sensor): `C3-SN2300642`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| **6** | `CDOM` | CDOM:timeseries:30min | **30min** | 2022-08-02 -> 2024-07-11 |
| **7** | `COIL` | COIL:timeseries:30min | **30min** | 2022-08-02 -> 2024-07-29 |
| **8** | `RFUL` | RFUL:timeseries:30min | **30min** | 2022-08-02 -> 2024-07-29 |
| **9** | `TEMP` | TEMP:timeseries:30min | **30min** | 2022-08-02 -> 2024-07-29 |
| 2 | `CDOM` | CDOM:timeseries:full | full | N -> A |
| 3 | `COIL` | COIL:timeseries:full | full | N -> A |
| 4 | `RFUL` | RFUL:timeseries:full | full | N -> A |
| 5 | `TEMP` | TEMP:timeseries:full | full | N -> A |

#### 🔬 Instrumento (Sensor): `SA8065-SN0000`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| **120** | `DEPTH` | DEPTH:timeseries:30min | **30min** | 2022-10-28 -> 2024-07-29 |
| **121** | `TEMP` | TEMP:timeseries:30min | **30min** | 2022-10-28 -> 2024-07-29 |
| **122** | `CNDC` | CNDC:timeseries:30min | **30min** | 2022-10-28 -> 2024-07-29 |
| **123** | `DOXY` | DOXY:timeseries:30min | **30min** | 2022-10-28 -> 2024-07-29 |
| **124** | `ALKY` | ALKY:timeseries:30min | **30min** | 2022-10-28 -> 2024-07-29 |
| **125** | `WBRX` | WBRX:timeseries:30min | **30min** | 2022-10-28 -> 2024-07-29 |
| **126** | `TURB` | TURB:timeseries:30min | **30min** | 2022-10-28 -> 2024-07-29 |
| 113 | `DEPTH` | DEPTH:timeseries:full | full | N -> A |
| 114 | `TEMP` | TEMP:timeseries:full | full | N -> A |
| 115 | `CNDC` | CNDC:timeseries:full | full | N -> A |
| 116 | `DOXY` | DOXY:timeseries:full | full | N -> A |
| 117 | `ALKY` | ALKY:timeseries:full | full | N -> A |
| 118 | `WBRX` | WBRX:timeseries:full | full | N -> A |
| 119 | `TURB` | TURB:timeseries:full | full | N -> A |

### 📍 Plataforma (Thing): `OBSEA_Buoy`
#### 🔬 Instrumento (Sensor): `Airmar_150WX-SN57323`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| **56** | `CAPH` | CAPH:timeseries:30min | **30min** | 2011-10-06 -> 2021-12-09 |
| **57** | `AIRT` | AIRT:timeseries:30min | **30min** | 2011-10-06 -> 2021-12-09 |
| **58** | `WDIR` | WDIR:timeseries:30min | **30min** | 2011-10-06 -> 2021-12-09 |
| **59** | `WSPD` | WSPD:timeseries:30min | **30min** | 2011-10-06 -> 2021-12-09 |
| 52 | `CAPH` | CAPH:timeseries:full | full | N -> A |
| 53 | `AIRT` | AIRT:timeseries:full | full | N -> A |
| 54 | `WDIR` | WDIR:timeseries:full | full | N -> A |
| 55 | `WSPD` | WSPD:timeseries:full | full | N -> A |

#### 🔬 Instrumento (Sensor): `C3-SN2300642`
| Datastream ID | Variable (Property) | Datastream Name | Res. Temporal | Cobertura Histórica |
|---------------|---------------------|-----------------|---------------|---------------------|
| **473** | `CDOM` | CDOM:timeseries:30min | **30min** | N -> A |
| **474** | `COIL` | COIL:timeseries:30min | **30min** | N -> A |
| **475** | `RFUL` | RFUL:timeseries:30min | **30min** | N -> A |
| **476** | `TEMP` | TEMP:timeseries:30min | **30min** | N -> A |
| 469 | `CDOM` | CDOM:timeseries:full | full | N -> A |
| 470 | `COIL` | COIL:timeseries:full | full | N -> A |
| 471 | `RFUL` | RFUL:timeseries:full | full | N -> A |
| 472 | `TEMP` | TEMP:timeseries:full | full | N -> A |

---

> *Nota de Filtro: Se han omitido los Datastreams correspondientes a las métricas algorítmicas de Computer Vision individuales (YOLOv8) para las ~21 especies (Coris julis, Sparus aurata, Muraena helena, etc.) para mantener este informe manejable y centrado en las variables Físicas, Meteorológicas y Oceanográficas de Interés Científico Core.*
