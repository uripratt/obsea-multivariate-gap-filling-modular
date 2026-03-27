import urllib.request
import urllib.error
import urllib.parse
import json
import logging
import time
import pandas as pd
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# STA API Resilience Configuration
_MAX_RETRIES = 3
_BASE_DELAY_S = 2
_REQUEST_TIMEOUT_S = 30


class STAConnector:
    """
    OGC SensorThings API (STA) v1.1 Connector para OBSEA.
    Maneja la lógica de paginación implícita (@iot.nextLink) para evitar el límite de 100 
    registros devueltos por el servidor Pangea.
    """
    
    # =========================================================================
    # DATASTREAM MAP - Auditoría STA v1.1 verificada (11/03/2026)
    # Resolución: 30 minutos. IDs extraídos de la auditoría exhaustiva de la API.
    # =========================================================================
    
    # Instrumento 1: CTD SBE16-SN57353-6479 (Thing: OBSEA)
    # Cobertura histórica: 2010-02-26 -> 2026-01-15 (~16 años)
    DATASTREAM_CTD_SBE16 = {
        'TEMP': 102,  # Temperatura del agua de mar
        'PRES': 103,  # Presión (profundidad)
        'CNDC': 104,  # Conductividad eléctrica
        'PSAL': 105,  # Salinidad práctica
        'SVEL': 106,  # Velocidad del sonido
    }

    # Instrument 1.1: CTD SBE37SMP-SN47472-5496 (Thing: OBSEA)
    # Cobertura histórica: 2009-05-29 -> 2025-07-11
    # Nota: Usamos la resolución 'full' (IDs 127-131) para asegurar cobertura desde 2009.
    DATASTREAM_CTD_SBE37 = {
        'TEMP': 127,
        'PRES': 128,
        'CNDC': 129,
        'PSAL': 130,
        'SVEL': 131,
    }
    
    # Instrumento 2: AWAC-SN5931 (Thing: OBSEA)
    # Cobertura histórica: 2010-04-08 -> 2025-04-17 (~15 años)
    # Perfiles ADCP con múltiples bins de profundidad. Seleccionamos dos niveles:
    #   - Superficie (2m): Capa superficial influida por oleaje y viento
    #   - Fondo (18m): Capa cercana al fondo (~20m) de interés para OBSEA
    # Los Datastream IDs son los mismos; el filtro se aplica por `parameters.depth`
    
    AWAC_DEPTH_BINS = {
        'AWAC_2M':  2.0,   # Bin de superficie
        'AWAC_18M': 18.0,  # Bin de fondo
    }
    
    # Base IDs del AWAC (comunes a todos los bins)
    _AWAC_BASE_IDS = {
        'CSPD': 189,  # Velocidad de corriente marina
        'CDIR': 190,  # Dirección de corriente marina
        'UCUR': 191,  # Componente Este de la corriente
        'VCUR': 192,  # Componente Norte de la corriente
        'ZCUR': 193,  # Componente Vertical de la corriente
    }
    
    # Variables desplegadas con prefijo de profundidad
    DATASTREAM_AWAC_2M = {f'AWAC2M_{k}': v for k, v in _AWAC_BASE_IDS.items()}
    DATASTREAM_AWAC_18M = {f'AWAC18M_{k}': v for k, v in _AWAC_BASE_IDS.items()}
    
    # Instrumento 3: Airmar_200WX-SN60390327 (Thing: OBSEA_Besos_Buoy)
    # Cobertura histórica: 2022-08-02 -> 2026-03-11 (boya justo encima de OBSEA)
    DATASTREAM_BUOY_METEO = {
        'BUOY_CAPH': 92,   # Presión atmosférica (boya)
        'BUOY_AIRT': 93,   # Temperatura del aire (boya)
        'BUOY_WDIR': 94,   # Dirección del viento (boya)
        'BUOY_WSPD': 95,   # Velocidad del viento (boya)
    }
    
    # Instrumento 4: Vantage_Pro2-SN6150CEU (Thing: CTVG)
    # Cobertura histórica: 2010-04-08 -> 2025-02-12 (~15 años, estación terrestre a 4km)
    DATASTREAM_CTVG_METEO = {
        'CTVG_CAPH': 200,  # Presión atmosférica (tierra)
        'CTVG_RELH': 201,  # Humedad relativa (tierra)
        'CTVG_AIRT': 202,  # Temperatura del aire (tierra)
        'CTVG_WDIR': 203,  # Dirección del viento (tierra)
        'CTVG_WSPD': 204,  # Velocidad del viento (tierra)
    }
    
    # Mapa unificado con TODOS los Datastreams seleccionados
    DATASTREAM_MAP_30MIN = {
        **DATASTREAM_CTD_SBE16,
        **DATASTREAM_CTD_SBE37,
        **DATASTREAM_AWAC_2M,
        **DATASTREAM_AWAC_18M,
        **DATASTREAM_BUOY_METEO,
        **DATASTREAM_CTVG_METEO,
    }
    
    # Agrupación por instrumento para iterar selectivamente
    # Los grupos AWAC llevan un campo 'depth_bin' especial
    INSTRUMENT_GROUPS = {
        'CTD_SBE16':  DATASTREAM_CTD_SBE16,
        'CTD_SBE37':  DATASTREAM_CTD_SBE37,
        'AWAC_2M':    DATASTREAM_AWAC_2M,
        'AWAC_18M':   DATASTREAM_AWAC_18M,
        'BUOY_METEO': DATASTREAM_BUOY_METEO,
        'CTVG_METEO': DATASTREAM_CTVG_METEO,
    }

    def __init__(self, base_url: str = "https://data.obsea.es/sta-timeseries/v1.1"):
        self.base_url = base_url.rstrip("/")

    def get_datastream_id(self, variable_name: str) -> int:
        """Traduce una variable de alto nivel (ej. 'TEMP') a su ID en la API resolucion 30min"""
        ds_id = self.DATASTREAM_MAP_30MIN.get(variable_name.upper())
        if not ds_id:
            raise KeyError(f"La variable {variable_name} no está en el registro 30min de STAConnector.")
        return ds_id

    def fetch_observations(self, datastream_id: int, start_time: str = None, end_time: str = None, depth_bin: float = None) -> pd.DataFrame:
        """
        Descarga iterativamente todas las observaciones (paginando con @iot.nextLink).
        Las fechas start_time y end_time deben ser formato ISO-8601, ej. '2024-01-01T00:00:00Z'.
        
        Parameters
        ----------
        depth_bin : float, optional
            Si se especifica, solo devuelve observaciones de esta profundidad exacta.
            Útil para AWAC/ADCP profiles que contienen múltiples bins por timestamp.
        """
        
        # Construimos la query (ordenado temporalmente ascendente para rellenar el DataFrame cronológico)
        url = f"{self.base_url}/Datastreams({datastream_id})/Observations?$orderBy=phenomenonTime%20asc"
        
        # Aplicar filtros temporales si existen (OGC Filter syntax)
        filters = []
        if start_time:
            filters.append(f"phenomenonTime ge {start_time}")
        if end_time:
            filters.append(f"phenomenonTime le {end_time}")
        
        if filters:
            # Junta filtros con ' and '
            filter_query = urllib.parse.quote(" and ".join(filters))
            url += f"&$filter={filter_query}"

        logger.info(f"Conectando a STA: Datastream {datastream_id} ...")
        
        all_results = []
        next_link = url

        # Bucle de paginación con retry + exponential backoff
        page_count = 0
        while next_link:
            data = None
            for attempt in range(_MAX_RETRIES):
                try:
                    req = urllib.request.Request(next_link, headers={'User-Agent': 'OBSEA-Pipeline/2.0'})
                    with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as response:
                        data = json.loads(response.read().decode('utf-8'))
                        break  # Success, exit retry loop
                except (urllib.error.URLError, TimeoutError, ConnectionResetError) as e:
                    if attempt < _MAX_RETRIES - 1:
                        delay = _BASE_DELAY_S * (2 ** attempt)
                        logger.warning(f"  [STA Retry {attempt+1}/{_MAX_RETRIES}] {e} — waiting {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"  [STA FAIL] Permanent failure after {_MAX_RETRIES} retries on page {page_count+1}: {e}")
                        if hasattr(e, 'read'):
                            try: logger.error(f"  Response: {e.read().decode('utf-8')[:500]}")
                            except: pass
            
            if data is None:
                logger.warning(f"  Aborting pagination at page {page_count+1}. Collected {len(all_results)} records so far.")
                break
                    
            if 'value' in data:
                all_results.extend(data['value'])
                
            page_count += 1
            if page_count % 10 == 0:
                logger.info(f"    ... fetched {page_count} pages ({len(all_results):,} records)")
            
            # Chequear si existe la siguiente página y arreglar bug del servidor ($orderby vs $orderBy)
            next_link = data.get('@iot.nextLink')
            if next_link:
                import re
                next_link = next_link.replace("$orderby=", "$orderBy=")
                # Arreglar bug critico 2: El servidor inyecta $skipFilter que el mismo rechaza luego
                next_link = re.sub(r'&\$skipFilter=[^&]+', '', next_link)

                
        if not all_results:
            logger.warning(f"No se han encontrado observaciones para el Datastream {datastream_id}.")
            return pd.DataFrame()

        # Filtrar por bin de profundidad si se especifica (para perfiles ADCP)
        if depth_bin is not None:
            all_results = [
                obs for obs in all_results
                if obs.get('parameters', {}).get('depth') == depth_bin
            ]
            if not all_results:
                logger.warning(f"No hay observaciones para bin de profundidad {depth_bin}m en Datastream {datastream_id}.")
                return pd.DataFrame()

        # Transformar a DataFrame
        df = pd.DataFrame([{
            # Extraemos el stamp inicial si viene como intervalo tipo '2023-04-30T00:00:00Z/2023-04-30T00:30:00Z'
            'Timestamp': str(obs.get('phenomenonTime')).split('/')[0],
            'Value': obs.get('result')
        } for obs in all_results])
        
        # Clean Pandas TimeIndex mapping (mismo formato que lup_data_obsea_analysis_jupyterhub.py)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
        # Quitar la información de zona timezone (como el csv local original)
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
            
        return df

