import logging

logger = logging.getLogger(__name__)

def get_device():
    """Detectar dispositivo de forma robusta."""
    try:
        import torch
        if torch.cuda.is_available():
            # Test real allocation (algunas máquinas reportan CUDA disponible pero fallan al usarlo)
            _ = torch.zeros(1).cuda()
            return 'cuda'
    except Exception as e:
        logger.debug(f"CUDA validation failed: {e}. Falling back to CPU.")
    return 'cpu'
