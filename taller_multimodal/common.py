"""Utilidades compartidas por las partes del taller: cliente Groq, imágenes en base64 y registro de salida."""
from __future__ import annotations

import base64
import contextlib
import io
import json
import mimetypes
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parent
IMAGENES = AQUI / "imagenes"
AUDIOS = AQUI / "audios"
RESULTADOS = AQUI / "resultados"

# La clave se lee del .env de la raíz del repositorio (nunca se escribe en el código).
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))  # para poder importar el paquete rag/ del proyecto

VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.8-27b")
TEXT_MODEL = os.getenv("TEXT_MODEL", "openai/gpt-oss-120b")  # modelo sin visión (para el error provocado)
AUDIO_MODEL = os.getenv("AUDIO_MODEL", "whisper-large-v3-turbo")


def cliente():
    """Cliente oficial de Groq. Lee GROQ_API_KEY del entorno."""
    from groq import Groq

    if not os.getenv("GROQ_API_KEY"):
        sys.exit("Falta GROQ_API_KEY: cópiala en el archivo .env de la raíz del repositorio.")
    return Groq(max_retries=3)


def data_url(ruta: Path, mime: str | None = None) -> str:
    """Codifica una imagen local en base64 como data URL. `mime` permite declarar un tipo distinto (error 1.4)."""
    mime = mime or mimetypes.guess_type(ruta.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(ruta.read_bytes()).decode()}"


def parte_imagen(ruta: Path, mime: str | None = None) -> dict:
    return {"type": "image_url", "image_url": {"url": data_url(ruta, mime)}}


def mensaje_usuario(texto: str, imagenes: list[Path]) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": texto}, *[parte_imagen(p) for p in imagenes]]}


def guardar_json(nombre: str, datos) -> Path:
    RESULTADOS.mkdir(exist_ok=True)
    ruta = RESULTADOS / nombre
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    return ruta


def titulo(texto: str) -> None:
    print("\n" + "=" * 78 + f"\n{texto}\n" + "=" * 78)


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


@contextlib.contextmanager
def registrar(nombre: str):
    """Muestra la salida en pantalla y la guarda en resultados/<nombre>.txt (sirve como evidencia/captura)."""
    RESULTADOS.mkdir(exist_ok=True)
    with open(RESULTADOS / f"{nombre}.txt", "w", encoding="utf-8") as f:
        f.write(f"# Ejecución: {datetime.now():%Y-%m-%d %H:%M}\n")
        viejo, sys.stdout = sys.stdout, _Tee(sys.stdout, f)
        try:
            yield
        finally:
            sys.stdout = viejo


def describir_error_api(exc: Exception) -> dict:
    """Extrae el código HTTP y el mensaje que devuelve la API (o el tipo de excepción si no hubo respuesta)."""
    info = {"tipo": type(exc).__name__, "status_code": getattr(exc, "status_code", None)}
    cuerpo = getattr(exc, "body", None)
    if isinstance(cuerpo, dict):
        err = cuerpo.get("error", cuerpo)
        info["mensaje"] = err.get("message") if isinstance(err, dict) else str(err)
        info["codigo"] = err.get("code") if isinstance(err, dict) else None
        info["cuerpo"] = cuerpo
    else:
        info["mensaje"] = str(exc)
    return info
