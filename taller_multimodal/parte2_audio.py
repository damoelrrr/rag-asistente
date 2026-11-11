"""Parte 2 del taller: audio → transcripción (Whisper en Groq) → respuesta del tutor RAG del proyecto.

Lee los audios de taller_multimodal/audios/ (wav, mp3, m4a, ogg, webm...). Los metadatos de cada audio
(hablante, condición y lo que se dijo de verdad) están en audios/guion.json; el archivo se asocia por el
prefijo de su nombre: audio1_*.m4a → "audio1".

Para cada audio:
  2.1  Transcribe el archivo y mide el error contra lo que se dijo (WER, tasa de error por palabra).
  2.2  Envía la transcripción al asistente RAG (rag/pipeline.py) y muestra la respuesta con sus fuentes.
       Como control, también responde a partir del texto real para ver si los errores de transcripción
       cambiaron la respuesta.

Uso:
    python taller_multimodal/parte2_audio.py
    python taller_multimodal/parte2_audio.py --solo-transcripcion
    python taller_multimodal/parte2_audio.py --comparar-modelos   # también prueba whisper-large-v3
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

from common import AUDIO_MODEL, AUDIOS, cliente, describir_error_api, guardar_json, registrar, titulo

EXTENSIONES = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac", ".mp4", ".mpeg", ".mpga", ".opus"}
LIMITE_MB = 25  # plan gratuito de Groq


def normalizar(texto: str) -> list[str]:
    s = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode()
    s = s.replace("-", " ")  # "TF-IDF" y "TF IDF" cuentan igual
    return re.sub(r"[^a-z0-9 ]", " ", s).split()


def wer(referencia: str, hipotesis: str) -> tuple[float, int, int]:
    """Word Error Rate = (sustituciones + borrados + inserciones) / palabras de la referencia."""
    r, h = normalizar(referencia), normalizar(hipotesis)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return (d[len(h)] / max(len(r), 1)), d[len(h)], len(r)


# Glosario opcional: Whisper usa el `prompt` como contexto previo, lo que ayuda a escribir bien los términos técnicos
VOCABULARIO = ("Tutor del curso de IA. Términos: TF-IDF, BM25, k1, RAG, LLM, Ollama, Llama 3, API, indexing, chunks, "
               "embeddings, base de datos vectorial, prompts, pgvector, PostgreSQL. Hablantes: Stiven y Daniel.")


def transcribir(c, ruta, modelo: str, prompt: str | None = None) -> tuple[str, float | None, float]:
    t0 = time.time()
    extra = {"prompt": prompt} if prompt else {}
    with open(ruta, "rb") as f:
        r = c.audio.transcriptions.create(
            file=(ruta.name, f.read()), model=modelo, language="es", response_format="verbose_json", temperature=0,
            **extra,
        )
    datos = r.model_dump() if hasattr(r, "model_dump") else dict(r)
    return (datos.get("text") or "").strip(), datos.get("duration"), time.time() - t0


def buscar_audios(guion: dict, carpeta=AUDIOS):
    archivos = sorted(p for p in carpeta.iterdir() if p.suffix.lower() in EXTENSIONES)
    for ruta in archivos:
        clave = next((k for k in guion if ruta.stem.lower().startswith(k.lower())), None)
        yield ruta, clave, guion.get(clave, {})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solo-transcripcion", action="store_true", help="no llama al asistente RAG (solo 2.1)")
    ap.add_argument("--comparar-modelos", action="store_true", help="también transcribe con whisper-large-v3")
    ap.add_argument("--vocabulario", action="store_true",
                    help="también transcribe con un glosario de términos técnicos en el parámetro prompt")
    ap.add_argument("--carpeta", type=Path, default=AUDIOS, help="carpeta con los audios (por defecto audios/)")
    a = ap.parse_args()

    guion = json.loads((AUDIOS / "guion.json").read_text(encoding="utf-8"))
    audios = list(buscar_audios(guion, a.carpeta))
    if not audios:
        raise SystemExit(f"No hay audios en {AUDIOS}. Graben los audios según GUION_AUDIOS.md y vuelvan a ejecutar.")

    c = cliente()
    # (nombre de la variante, modelo, prompt). La primera es la que se envía al tutor en 2.2.
    modelos = [(AUDIO_MODEL, AUDIO_MODEL, None)]
    if a.comparar_modelos and AUDIO_MODEL != "whisper-large-v3":
        modelos.append(("whisper-large-v3", "whisper-large-v3", None))
    if a.vocabulario:
        modelos.append((f"{AUDIO_MODEL}+vocabulario", AUDIO_MODEL, VOCABULARIO))
    resultados = []

    with registrar("parte2_transcripcion"):
        titulo(f"2.1 TRANSCRIPCIÓN DESDE ARCHIVO   (variantes: {', '.join(m[0] for m in modelos)})")
        if a.vocabulario:
            print(f"Glosario usado como prompt en la variante +vocabulario: {VOCABULARIO}")
        for ruta, clave, meta in audios:
            mb = ruta.stat().st_size / 1024 / 1024
            print(f"\n--- {ruta.name}  ({mb:.2f} MB) | hablante: {meta.get('hablante', '?')} | "
                  f"condición: {meta.get('condicion', '?')} ---")
            item = {"audio": ruta.name, "clave": clave, **meta, "mb": round(mb, 2), "transcripciones": {}}
            if mb > LIMITE_MB:
                print(f"El archivo supera {LIMITE_MB} MB (límite del plan gratuito): no se envía.")
                resultados.append(item)
                continue
            for modelo, nombre_modelo, prompt in modelos:
                try:
                    texto, duracion, seg = transcribir(c, ruta, nombre_modelo, prompt)
                except Exception as e:
                    info = describir_error_api(e)
                    print(f"[{modelo}] Error de la API: HTTP {info['status_code']} {info['mensaje']}")
                    item["transcripciones"][modelo] = {"error": info}
                    continue
                t = {"texto": texto, "duracion_s": duracion, "segundos_api": round(seg, 2)}
                print(f"[{modelo}] duración del audio: {duracion} s | tiempo de respuesta: {seg:.1f} s")
                print(f"  Transcripción: {texto}")
                if meta.get("texto"):
                    tasa, errores, n = wer(meta["texto"], texto)
                    t.update(wer=round(tasa, 3), errores=errores, palabras=n)
                    print(f"  Lo que se dijo: {meta['texto']}")
                    print(f"  WER: {tasa:.1%} ({errores} errores en {n} palabras)")
                item["transcripciones"][modelo] = t
            resultados.append(item)
        titulo("Resumen: WER promedio por variante")
        for modelo, *_ in modelos:
            tasas = [it["transcripciones"][modelo]["wer"] for it in resultados if "wer" in it["transcripciones"].get(modelo, {})]
            if tasas:
                print(f"  {modelo:40s} {sum(tasas) / len(tasas):6.1%}   (por audio: {', '.join(f'{t:.0%}' for t in tasas)})")
        guardar_json("2_1_transcripciones.json", resultados)

    if a.solo_transcripcion:
        return

    from rag.pipeline import RAGPipeline

    rag = RAGPipeline()
    rag.ensure_index()
    with registrar("parte2_respuestas"):
        llm = rag.s.llm_model or rag.s.default_model(rag.s.llm_provider)
        titulo(f"2.2 RESPUESTA DEL TUTOR RAG A PARTIR DE LA TRANSCRIPCIÓN   (LLM: {rag.s.llm_provider} / {llm})")
        for item in resultados:
            t = item["transcripciones"].get(AUDIO_MODEL, {})
            if not t.get("texto"):
                continue
            print(f"\n--- {item['audio']} ---\nAudio → transcripción: {t['texto']}")
            resp = rag.answer(t["texto"])
            fuentes = [f"[{h.n}] {h.source} p.{h.page}" for h in resp.hits if h.n in resp.cited] or \
                      [f"[{h.n}] {h.source} p.{h.page}" for h in resp.hits]
            print(f"Consulta usada para buscar: {resp.standalone}")
            print(f"Transcripción → respuesta ({resp.seconds:.1f} s){' [SIN CONTEXTO: se abstuvo]' if resp.refused else ''}:\n{resp.text}")
            print("Fuentes:", "; ".join(fuentes) or "ninguna")
            item["respuesta"] = {"texto": resp.text, "rechazo": resp.refused, "fuentes": fuentes,
                                 "consulta": resp.standalone, "segundos": round(resp.seconds, 2)}
            # Control: la misma pregunta con el texto real, para ver si la transcripción cambió la respuesta
            if item.get("texto") and t.get("errores"):
                control = rag.answer(item["texto"])
                pag = lambda hits: sorted({f"{h.source} p.{h.page}" for h in hits})  # noqa: E731
                propios, de_control = pag(resp.hits), pag(control.hits)
                print(f"Control con el texto real → mismos fragmentos (archivo y página): {propios == de_control}")
                if propios != de_control:
                    print(f"  Solo con la transcripción: {sorted(set(propios) - set(de_control)) or 'ninguno'}")
                    print(f"  Solo con el texto real:    {sorted(set(de_control) - set(propios)) or 'ninguno'}")
                print(f"  Respuesta de control: {control.text}")
                item["respuesta_control"] = {"texto": control.text, "rechazo": control.refused, "fuentes": de_control,
                                             "fuentes_transcripcion": propios}
        guardar_json("2_2_respuestas.json", resultados)


if __name__ == "__main__":
    main()
