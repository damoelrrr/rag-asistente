"""Parte 1 del taller: visión sobre imágenes del dominio (material del curso que usa el tutor RAG).

Subcomandos:
    pregunta    1.1  Una imagen local en base64 + una pregunta del dominio.
    extraccion  1.2  Extracción estructurada a JSON, validación con Pydantic y % de aciertos.
    multiples   1.3  Varias imágenes relacionadas en una sola solicitud.
    error       1.4  Errores provocados a propósito y el mensaje de la API.
    todo        Ejecuta las cuatro partes en orden.

Ejemplos:
    python taller_multimodal/parte1_vision.py todo
    python taller_multimodal/parte1_vision.py extraccion
    python taller_multimodal/parte1_vision.py error --caso modelo_sin_vision
"""
from __future__ import annotations

import argparse
import io
import json
import re
import time
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from common import (
    IMAGENES,
    TEXT_MODEL,
    VISION_MODEL,
    cliente,
    describir_error_api,
    guardar_json,
    mensaje_usuario,
    parte_imagen,
    registrar,
    titulo,
)

# ---------------------------------------------------------------------------
# 1.1  Pregunta sobre una imagen
# ---------------------------------------------------------------------------
PREGUNTA_IMAGEN = IMAGENES / "06_survey_tabla3_metricas.png"
PREGUNTA = (
    "Esta es la Tabla III del survey 'Retrieval-Augmented Generation for Large Language Models'. "
    "Según la tabla, ¿qué métricas sirven para evaluar el aspecto 'Context Relevance' y cuáles sirven para "
    "'Faithfulness'? Responde en español con dos listas y no agregues métricas que no estén marcadas."
)
METRICAS_TABLA = ["Accuracy", "EM", "Recall", "Precision", "R-Rate", "Cosine Similarity", "Hit Rate", "MRR",
                  "NDCG", "BLEU", "ROUGE/ROUGE-L"]
RESPUESTA_ESPERADA = {
    "Context Relevance": ["Accuracy", "Recall", "Precision", "Hit Rate", "MRR", "NDCG", "BLEU", "ROUGE/ROUGE-L"],
    "Faithfulness": ["Accuracy", "BLEU", "ROUGE/ROUGE-L"],
}


def preguntar(c, texto: str, imagenes, modelo: str = VISION_MODEL, **extra) -> tuple[str, float, dict]:
    t0 = time.time()
    r = c.chat.completions.create(
        model=modelo, messages=[mensaje_usuario(texto, imagenes)], temperature=0, max_completion_tokens=1500, **extra
    )
    uso = r.usage.model_dump() if r.usage else {}
    return r.choices[0].message.content or "", time.time() - t0, uso


def parte_pregunta(c) -> None:
    titulo(f"1.1 PREGUNTA SOBRE UNA IMAGEN   (modelo: {VISION_MODEL})")
    print(f"Imagen:   {PREGUNTA_IMAGEN.name}  ({PREGUNTA_IMAGEN.stat().st_size / 1024:.0f} KB, enviada en base64)")
    print(f"Pregunta: {PREGUNTA}\n")
    resp, seg, uso = preguntar(c, PREGUNTA, [PREGUNTA_IMAGEN])
    print(f"Respuesta del modelo ({seg:.1f} s, tokens de entrada: {uso.get('prompt_tokens')}):\n{resp}\n")
    print("Respuesta esperada (leída en la imagen):")
    for aspecto, metricas in RESPUESTA_ESPERADA.items():
        print(f"  - {aspecto}: {', '.join(metricas)}")
    # Verificación lista por lista: se divide la respuesta en la sección de cada aspecto
    corte = resp.lower().rfind("faithfulness")
    secciones = {"Context Relevance": resp[:corte], "Faithfulness": resp[corte:]}
    evaluacion, aciertos, total = {}, 0, 0
    print("\nVerificación por aspecto (11 métricas posibles en la tabla):")
    for aspecto, esperadas in RESPUESTA_ESPERADA.items():
        dichas = {m for m in METRICAS_TABLA if re.search(rf"\b{re.escape(m.split('/')[0])}\b", secciones[aspecto], re.I)}
        faltan, sobran = sorted(set(esperadas) - dichas), sorted(dichas - set(esperadas))
        ok = len(set(esperadas) & dichas)
        aciertos, total = aciertos + ok, total + len(esperadas)
        evaluacion[aspecto] = {"mencionadas": sorted(dichas), "faltan": faltan, "sobran": sobran}
        print(f"  - {aspecto}: {ok}/{len(esperadas)} correctas | faltan: {faltan or 'ninguna'} | "
              f"sobran (no marcadas): {sobran or 'ninguna'}")
    print(f"Resultado: {'CORRECTA' if all(not v['faltan'] and not v['sobran'] for v in evaluacion.values()) else 'INCORRECTA'}"
          f" ({aciertos}/{total} métricas esperadas encontradas)")
    guardar_json("1_1_pregunta.json", {
        "modelo": VISION_MODEL, "imagen": PREGUNTA_IMAGEN.name, "pregunta": PREGUNTA, "respuesta": resp,
        "esperada": RESPUESTA_ESPERADA, "evaluacion": evaluacion, "segundos": round(seg, 2), "uso": uso,
    })


# ---------------------------------------------------------------------------
# 1.2  Extracción estructurada
# ---------------------------------------------------------------------------
class FichaDiapositiva(BaseModel):
    """Ficha de una diapositiva del curso, útil para catalogar el material que consulta el tutor."""

    model_config = ConfigDict(extra="forbid")

    titulo: str = Field(description="Título visible de la diapositiva, copiado tal cual")
    metodo_principal: Literal["TF", "TF-IDF", "BM25", "RAG", "Keyword Search", "Semantic Search", "Otro"] = Field(
        description="Método de recuperación o concepto principal que explica la diapositiva"
    )
    tipo_contenido: Literal["formula", "ejemplo_numerico", "diagrama", "tabla", "texto"] = Field(
        description="Tipo de contenido dominante de la diapositiva"
    )
    idioma_predominante: Literal["es", "en"] = Field(description="Idioma en que está escrita la mayor parte del texto")
    contiene_formula: bool = Field(description="true si aparece una fórmula matemática escrita")
    ejemplo_termino: str | None = Field(
        description="Palabra o consulta concreta usada como ejemplo principal (sin comillas); null si no hay"
    )
    num_documentos_ejemplo: int = Field(
        ge=0, description="Cuántos documentos de ejemplo se muestran o mencionan (Documento A, Document 1...); 0 si ninguno"
    )


IMAGENES_EXTRACCION = ["01_bm25_formula.png", "02_tfidf_formula.png", "03_term_frequency_ejemplo.png"]
VALORES_REALES = IMAGENES / "valores_reales.json"  # lo que leímos nosotros en cada imagen (verdad de referencia)

PROMPT_EXTRACCION = (
    "Analiza la diapositiva de un curso sobre recuperación de información y RAG. "
    "Devuelve SOLO un objeto JSON (sin texto adicional ni bloques de código) que cumpla este JSON Schema:\n"
    + json.dumps(FichaDiapositiva.model_json_schema(), ensure_ascii=False)
    + "\nUsa exactamente los nombres de campo y los valores permitidos del esquema."
)


def extraer_json(texto: str):
    """Convierte la salida del modelo en dict. Tolera bloques ```json``` pero no inventa nada."""
    texto = texto.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", texto, re.S)
    if m:
        texto = m.group(1).strip()
    return json.loads(texto)


def validar(datos) -> tuple[FichaDiapositiva | None, list[str]]:
    """Valida contra el esquema. Nunca lanza excepción: devuelve la lista de problemas encontrados."""
    try:
        return FichaDiapositiva.model_validate(datos), []
    except ValidationError as e:
        return None, [f"{'.'.join(map(str, err['loc'])) or '(raíz)'}: {err['msg']}" for err in e.errors()]


def normalizar(v) -> str:
    if v is None:
        return "null"
    s = unicodedata.normalize("NFKD", str(v)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[¿?¡!\"'“”()\[\].,:;]", " ", s)
    return " ".join(s.split())


def coincide(real, extraido) -> bool:
    if isinstance(real, bool) or isinstance(real, int):
        return extraido == real and type(extraido) is type(real)
    return normalizar(real) == normalizar(extraido)


def demo_validacion() -> None:
    """Muestra que un JSON que no cumple el esquema se informa en lugar de romper el programa."""
    titulo("1.2 (demostración) VALIDACIÓN DE UN JSON INCORRECTO")
    roto = {"titulo": "BM25", "metodo_principal": "Okapi", "tipo_contenido": "formula",
            "idioma_predominante": "español", "contiene_formula": "sí", "num_documentos_ejemplo": -1, "extra": 1}
    print("JSON de prueba:", json.dumps(roto, ensure_ascii=False))
    _, errores = validar(roto)
    print("Resultado: NO cumple el esquema →")
    for e in errores:
        print("   -", e)
    try:
        extraer_json("La diapositiva trata de BM25 {titulo: sin comillas}")
    except json.JSONDecodeError as e:
        print(f"Texto que no es JSON → se informa: JSONDecodeError: {e}")


def parte_extraccion(c) -> None:
    titulo(f"1.2 EXTRACCIÓN ESTRUCTURADA   (modelo: {VISION_MODEL})")
    print("Esquema (Pydantic → JSON Schema):")
    for nombre, campo in FichaDiapositiva.model_fields.items():
        print(f"  - {nombre:22s} {str(campo.annotation).replace('typing.', ''):70.70s}  {campo.description}")
    reales = json.loads(VALORES_REALES.read_text(encoding="utf-8"))
    filas, resultados = [], []
    for nombre in IMAGENES_EXTRACCION:
        print(f"\n--- {nombre} ---")
        try:
            r = c.chat.completions.create(
                model=VISION_MODEL,
                messages=[mensaje_usuario(PROMPT_EXTRACCION, [IMAGENES / nombre])],
                temperature=0,
                max_completion_tokens=1000,
                response_format={"type": "json_object"},
            )
            crudo = r.choices[0].message.content or ""
        except Exception as e:  # el programa informa y continúa con la siguiente imagen
            info = describir_error_api(e)
            print(f"La API falló: {info['status_code']} {info['mensaje']}")
            crudo = ""
        print("Salida cruda del modelo:", crudo)
        datos, problemas = None, []
        try:
            datos = extraer_json(crudo)
        except json.JSONDecodeError as e:
            problemas = [f"no es JSON válido: {e}"]
        if datos is not None:
            ficha, problemas = validar(datos)
        print("Validación:", "OK, cumple el esquema" if not problemas else "NO cumple el esquema")
        for p in problemas:
            print("   -", p)
        extraido = datos if isinstance(datos, dict) else {}
        for campo, valor_real in reales[nombre].items():
            valor = extraido.get(campo, "(falta)")
            ok = coincide(valor_real, valor)
            filas.append({"imagen": nombre, "campo": campo, "valor_real": valor_real, "valor_extraido": valor, "correcto": ok})
        resultados.append({"imagen": nombre, "salida_cruda": crudo, "valido": not problemas, "problemas": problemas})

    titulo("Tabla de aciertos (valor real vs. extraído)")
    print(f"{'Imagen':32s} {'Campo':22s} {'Real':28s} {'Extraído':28s} OK")
    for f in filas:
        print(f"{f['imagen']:32s} {f['campo']:22s} {str(f['valor_real'])[:27]:28s} {str(f['valor_extraido'])[:27]:28s} "
              f"{'Sí' if f['correcto'] else 'No'}")
    aciertos = sum(f["correcto"] for f in filas)
    print(f"\nPorcentaje de aciertos: {aciertos} de {len(filas)} campos = {100 * aciertos / len(filas):.1f} %")
    guardar_json("1_2_extraccion.json", {
        "modelo": VISION_MODEL, "esquema": FichaDiapositiva.model_json_schema(), "resultados": resultados,
        "tabla": filas, "aciertos": aciertos, "total": len(filas),
    })
    demo_validacion()


# ---------------------------------------------------------------------------
# 1.3  Varias imágenes en una solicitud
# ---------------------------------------------------------------------------
IMAGENES_MULTIPLES = ["02_tfidf_formula.png", "01_bm25_formula.png", "07_bm25_vs_tfidf_tabla.png"]
PREGUNTA_MULTIPLE = (
    "Las tres imágenes son diapositivas del mismo curso (imagen 1: TF-IDF, imagen 2: fórmula de BM25, "
    "imagen 3: resultados y tabla comparativa). Usando solo lo que aparece en ellas, compara TF-IDF y BM25: "
    "1) escribe la fórmula de cada uno; 2) ¿qué problemas de TF-IDF corrige BM25 y con qué parámetro lo hace cada uno?; "
    "3) en el ejemplo de resultados, ¿qué documento queda primero con cada método y por qué? Responde en español."
)


def parte_multiples(c) -> None:
    titulo(f"1.3 VARIAS IMÁGENES EN UNA SOLICITUD   (modelo: {VISION_MODEL})")
    rutas = [IMAGENES / n for n in IMAGENES_MULTIPLES]
    print("Imágenes:", ", ".join(IMAGENES_MULTIPLES))
    print(f"Pregunta: {PREGUNTA_MULTIPLE}\n")
    resp, seg, uso = preguntar(c, PREGUNTA_MULTIPLE, rutas)
    print(f"Respuesta del modelo ({seg:.1f} s, tokens de entrada: {uso.get('prompt_tokens')}):\n{resp}")
    guardar_json("1_3_multiples.json", {
        "modelo": VISION_MODEL, "imagenes": IMAGENES_MULTIPLES, "pregunta": PREGUNTA_MULTIPLE,
        "respuesta": resp, "segundos": round(seg, 2), "uso": uso,
    })


# ---------------------------------------------------------------------------
# 1.4  Errores provocados
# ---------------------------------------------------------------------------
def _imagen_gigante_png(lado: int = 2900) -> bytes:
    """PNG de ruido aleatorio (no se comprime): ~25 MB, ~34 MB en base64. Supera el límite de 20 MB por solicitud."""
    import numpy as np
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 256, (lado, lado, 3), dtype=np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def _casos_error():
    img = IMAGENES / "01_bm25_formula.png"
    texto = "Describe brevemente esta diapositiva."
    cuatro = ["01_bm25_formula.png", "02_tfidf_formula.png", "04_tecnicas_busqueda.png", "05_survey_fig2_proceso_rag.png"]
    return {
        "mas_imagenes": (
            "Se envían 4 imágenes en un solo mensaje (el límite documentado es 3).",
            lambda: dict(model=VISION_MODEL, messages=[mensaje_usuario(texto, [IMAGENES / n for n in cuatro])]),
        ),
        "modelo_sin_vision": (
            f"Se envía una imagen a {TEXT_MODEL}, un modelo de solo texto.",
            lambda: dict(model=TEXT_MODEL, messages=[mensaje_usuario(texto, [img])]),
        ),
        "tipo_mal_declarado": (
            "Se envía un PNG declarando el tipo 'text/plain' en la data URL.",
            lambda: dict(model=VISION_MODEL, messages=[{"role": "user", "content": [
                {"type": "text", "text": texto}, parte_imagen(img, mime="text/plain")]}]),
        ),
        "imagen_muy_grande": (
            "Se envía un PNG de ruido de 2900x2900 (~25 MB; ~34 MB en base64), por encima de 20 MB por solicitud.",
            lambda: dict(model=VISION_MODEL, messages=[{"role": "user", "content": [
                {"type": "text", "text": texto},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,"
                                                    + __import__("base64").b64encode(_imagen_gigante_png()).decode()}}]}]),
        ),
    }


def parte_error(c, caso: str | None = None) -> None:
    titulo("1.4 ERRORES PROVOCADOS")
    casos = _casos_error()
    salida = []
    for nombre, (descripcion, construir) in casos.items():
        if caso and nombre != caso:
            continue
        print(f"\n--- Caso: {nombre} ---\nQué se hizo: {descripcion}")
        try:
            r = c.chat.completions.create(max_completion_tokens=300, **construir())
            texto = r.choices[0].message.content or ""
            print("La API NO devolvió error. Respuesta:", texto[:400])
            salida.append({"caso": nombre, "descripcion": descripcion, "error": None, "respuesta": texto})
        except Exception as e:
            info = describir_error_api(e)
            print(f"La API devolvió un error → {info['tipo']} (HTTP {info['status_code']})")
            print(f"Mensaje: {info['mensaje']}")
            if info.get("codigo"):
                print(f"Código: {info['codigo']}")
            salida.append({"caso": nombre, "descripcion": descripcion, "error": info})
    guardar_json("1_4_errores.json", salida)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parte", choices=["pregunta", "extraccion", "multiples", "error", "todo"])
    ap.add_argument("--caso", choices=list(_casos_error()), help="solo para 'error': ejecuta un único caso")
    a = ap.parse_args()
    c = cliente()
    pasos = {
        "pregunta": lambda: parte_pregunta(c),
        "extraccion": lambda: parte_extraccion(c),
        "multiples": lambda: parte_multiples(c),
        "error": lambda: parte_error(c, a.caso),
    }
    for nombre in (pasos if a.parte == "todo" else [a.parte]):
        with registrar(f"parte1_{nombre}"):
            pasos[nombre]()


if __name__ == "__main__":
    main()
