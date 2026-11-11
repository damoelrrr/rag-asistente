# Taller: entrada multimodal y audio

**Integrantes:** Stiven Rodriguez y Daniel Rodriguez.

El dominio es el mismo del asistente RAG: el material del curso *Desarrollo de aplicaciones con IA*.

- **Imágenes:** diapositivas, fórmulas, una figura y una tabla sacadas de los PDFs de `data/corpus/`.
- **Audios:** preguntas habladas sobre esos temas, que responde el tutor RAG del proyecto.

## Contenido

| Archivo | Qué hace |
|---|---|
| `preparar_imagenes.py` | Genera las 7 imágenes de `imagenes/` a partir de los PDFs del corpus (reproducible) |
| `parte1_vision.py` | Parte 1: pregunta sobre una imagen (1.1), extracción estructurada (1.2), varias imágenes (1.3) y errores provocados (1.4) |
| `parte2_audio.py` | Parte 2: transcripción con Whisper (2.1) y respuesta del tutor RAG a partir de la transcripción (2.2) |
| `generar_informe.py` | Llena la plantilla de entrega (`informe/Informe_Taller_Multimodal.docx`) con los resultados y las capturas |
| `common.py` | Cliente de Groq, imágenes en base64 y registro de la salida |
| `imagenes/valores_reales.json` | Valores que leímos nosotros en cada imagen (verdad de referencia para 1.2) |
| `audios/guion.json` | Hablante, condición y texto real de cada audio (para medir el error de transcripción) |
| `GUION_AUDIOS.md` | Qué decir en cada audio y cómo grabarlo |
| `resultados/` | Salida de cada ejecución (`.txt`, que sirven como capturas, y `.json`) |

## Variables de entorno

Se leen del archivo `.env` en la **raíz del repositorio** (copiado de `.env.example`). Ninguna clave va en el código.

| Variable | Obligatoria | Valor por defecto | Uso |
|---|---|---|---|
| `GROQ_API_KEY` | sí | — | Visión, Whisper y el LLM del tutor |
| `LLM_PROVIDER` / `LLM_MODEL` | para 2.2 | `groq` / `openai/gpt-oss-120b` | LLM con el que responde el tutor RAG |
| `VISION_MODEL` | no | `qwen/qwen3.8-27b` | Modelo con visión |
| `AUDIO_MODEL` | no | `whisper-large-v3-turbo` | Modelo de transcripción |
| `TEXT_MODEL` | no | `openai/gpt-oss-120b` | Modelo sin visión, usado en el error provocado |

## Cómo ejecutar

Todos los comandos se corren desde la raíz del repositorio, con el entorno del proyecto activado:

```bash
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt pypdfium2

# Imágenes (ya están en el repo; solo hace falta si se quieren regenerar)
python taller_multimodal/preparar_imagenes.py

# Parte 1: todo seguido, o cada punto por separado
python taller_multimodal/parte1_vision.py todo
python taller_multimodal/parte1_vision.py pregunta      # 1.1
python taller_multimodal/parte1_vision.py extraccion    # 1.2
python taller_multimodal/parte1_vision.py multiples     # 1.3
python taller_multimodal/parte1_vision.py error         # 1.4 (o --caso modelo_sin_vision)

# Parte 2 (primero hay que poner los audios en audios/, ver GUION_AUDIOS.md)
python taller_multimodal/parte2_audio.py --comparar-modelos --vocabulario   # lo que usamos en el informe
python taller_multimodal/parte2_audio.py --solo-transcripcion   # solo 2.1

# Informe en Word
python taller_multimodal/generar_informe.py
```

## Notas

- **Límites de Groq:**
  - Visión: máximo 3 imágenes y 20 MB por solicitud, y cada imagen cuenta como 2.048 tokens de entrada.
  - Audio: hasta 25 MB por archivo, y se cobran mínimo 10 s por solicitud.
  - Las imágenes de `imagenes/` pesan menos de 300 KB cada una.
- **Validación del JSON (1.2):** el esquema es un modelo de Pydantic (`FichaDiapositiva`). Si el modelo devuelve algo que no es JSON o que no cumple el esquema, el programa lista los problemas y sigue con la siguiente imagen en lugar de detenerse. Esos campos cuentan como errores en el porcentaje de aciertos.
- **Glosario (`--vocabulario`):** también transcribe con whisper-large-v3-turbo pasando un glosario de términos técnicos en el parámetro `prompt`. En nuestras pruebas bajó el WER promedio de 9,6 % a 1,7 %.
- **Respuesta a partir del audio (2.2):** la transcripción pasa por el mismo pipeline del asistente (`rag/pipeline.py`): búsqueda híbrida en el índice, generación con citas y abstención si la pregunta está fuera del corpus. Como control, también se responde con el texto real y se comparan los fragmentos recuperados (archivo y página).
