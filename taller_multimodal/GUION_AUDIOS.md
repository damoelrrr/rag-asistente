# Guion de los audios (Parte 2)

Son **4 audios obligatorios** y **1 opcional**. Hablan Stiven y Daniel, y el **audio 3 lleva ruido de fondo**.
Todas las preguntas tienen respuesta en el corpus del tutor, salvo la del audio 5, que está fuera a propósito para ver si el asistente se abstiene.

## Cómo grabar

1. **Con qué grabar:** con el celular, en *Notas de voz* (iPhone, guarda `.m4a`) o en la grabadora de Android (`.m4a` o `.mp3`). Un audio de WhatsApp (`.ogg` / `.opus`) también sirve.
2. **Duración:** cada audio debe durar **entre 10 y 30 segundos**. Leyendo a ritmo normal, cada texto toma unos 15 a 20 s.
3. **Silencio:** dejen **1 segundo de silencio** al empezar y al terminar.
4. **Cómo leer:** lean **exactamente** el texto, con naturalidad, sin deletrear ni leer los signos de puntuación.
   - Si se equivocan o cambian alguna palabra, **no hay problema**, pero anoten lo que dijeron de verdad en [audios/guion.json](audios/guion.json), en el campo `"texto"` de ese audio. El script compara la transcripción con ese texto.
5. **Nombre del archivo:** usen exactamente estos nombres (la extensión puede variar) y guárdenlos en [`taller_multimodal/audios/`](audios/):
   - `audio1_stiven.m4a`
   - `audio2_daniel.m4a`
   - `audio3_stiven_ruido.m4a`
   - `audio4_daniel.m4a`
   - `audio5_daniel.m4a` (opcional)

### Cómo se pronuncian los términos técnicos

Díganlos como los diría un estudiante en clase; eso es justo lo que se quiere evaluar.

| Se escribe | Se dice |
|---|---|
| BM25 | "be eme veinticinco" |
| TF-IDF | "te efe i de efe" |
| k1 | "ka uno" |
| RAG | "rag" |
| indexing, chunks, embeddings | en inglés: "índexin", "chonks", "embédings" |
| LLM | "ele ele eme" |
| Ollama | "o-yama" |
| API | "a pe i" |
| Llama 3 | "yama tres" |
| pgvector | "pe ge véctor" |
| PostgreSQL | "postgres" o "póstgres cu ele" |

---

## Audio 1: Stiven, sin ruido

**Archivo:** `audio1_stiven.m4a`. Busquen un lugar silencioso.

> Hola, soy Stiven. Tengo una duda sobre búsqueda por palabras clave. ¿En qué se diferencia BM25 de TF-IDF? Y en la fórmula de BM25, ¿qué controlan los parámetros k1 y b, y qué valores se usan normalmente?

*Respuesta esperada:* BM25 satura la frecuencia del término y normaliza por la longitud del documento. k1 controla la saturación (normalmente entre 1.2 y 2.0) y b la penalización por longitud (normalmente 0.75).

---

## Audio 2: Daniel, sin ruido

**Archivo:** `audio2_daniel.m4a`. Busquen un lugar silencioso.

> Hola, soy Daniel. En un sistema RAG, ¿qué pasa en la etapa de indexing? Quiero entender cómo se dividen los documentos en chunks y para qué sirven los embeddings y la base de datos vectorial cuando el usuario hace una pregunta.

*Respuesta esperada:* los documentos se limpian y se dividen en chunks. Cada chunk se convierte en un vector con un modelo de embeddings y se guarda en una base de datos vectorial. Cuando llega una pregunta, se convierte con el mismo modelo y se recuperan los top-k chunks más similares.

---

## Audio 3: Stiven, CON RUIDO DE FONDO

**Archivo:** `audio3_stiven_ruido.m4a`.

**El ruido:** pongan música con letra o un video o la TV a **volumen medio**, a 1 o 2 metros del celular. También sirve grabar en la cafetería o en la calle. El ruido debe notarse, pero la voz se tiene que seguir entendiendo.

> Soy Stiven otra vez. ¿Qué ventajas tiene ejecutar un LLM de forma local con Ollama en vez de usar una API en la nube? ¿Y qué comando se usa para descargar el modelo Llama 3?

*Respuesta esperada:* las ventajas son privacidad, costo cero, no depender de internet y control total. El comando es `ollama pull llama3`.

---

## Audio 4: Daniel, sin ruido

**Archivo:** `audio4_daniel.m4a`. Busquen un lugar silencioso.

> Soy Daniel. Según la ingeniería de prompts, ¿cuáles son los componentes de un prompt correcto? Por ejemplo, ¿qué diferencia hay entre la tarea, el formato y el contexto, y para qué sirven las restricciones?

*Respuesta esperada:* los componentes son tarea, formato, tema, tono o estilo, contexto y restricciones. La tarea es la acción, el formato es el tipo de resultado y el contexto es la información de fondo. Las restricciones limitan lo que debe o no debe hacer la respuesta.

---

## Audio 5 (opcional): Daniel, pregunta fuera del corpus

**Archivo:** `audio5_daniel.m4a`. Dura unos 10 s; si queda más corto, lean un poco más despacio.

> Última pregunta. ¿Cómo instalo la extensión pgvector en PostgreSQL para guardar los embeddings de mi proyecto?

*Respuesta esperada:* el tutor debe decir que **no tiene esa información** en sus fuentes, porque pgvector no está en el corpus.

---

## Después de grabar

```bash
cd rag-asistente
source .venv/bin/activate
python taller_multimodal/parte2_audio.py --comparar-modelos --vocabulario
```

Los resultados quedan en `taller_multimodal/resultados/`. Para el informe, anoten también la **duración real** de cada audio, aunque el script la saca de la API.
