# MIDI-Sketchbook

**English** · [Español](#español)

MIDI-Sketchbook is an experimental, educational repository for exploring
symbolic music generation from a personal collection of MIDI files. Its goal
is a small “musical sketch machine”: it will learn statistical patterns from
the dataset supplied by the user and eventually produce editable ideas of 2–8
bars for tools such as REAPER or Guitar Pro. It is not intended to produce
finished songs, audio, tablature, or a legal guarantee of “original” music.

> **Data and copyright:** use only MIDI files that you created, that have a
> compatible license, or that you are explicitly authorized to use. Do not add
> commercial MIDI files or protected third-party material to this repository.
> Dataset MIDI files are ignored by Git and none are included here.

## Current status: Stage 1

Stage 1 is implemented: repository setup, MIDI inspection and basic
preprocessing. The executable workflow can:

- inspect `.mid` and `.midi` files without stopping the whole batch when one
  file is corrupt or incompatible;
- validate pitch range, 4/4 meter, constant tempo, pitch bends, lyrics,
  control changes, drum tracks, ambiguous note events, and minimum note counts;
- select either the non-drum `pretty_midi` instrument stream with the most
  notes or an explicit instrument index;
- remove initial silence and optionally quantize note onsets and durations;
- strip Guitar Pro's inert time-zero pitch-bend-range setup and collapse exact
  duplicate unison notes while retaining strict checks for musical controls;
- split material into configurable phrases of 2, 4, or 8 bars;
- assign each source file to exactly one train, validation, or test split
  before creating phrases, grouping byte-identical sources to prevent
  duplicate-content leakage;
- optionally transpose training material from -5 through +6 semitones while
  rejecting results outside the configured pitch range; and
- write each reproducible immutable run under
  `data/processed/runs/<run_id>/{train,validation,test}/` and an authoritative
  JSON manifest at `data/splits/manifest.json`.

REMI tokenization, the dataset/data loader, GRU training, checkpoints,
TensorBoard logging, autoregressive sampling, seed-MIDI generation, and token
decoding are **not implemented yet**. `configs/train.yaml` and
`configs/generate.yaml` are clearly marked design placeholders for those later
stages; changing them does not train or generate anything.

## Architecture and main decisions

The repository uses a small `src`-layout Python package named
`midi_idea_generator`. Command-line scripts stay thin; configuration,
validation, MIDI I/O, and transformations live in separate, testable modules.
YAML files hold experiment settings, errors are reported per file, random
seeds make splits reproducible, and generated artifacts remain outside version
control.

The manifest is the dataset boundary: it records the normalized configuration,
implementation and dependency versions, source SHA-256 hashes, split,
instrument index, source duration/resolution, and every current fragment. A
later tokenization stage must read its `fragments` entries and must not glob
all of `data/processed`, because that directory can contain older immutable
runs. Identical inputs and configuration reuse the same content-derived run
ID; changed inputs or settings create an isolated run without deleting prior
data.

The current preprocessing boundary deliberately assumes one principal
instrumental track, 4/4 meter, constant tempo per fragment, no pitch bends,
and monophonic or modestly polyphonic material. The source file—not a derived
phrase—is the unit of dataset splitting. Transposition never changes that
source split.

`track_number` in the manifest is retained to match the project requirement,
but in Stage 1 it means the zero-based index in `pretty_midi.instruments`. It
is also emitted explicitly as `instrument_index`; it is not the raw Standard
MIDI File track number, because one raw track can contain several channels or
program streams.

The planned model is a compact two-layer autoregressive GRU rather than a
Transformer. Its roadmap defaults (128-dimensional embedding, 256 hidden
units, batch size 4) are intended as a conservative starting point for a GPU
with 4 GB of VRAM, but no memory or training claim can be tested until the
training stage exists.

```text
configs/                    executable preprocessing config + future templates
data/raw/                   user-supplied MIDI (ignored by Git)
data/processed/runs/        immutable normalized runs, each grouped by split
data/tokenized/             reserved for a later stage
data/splits/                generated manifest
outputs/                    generated results, plots, and logs (ignored)
checkpoints/                future model checkpoints (ignored)
scripts/                    Stage 1 command-line entry points
src/midi_idea_generator/    reusable package code
tests/                      automated tests
```

## Technology and compatibility

The target is Python 3.12. Dependencies are declared in `pyproject.toml`:
PyTorch, MidiTok 3.x, pretty_midi 0.2.11, Mido, NumPy, PyYAML, Matplotlib,
TensorBoard, and pytest.

Python, PyTorch, and CUDA compatibility is wheel- and platform-specific. The
declared conservative baseline targets PyTorch 2.12.1, MidiTok 3.0.6.post1,
pretty_midi 0.2.11, and NumPy 1.26.4. Older PyTorch or MidiTok releases may not
provide Python 3.12 support, while MidiTok major versions have incompatible
APIs. In particular, MidiTok 3 consumes MIDI paths or `symusic.Score` objects
and does not consume `pretty_midi.PrettyMIDI` objects. A later tokenization
stage will keep a file boundary between pretty_midi preprocessing and MidiTok,
and will not use legacy MidiTok 2.x examples.

This checkout was functionally exercised with the available Python 3.14.6
interpreter, pretty_midi 0.2.11, Mido 1.3.3, PyYAML 6.0.3, and NumPy 2.5.1;
the full test suite passes and all source files parse with Python 3.12 grammar.
The host did not provide a Python 3.12 runtime, so the exact pinned 3.12/NumPy
1.26 environment still needs a smoke run on the target machine. Python 3.12
support and wheel availability were checked against official package metadata.

On Linux, the default PyPI PyTorch wheel may target a CUDA release unsupported
by an older 4 GB GPU. Before installing the project, choose a CPU or CUDA wheel
that matches the local driver and GPU using the [official PyTorch
selector](https://pytorch.org/get-started/locally/). Stage 1 itself does not
require a GPU.

## Installation

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install exactly one PyTorch wheel before the project. CUDA 12.6 is the
conservative GPU baseline; use the CPU wheel when CUDA is unavailable:

```bash
# NVIDIA/CUDA 12.6 option
python -m pip install torch==2.12.1 \
  --index-url https://download.pytorch.org/whl/cu126

# CPU-only option (run this instead of the command above)
python -m pip install torch==2.12.1 \
  --index-url https://download.pytorch.org/whl/cpu
```

Then install the package and tests:

```bash
python -m pip install -e ".[dev]"
```

Use `python -m pip install -r requirements.txt` instead of the last command to
install the pinned top-level baseline. On a GPU system, install the appropriate
PyTorch 2.12.1 wheel first; the remaining install will reuse it. The project
intentionally has no Docker, web UI, desktop application, database, cloud
service, or audio-model setup.

## Prepare data

1. Copy only authorized `.mid` or `.midi` files into `data/raw/`. Nested
   folders may be used if desired.
2. Review `configs/preprocess.yaml`. Relative paths are resolved from the
   repository root (the parent directory of `configs/`).
3. Keep the source files private: Git ignores MIDI and all derived data.

The default pitch range is MIDI 21–108. Missing time-signature events may be
treated as 4/4, but explicit non-4/4 events are rejected. Quantization is off
by default; when enabled, `subdivisions_per_beat: 4` uses a sixteenth-note
grid. The configured split fractions must total 1.0.

Each successful preprocessing manifest points to one immutable run. Old runs
are intentionally left untouched for safety and can be removed manually only
after confirming that their run ID is not referenced by the current manifest.

## Tested Stage 1 commands

Inspect the raw collection:

```bash
python scripts/inspect_midis.py --input data/raw
```

Optional inspection flags are `--pitch-min`, `--pitch-max`, and `--output`.
For example, a machine-readable report can be requested with:

```bash
python scripts/inspect_midis.py --input data/raw --output outputs/logs/inspection.json
```

Preprocess and build the leakage-safe splits and manifest:

```bash
python scripts/preprocess.py --config configs/preprocess.yaml
```

Run the test suite:

```bash
python -m pytest
```

The current suite contains 73 synthetic tests and writes all MIDI fixtures to
temporary directories; no dataset MIDI is committed.

There is intentionally no working `train.py` or `generate.py` command in
Stage 1. Their planned configuration files are documentation, not executable
features.

## Training and generation (future)

Training will use REMI token sequences, next-token cross-entropy with padding
ignored, gradient clipping, optional stable mixed precision, checkpoints and
TensorBoard loss metrics. Generation will support an empty start or seed MIDI,
a reproducible random seed, maximum token count, temperature, top-k, top-p,
sample count, and a simple repetition penalty. Empty or invalid decoded MIDI
must not be saved. None of this behavior is available in Stage 1.

## Limitations

- Only one selected instrumental track is retained per source example.
- The supported meter is 4/4 and tempo must remain constant within a source.
- Non-neutral pitch bends, lyrics, audio, and guitar fingering are outside the
  current scope; pitchwheel reset events at value zero are harmlessly removed.
- Rhythm augmentation is not implemented.
- Simple polyphony is accepted, but complex multi-track arrangement semantics
  are not preserved.
- Exact duplicate unisons are collapsed. Non-identical overlapping note-on
  events for the same pitch/channel and dangling notes remain rejected because
  their Standard MIDI File interpretation is ambiguous.
- Control changes are rejected rather than silently discarded, except for the
  exact time-zero pitch-bend-range RPN emitted by Guitar Pro; sustain-pedal
  rendering is not implemented yet.
- Final phrase completeness is measured with the source End-of-Track duration,
  and generated phrase files preserve trailing rests up to their nominal
  2/4/8-bar boundary.
- Preprocessing cannot make an unsuitable or mislabeled dataset legally safe.
- No model is trained and no MIDI is generated in Stage 1.

## Roadmap

1. **Done:** structure, configuration, inspection, preprocessing, manifest,
   leakage-safe source splits, transposition, and critical Stage 1 tests.
2. Add REMI tokenization with MidiTok and token round-trip tests.
3. Add a sequence dataset and data loader with correct padding.
4. Add the small GRU, CPU/CUDA training, gradient clipping, checkpoints,
   resuming, mixed precision where stable, and TensorBoard metrics.
5. Add temperature, top-k, and top-p sampling; optional seed MIDI; validation;
   and export to `outputs/generated/`.

The envisioned future interfaces are `python scripts/train.py --config
configs/train.yaml` and `python scripts/generate.py --config
configs/generate.yaml`, optionally with `--seed-midi`. **These commands are not
implemented or tested yet.**

## License

The source code is available under the MIT License. That license covers the
repository's code, not any MIDI dataset supplied by a user and not the legal
status of generated results. You are responsible for the rights and permitted
uses of your data.

---

## Español

MIDI-Sketchbook es un repositorio experimental y educativo para explorar la
generación musical simbólica a partir de una colección personal de archivos
MIDI. Su meta es una pequeña “máquina de bocetos musicales”: aprenderá patrones
estadísticos del dataset proporcionado por el usuario y, en etapas posteriores,
producirá ideas editables de 2 a 8 compases para herramientas como REAPER o
Guitar Pro. No pretende producir canciones terminadas, audio, tablaturas ni
garantizar “originalidad” musical en sentido legal.

> **Datos y copyright:** utiliza solamente archivos MIDI propios, con una
> licencia compatible o cuyo uso tengas expresamente autorizado. No agregues
> MIDIs comerciales ni material protegido de terceros al repositorio. Git
> ignora los archivos MIDI del dataset y este repositorio no incluye ninguno.

## Estado actual: Etapa 1

La Etapa 1 está implementada: estructura del repositorio, inspección MIDI y
preprocesamiento básico. El flujo ejecutable permite:

- inspeccionar archivos `.mid` y `.midi` sin detener todo el lote cuando uno
  está corrupto o es incompatible;
- validar rango tonal, compás 4/4, tempo constante, pitch bends, letras,
  control changes, pistas de batería, eventos ambiguos y cantidad mínima de
  notas;
- seleccionar el flujo instrumental no percusivo de `pretty_midi` con más
  notas o un índice de instrumento explícito;
- eliminar el silencio inicial y cuantizar opcionalmente inicios y duraciones;
- quitar la configuración inerte de rango de pitch bend que Guitar Pro escribe
  al inicio y colapsar unísonos duplicados exactos, manteniendo controles
  estrictos para eventos musicales;
- dividir el material en frases configurables de 2, 4 u 8 compases;
- asignar cada archivo fuente a un único split de entrenamiento, validación o
  prueba antes de fragmentarlo y agrupar fuentes idénticas byte a byte para
  evitar leakage por contenido duplicado;
- transponer opcionalmente el material de entrenamiento entre -5 y +6
  semitonos, descartando resultados fuera del rango configurado; y
- guardar cada corrida inmutable y reproducible en
  `data/processed/runs/<run_id>/{train,validation,test}/` y el manifiesto JSON
  autoritativo en `data/splits/manifest.json`.

La tokenización REMI, el dataset/data loader, el entrenamiento GRU, los
checkpoints, TensorBoard, el muestreo autorregresivo, la generación condicionada
por un MIDI semilla y la decodificación a MIDI **todavía no están
implementados**. `configs/train.yaml` y `configs/generate.yaml` son plantillas
de diseño claramente marcadas para etapas futuras; modificarlas no entrena ni
genera nada.

## Arquitectura y decisiones principales

El repositorio usa un paquete Python con layout `src` llamado
`midi_idea_generator`. Los scripts de línea de comandos son delgados; la
configuración, validación, entrada/salida MIDI y transformaciones viven en
módulos separados y comprobables. YAML almacena los parámetros, los errores se
reportan por archivo, las semillas hacen reproducibles los splits y los
artefactos generados quedan fuera del control de versiones.

El manifiesto define el dataset: registra configuración normalizada, versiones
de implementación y dependencias, SHA-256 de cada fuente, split, índice de
instrumento, duración/resolución y cada fragmento vigente. La tokenización
posterior deberá leer `fragments` y nunca recorrer todo `data/processed`, que
puede contener corridas inmutables antiguas. Entradas y configuración idénticas
reutilizan el mismo run ID derivado del contenido; cualquier cambio crea una
corrida aislada sin borrar datos previos.

El límite actual asume una pista instrumental principal, compás 4/4, tempo
constante por fragmento, ausencia de pitch bends y material monofónico o con
polifonía sencilla. La unidad para dividir el dataset es el archivo fuente, no
la frase derivada. La transposición nunca cambia el split de origen.

El campo `track_number` se conserva en el manifiesto para cumplir el contrato
del proyecto, pero en esta etapa significa el índice base cero dentro de
`pretty_midi.instruments`. También se publica como `instrument_index`; no es el
número de track crudo del Standard MIDI File, porque un track puede contener
varios canales o programas.

El modelo previsto es una GRU autorregresiva pequeña de dos capas, no un
Transformer. Los valores de roadmap (embedding 128, estado oculto 256 y batch
4) son un punto de partida conservador para una GPU con 4 GB de VRAM, pero el
consumo y entrenamiento no podrán comprobarse hasta implementar esa etapa.

## Estructura del proyecto

```text
configs/                    config ejecutable y plantillas futuras
data/raw/                   MIDI del usuario (ignorado por Git)
data/processed/runs/        corridas inmutables, cada una separada por split
data/tokenized/             reservado para una etapa posterior
data/splits/                manifiesto generado
outputs/                    resultados, gráficos y logs (ignorados)
checkpoints/                checkpoints futuros (ignorados)
scripts/                    comandos de la Etapa 1
src/midi_idea_generator/    código reutilizable del paquete
tests/                      tests automáticos
```

## Tecnologías y compatibilidad

El objetivo es Python 3.12. `pyproject.toml` declara PyTorch, MidiTok 3.x,
pretty_midi 0.2.11, Mido, NumPy, PyYAML, Matplotlib, TensorBoard y pytest.

La compatibilidad entre Python, PyTorch y CUDA depende de los wheels y de la
plataforma. El baseline conservador declarado usa PyTorch 2.12.1,
MidiTok 3.0.6.post1, pretty_midi 0.2.11 y NumPy 1.26.4. Versiones antiguas
pueden no soportar Python
3.12 y las versiones mayores de MidiTok tienen APIs incompatibles. En
particular, MidiTok 3 recibe rutas MIDI u objetos `symusic.Score`, no objetos
`pretty_midi.PrettyMIDI`. La etapa posterior de tokenización mantendrá un límite
por archivo entre el preprocesamiento con pretty_midi y MidiTok, sin copiar
ejemplos antiguos de la API 2.x.

Este checkout se ejercitó funcionalmente con el intérprete disponible Python
3.14.6, pretty_midi 0.2.11, Mido 1.3.3, PyYAML 6.0.3 y NumPy 2.5.1; la suite
completa pasa y todo el código se analiza con la gramática de Python 3.12. El
host no dispone de un runtime 3.12, por lo que aún corresponde ejecutar un
smoke test del entorno exacto Python 3.12/NumPy 1.26 en la máquina objetivo. El
soporte y los wheels para 3.12 sí se comprobaron contra metadata oficial.

En Linux, el wheel PyTorch predeterminado de PyPI puede apuntar a una versión
CUDA incompatible con una GPU antigua de 4 GB. Antes de instalar el proyecto,
elige un wheel CPU o CUDA que corresponda al driver y GPU local mediante el
[selector oficial de PyTorch](https://pytorch.org/get-started/locally/). La
Etapa 1 no necesita GPU.

## Instalación

Desde la raíz del repositorio:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Instala exactamente un wheel de PyTorch antes del proyecto. CUDA 12.6 es el
baseline conservador para GPU; usa el wheel CPU si CUDA no está disponible:

```bash
# Opción NVIDIA/CUDA 12.6
python -m pip install torch==2.12.1 \
  --index-url https://download.pytorch.org/whl/cu126

# Opción solo CPU (ejecuta esta en vez del comando anterior)
python -m pip install torch==2.12.1 \
  --index-url https://download.pytorch.org/whl/cpu
```

Luego instala el paquete y los tests:

```bash
python -m pip install -e ".[dev]"
```

Usa `python -m pip install -r requirements.txt` en vez del último comando para
instalar el baseline fijado de dependencias directas. En un equipo con GPU,
instala primero el wheel adecuado de PyTorch 2.12.1; la instalación restante lo
reutilizará. El proyecto no incluye Docker, interfaz web, aplicación de
escritorio, base de datos, servicios cloud ni modelos de audio.

## Preparación de datos

1. Copia únicamente archivos `.mid` o `.midi` autorizados en `data/raw/`.
   Puedes usar subdirectorios.
2. Revisa `configs/preprocess.yaml`. Las rutas relativas se resuelven desde la
   raíz del repositorio (el directorio padre de `configs/`).
3. Mantén privadas las fuentes: Git ignora los MIDI y todos los datos
   derivados.

El rango predeterminado es MIDI 21–108. Los eventos de compás ausentes pueden
interpretarse como 4/4, pero se rechazan eventos explícitos distintos de 4/4.
La cuantización está desactivada por defecto; al activarla,
`subdivisions_per_beat: 4` usa una grilla de semicorcheas. Las proporciones de
los splits deben sumar 1.0.

Cada manifiesto exitoso apunta a una sola corrida inmutable. Las corridas
anteriores se conservan por seguridad y solo deben borrarse manualmente después
de comprobar que su run ID no está referenciado por el manifiesto actual.

## Comandos probados de la Etapa 1

Inspecciona la colección:

```bash
python scripts/inspect_midis.py --input data/raw
```

Las opciones de inspección son `--pitch-min`, `--pitch-max` y `--output`. Por
ejemplo, para solicitar un reporte legible por máquinas:

```bash
python scripts/inspect_midis.py --input data/raw --output outputs/logs/inspection.json
```

Preprocesa, crea splits seguros y escribe el manifiesto:

```bash
python scripts/preprocess.py --config configs/preprocess.yaml
```

Ejecuta los tests:

```bash
python -m pytest
```

La suite actual contiene 73 tests sintéticos y escribe todos los fixtures MIDI
en directorios temporales; no se versiona ningún MIDI de dataset.

En la Etapa 1 no existe un comando funcional `train.py` ni `generate.py`. Sus
archivos de configuración documentan el roadmap, no funciones ejecutables.

## Entrenamiento y generación (futuro)

El entrenamiento usará secuencias REMI, cross-entropy del token siguiente
ignorando padding, gradient clipping, precisión mixta opcional cuando sea
estable, checkpoints y métricas de loss en TensorBoard. La generación admitirá
inicio vacío o MIDI semilla, semilla aleatoria reproducible, máximo de tokens,
temperature, top-k, top-p, número de muestras y una penalización sencilla de
repetición. No deberán guardarse MIDIs vacíos o inválidos. Ninguna de estas
funciones está disponible en la Etapa 1.

## Limitaciones actuales

- Se conserva una sola pista instrumental por ejemplo fuente.
- Solo se admite compás 4/4 y el tempo debe ser constante.
- Pitch bends no neutros, letras, audio y digitaciones de guitarra quedan fuera
  del alcance actual; los resets de pitchwheel con valor cero se eliminan.
- El aumento rítmico no está implementado.
- Se acepta polifonía sencilla, pero no se conserva la semántica de arreglos
  multipista complejos.
- Los unísonos duplicados exactos se colapsan. Se siguen rechazando note-on no
  idénticos superpuestos del mismo pitch/canal y notas colgantes, porque su
  interpretación en Standard MIDI File es ambigua.
- Los control changes se rechazan en vez de descartarlos silenciosamente, salvo
  el RPN exacto de rango de pitch bend que Guitar Pro escribe al inicio; aún no
  se renderiza el pedal de sustain.
- La completitud de la última frase usa la duración End-of-Track de la fuente y
  los fragmentos preservan silencios finales hasta el límite nominal de 2/4/8
  compases.
- El preprocesamiento no vuelve legalmente seguro un dataset inadecuado o mal
  etiquetado.
- En la Etapa 1 no se entrena un modelo ni se genera MIDI.

## Roadmap

1. **Hecho:** estructura, configuración, inspección, preprocesamiento,
   manifiesto, splits por fuente sin leakage, transposición y tests críticos de
   la Etapa 1.
2. Agregar tokenización REMI con MidiTok y tests de ida y vuelta de tokens.
3. Agregar dataset secuencial y data loader con padding correcto.
4. Agregar GRU pequeña, entrenamiento CPU/CUDA, gradient clipping,
   checkpoints, reanudación, precisión mixta cuando sea estable y métricas en
   TensorBoard.
5. Agregar muestreo con temperature, top-k y top-p; MIDI semilla opcional;
   validación; y exportación a `outputs/generated/`.

Las interfaces futuras previstas son `python scripts/train.py --config
configs/train.yaml` y `python scripts/generate.py --config
configs/generate.yaml`, opcionalmente con `--seed-midi`. **Estos comandos aún
no están implementados ni probados.**

## Licencia

El código fuente está bajo licencia MIT. La licencia cubre el código del
repositorio, no el dataset MIDI aportado por el usuario ni la situación legal
de los resultados generados. La responsabilidad sobre los derechos y usos
permitidos de los datos corresponde al usuario.
