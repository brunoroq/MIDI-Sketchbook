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

## Current status: Stages 1–4

Stage 1 is implemented: repository setup, MIDI inspection and basic
preprocessing. The executable workflow can:

- inspect `.mid` and `.midi` files without stopping the whole batch when one
  file is corrupt or incompatible;
- validate pitch range, 4/4 meter, constant tempo, pitch bends, lyrics,
  control changes, drum tracks, ambiguous note events, and minimum note counts;
- select either the non-drum `pretty_midi` instrument stream with the most
  notes or an explicit instrument index;
- remove initial silence and optionally quantize note onsets and durations;
- preserve expressive pitch-wheel curves after normalizing an explicit,
  constant source sensitivity to the canonical +/-6-semitone range, while
  removing neutral exporter bookkeeping and rejecting ambiguous curves;
- load optional hash-linked `*.mid.techniques.json` sidecars for dead notes,
  palm mute, slide direction, and vibrato, without treating an unannotated
  legacy MIDI as a confirmed negative example;
- collapse exact duplicate unison notes while retaining strict checks for
  unsupported musical controls;
- split material into configurable phrases of 2, 4, or 8 bars;
- assign each source file to exactly one train, validation, or test split
  before creating phrases, grouping byte-identical sources to prevent
  duplicate-content leakage;
- optionally transpose training material from -5 through +6 semitones while
  rejecting results outside the configured pitch range; and
- write each reproducible immutable run under
  `data/processed/runs/<run_id>/{train,validation,test}/` and an authoritative
  JSON manifest at `data/splits/manifest.json`.

Stage 2 REMI tokenization is also implemented. It:

- consumes only the fragments named by the authoritative Stage 1 manifest,
  never every file found under `data/processed`;
- encodes each one-track phrase with MidiTok 3, adds explicit `BOS` and `EOS`
  tokens, and reserves `PAD` for the Stage 3 data loader without storing it in
  the unpadded sequences;
- validates every sequence with an encode/decode/re-encode round trip while
  retaining the MIDI instrument program as out-of-band metadata;
- uses the local `GuitarREMI` vocabulary: native `PitchBend` events plus six
  postfix technique tokens (`DEAD_NOTE`, palm-mute on/off, slide up/down, and
  `VIBRATO`);
- saves the complete tokenizer vocabulary and one JSON sequence per fragment,
  preserving the Stage 1 train/validation/test assignment; and
- publishes content-derived immutable runs under `data/tokenized/runs/` plus
  the authoritative `data/tokenized/manifest.json`.

Stage 3 training is implemented as well. It:

- builds strict autoregressive train and validation datasets only from the
  entries in the authoritative Stage 2 manifest, verifying their hashes and
  refusing to silently truncate sequences;
- masks the post-`Duration` technique decision from loss for `UNLABELED`
  sequences, so legacy files still teach notes and rhythm without becoming
  false negative technique examples; `COMPLETE` examples remain fully trained;
- shifts each sequence into input/next-token targets and pads each batch only
  to its longest member, using the reserved `PAD` token;
- trains the configured two-layer GRU with AdamW and token-weighted
  cross-entropy that ignores padding, gradient clipping, deterministic seeds,
  CPU/CUDA selection, and optional CUDA mixed precision;
- evaluates the validation split after every epoch and records loss,
  perplexity, token counts, gradient norm, learning rate, and epoch duration;
- writes TensorBoard events, a JSON training report, and atomic `best.pt`,
  `latest.pt`, and periodic epoch checkpoints in an isolated training-run
  directory; and
- resumes only from a compatible checkpoint, restoring model, optimizer,
  scaler, data-loader, and random-number-generator state before continuing.

Stage 4 has a first unconditional generator. It:

- safely loads `best.pt` on CPU first, verifies its exact manifest, tokenizer,
  vocabulary, architecture, state tensors, and hashes, and only then moves the
  model to the requested CPU or CUDA device;
- samples reproducibly from `BOS` without a seed MIDI using temperature,
  top-k, top-p, repetition penalty, a token limit, and bounded retries;
- constrains token types and values to valid GuitarREMI continuations, limits
  simultaneous guitar notes, rejects overlapping same-pitch notes, and
  validates every decoded sequence before publishing it;
- exports editable MIDI, adding an explicit constant +/-6-semitone RPN when
  pitch-bend events are present;
- writes exact token/provenance JSON, a non-ingestible generated-technique
  sidecar, and an optional piano-roll PNG with bends and technique labels; and
- publishes each complete run atomically under `outputs/generated/<run_id>/`
  with SHA-256 hashes in a run manifest and never overwrites prior samples.

Generation conditioned by a seed MIDI is **not implemented yet**. The current
checkpoint is a one-epoch smoke model, so its outputs prove the complete path
rather than represent a musically mature model.

## Architecture and main decisions

The repository uses a small `src`-layout Python package named
`midi_idea_generator`. Command-line scripts stay thin; configuration,
validation, MIDI I/O, and transformations live in separate, testable modules.
YAML files hold experiment settings, errors are reported per file, random
seeds make splits reproducible, and generated artifacts remain outside version
control.

The Stage 1 manifest is the preprocessing/tokenization boundary: it records the
normalized configuration, implementation and dependency versions, source
SHA-256 hashes, split, instrument index, source duration/resolution, and every
current fragment. Stage 2 reads exactly its `fragments` entries and does not
glob `data/processed`, because that directory can contain older immutable runs.
The first two stages derive run IDs from their inputs and normalized
configuration:
identical work reuses the existing run, while changed inputs or settings create
an isolated run without deleting prior data.

The Stage 2 manifest is also the input contract for Stage 3. The data loader
reads exactly the recorded sequence paths and checks the manifest, tokenizer,
configuration, and sequence hashes. Checkpoints record these identities plus
the model, data, optimizer, implementation, PyTorch version, and seed settings;
resume is rejected if the current experiment is incompatible. The test split
is deliberately not evaluated during model fitting.

The REMI settings in `configs/tokenize.yaml` were chosen from the current riff
corpus rather than copied from a large generic dataset. The active grid uses 24
positions per beat for durations from 0 to 4 beats, preserving binary
subdivisions and triplets, then 4 positions per beat for durations up to 16
beats to avoid unnecessary vocabulary growth. Tempo tokens use a configured
40–250 BPM range, which covers the observed approximately 92–230 BPM. Velocity
tokens are disabled because the inspected
exports contain source-level velocity choices but no useful within-riff
dynamics; preserving them would lengthen every note without adding meaningful
variation. The guitar program is stored beside the token IDs instead of in the
REMI vocabulary, consistent with the one-track Stage 1 boundary.

The current preprocessing boundary deliberately assumes one principal
instrumental track, 4/4 meter, constant tempo per fragment, pitch bends with
an explicitly declared and constant source sensitivity and actual excursions
no wider than +/-6 semitones, and monophonic or modestly polyphonic material.
Techniques that ordinary MIDI
cannot distinguish use the optional strict sidecar contract documented in
[`docs/technique-sidecars.md`](docs/technique-sidecars.md). The source
file—not a derived phrase—is the unit of dataset splitting. Transposition
never changes that source split.

`track_number` in the manifest is retained to match the project requirement,
but in Stage 1 it means the zero-based index in `pretty_midi.instruments`. It
is also emitted explicitly as `instrument_index`; it is not the raw Standard
MIDI File track number, because one raw track can contain several channels or
program streams.

The implemented model is a compact two-layer autoregressive GRU rather than a
Transformer. The current baseline uses a 64-dimensional embedding, 128 hidden
units, dropout 0.2, and batch size 4. `data.max_sequence_length: 512` is a
safety bound rather than a truncation target: the loader raises an error if a
stored sequence exceeds it.

```text
configs/                    executable Stage 1–4 configs
data/raw/                   user-supplied MIDI (ignored by Git)
data/processed/runs/        immutable normalized runs, each grouped by split
data/tokenized/runs/        immutable REMI runs, each grouped by split
data/tokenized/manifest.json authoritative current tokenization manifest
data/splits/                authoritative Stage 1 preprocessing manifest
outputs/                    generated results, plots, and logs (ignored)
checkpoints/                Stage 3 training runs and checkpoints (ignored)
scripts/                    Stage 1–4 command-line entry points
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
and does not consume `pretty_midi.PrettyMIDI` objects. Stage 2 therefore keeps
a MIDI file boundary between preprocessing with pretty_midi and tokenization
with MidiTok instead of relying on legacy MidiTok 2.x examples.

The preprocessing/tokenization workflow and current corpus were exercised with
Python 3.12.13, MidiTok 3.0.6.post1, Symusic 0.5.9, pretty_midi 0.2.11, Mido
1.3.3, PyYAML 6.0.3, and NumPy 1.26.4. The automated tests use synthetic MIDI
fixtures in temporary directories; no dataset MIDI is committed.

On Linux, the default PyPI PyTorch wheel may target a CUDA release unsupported
by an older 4 GB GPU. Before installing the project, choose a CPU or CUDA wheel
that matches the local driver and GPU using the [official PyTorch
selector](https://pytorch.org/get-started/locally/). Stages 1 and 2 do not
require a GPU, and Stage 3 also supports CPU training, although it is slower.

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

Legacy MIDI files without a sidecar remain valid and are recorded as
`UNLABELED`; they teach notes and rhythm but do not provide technique targets.
For reviewed material, add the sibling sidecar described in
[`docs/technique-sidecars.md`](docs/technique-sidecars.md). A present empty
`COMPLETE` sidecar explicitly means “reviewed and contains no supported
techniques.” Keep the original Guitar Pro or MusicXML files as well: standard
MIDI cannot reliably tell a slide from a bend or recover every articulation.

The default pitch range is MIDI 21–108. Missing time-signature events may be
treated as 4/4, but explicit non-4/4 events are rejected. Quantization is off
by default; when enabled, `subdivisions_per_beat: 4` uses a sixteenth-note
grid. The configured split fractions must total 1.0.

Each successful preprocessing manifest points to one immutable run. Old runs
are intentionally left untouched for safety and can be removed manually only
after confirming that their run ID is not referenced by the current manifest.

Before tokenizing, review `configs/tokenize.yaml`. Stage 2 treats
`data/splits/manifest.json` as authoritative and verifies every referenced
fragment and hash. It does not discover additional MIDI files placed in an old
processed run.

## Stage 1–4 commands

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

Tokenize the current Stage 1 manifest with REMI:

```bash
python scripts/tokenize_midis.py --config configs/tokenize.yaml
```

The command writes one immutable, split-preserving tokenization run:

```text
data/tokenized/runs/<run_id>/
├── tokenizer.json
├── manifest.json
├── train/*.json
├── validation/*.json
└── test/*.json
data/tokenized/manifest.json       authoritative current-run manifest
```

The authoritative manifests report the live corpus size, accepted/rejected
sources, split counts, technique coverage, bend-token counts, and sequence
lengths. These values are intentionally not hard-coded here because `data/raw`
is a growing private collection. Stage 2 never truncates or pads stored files;
Stage 3 applies dynamic padding and rejects any sequence above the configured
maximum instead of silently shortening it.

The GuitarREMI vocabulary and schema differ from the earlier note-only
baseline. Checkpoints trained against an older tokenization manifest are
intentionally incompatible; preprocess, tokenize, and start a new training run
after this migration.

Train the GRU using the authoritative token manifest:

```bash
python scripts/train.py --config configs/train.yaml
```

For a short end-to-end smoke run, override the configured total epoch count:

```bash
python scripts/train.py --config configs/train.yaml --epochs 1
```

Each new training run writes artifacts to:

```text
checkpoints/<training_run_id>/
├── best.pt
├── latest.pt
├── epoch-0001.pt
├── ...
└── training_report.json
outputs/logs/training/<training_run_id>/   TensorBoard events
```

To continue a run, set `paths.resume_from` in `configs/train.yaml` to that
run's `latest.pt` and rerun the same command. Historical and `best.pt`
checkpoints are intentionally rejected to avoid branching over an existing
run directory. `training.epochs` and `--epochs` mean the desired **total** epoch
number, not the number of extra epochs. The checkpoint must still match the
tokenized corpus and all compatibility-critical training settings.

Inspect the live and saved metrics with:

```bash
tensorboard --logdir outputs/logs/training
```

Generate four unconditional ideas with the exact checkpoint and tokenization
manifest configured in `configs/generate.yaml`:

```bash
python scripts/generate.py --config configs/generate.yaml
```

The checkpoint path is explicit and must name a `best.pt` compatible with the
configured Stage 2 manifest. Generation refuses stale tokenizers, mismatched
vocabularies, unsafe output paths, malformed checkpoints, invalid token
streams, empty scores, and ambiguous overlapping notes. The default output is:

```text
outputs/generated/<generation_run_id>/
├── manifest.json
├── sample-001.mid
├── sample-001.tokens.json
├── sample-001.techniques.generated.json
├── sample-001.piano-roll.png
└── ...
```

The generated-technique JSON intentionally does not use the ingestible
`*.mid.techniques.json` suffix. It preserves exact `{type, note_index}` token
semantics; GuitarREMI v1 knows slide direction but not its original target
pitch. `generation.max_simultaneous_notes` defaults to 3 for riff-like output
and can be set from 1 through the physical six-note guitar limit.

Run the test suite:

```bash
python -m pytest
```

The suite writes all synthetic MIDI fixtures and training artifacts to
temporary directories; no dataset MIDI is committed.

## Training and generation

Training is available through `scripts/train.py` as described above. Its
default 50-epoch configuration includes validation-based early stopping after
8 epochs without sufficient improvement and saves enough state for an exact,
compatible continuation.

Unconditional generation is available through `scripts/generate.py`. It uses a
reproducible random seed, maximum/minimum token counts, temperature, top-k,
top-p, sample count, repetition penalty, a simultaneous-note cap, and bounded
retries. Invalid decoded MIDI is never published. Conditional continuation
from a seed MIDI remains future work.

## Limitations

- Only one selected instrumental track is retained per source example.
- The supported meter is 4/4 and tempo must remain constant within a source.
- Pitch bends require one explicit, constant source sensitivity and excursions
  no wider than the canonical +/-6-semitone range. Missing or changing
  sensitivity and wider actual excursions are rejected rather than guessed or
  clipped.
- Standard MIDI does not identify dead notes, palm mute, slide semantics, or
  vibrato reliably. They require a `COMPLETE` sidecar. Version 1 learns slide
  direction but intentionally does not tokenize its exact target pitch.
- Lyrics, audio, tablature strings/frets, pick direction, and other guitar
  fingering remain outside the current language.
- Rhythm augmentation is not implemented.
- Simple polyphony is accepted, but complex multi-track arrangement semantics
  are not preserved.
- Exact duplicate unisons are collapsed. Non-identical overlapping note-on
  events for the same pitch/channel and dangling notes remain rejected because
  their Standard MIDI File interpretation is ambiguous.
- Control changes are rejected rather than silently discarded, except for the
  validated pitch-bend-sensitivity RPN; sustain-pedal rendering is not
  implemented yet.
- Final phrase completeness is measured with the source End-of-Track duration,
  and generated phrase files preserve trailing rests up to their nominal
  2/4/8-bar boundary.
- Stage 2 intentionally discards velocity variation, quantizes timing to its
  configured REMI grid, and keeps the single guitar program outside the token
  stream as sequence metadata.
- Preprocessing cannot make an unsuitable or mislabeled dataset legally safe.
- Seed-MIDI conditioning, beam search, and an interactive audition interface
  are not implemented.
- The current corpus has pitch-bend examples but no `COMPLETE` technique
  sidecars. Technique tokens exist in the language and exporter, but a model
  cannot learn their musical use until supervised examples are added.

## Roadmap

1. **Done:** structure, configuration, inspection, preprocessing, manifest,
   leakage-safe source splits, transposition, and critical Stage 1 tests.
2. **Done:** data-driven REMI tokenization with MidiTok, explicit `PAD`/`BOS`/
   `EOS`, native pitch bends, the first guitar-technique token language, token
   round-trip validation, immutable token runs, and Stage 2 tests.
3. **Done:** strict autoregressive datasets and dynamic padding.
4. **Done:** compact GRU, CPU/CUDA training, validation, gradient clipping,
   compatible exact resume, atomic checkpoints, early stopping, reports, and
   TensorBoard metrics.
5. **Done:** reproducible unconditional temperature/top-k/top-p sampling,
   GuitarREMI constraints, strict checkpoint provenance, MIDI/bend export,
   generated-technique metadata, piano-roll images, and immutable manifests in
   `outputs/generated/`.
6. Add seed-MIDI conditioning, model-quality evaluation, and an audition/
   curation workflow after training on the expanded corpus.

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

## Estado actual: Etapas 1–4

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
- conservar curvas expresivas de pitch bend normalizando una sensibilidad
  fuente explícita y constante al rango canónico de +/-6 semitonos, eliminando
  solo bookkeeping neutro y rechazando curvas ambiguas;
- cargar sidecars opcionales `*.mid.techniques.json`, ligados por hash, para
  notas muertas, palm mute, dirección de slide y vibrato, sin interpretar un
  MIDI antiguo no anotado como ejemplo negativo confirmado;
- colapsar unísonos duplicados exactos y mantener controles estrictos para
  eventos musicales no soportados;
- dividir el material en frases configurables de 2, 4 u 8 compases;
- asignar cada archivo fuente a un único split de entrenamiento, validación o
  prueba antes de fragmentarlo y agrupar fuentes idénticas byte a byte para
  evitar leakage por contenido duplicado;
- transponer opcionalmente el material de entrenamiento entre -5 y +6
  semitonos, descartando resultados fuera del rango configurado; y
- guardar cada corrida inmutable y reproducible en
  `data/processed/runs/<run_id>/{train,validation,test}/` y el manifiesto JSON
  autoritativo en `data/splits/manifest.json`.

La tokenización REMI de la Etapa 2 también está implementada. Esta etapa:

- consume únicamente los fragmentos enumerados por el manifiesto autoritativo
  de la Etapa 1, sin recorrer todos los archivos de `data/processed`;
- codifica cada frase de una pista con MidiTok 3, agrega tokens `BOS` y `EOS`
  explícitos y reserva `PAD` para el data loader de la Etapa 3 sin guardarlo
  dentro de las secuencias aún no rellenadas;
- valida cada secuencia mediante codificación/decodificación/recodificación y
  conserva el programa del instrumento MIDI como metadata fuera del stream;
- usa el vocabulario local `GuitarREMI`: eventos `PitchBend` nativos y seis
  tokens postfix (`DEAD_NOTE`, palm mute on/off, slide up/down y `VIBRATO`);
- guarda el vocabulario completo y una secuencia JSON por fragmento sin cambiar
  su asignación a entrenamiento, validación o prueba; y
- publica corridas inmutables derivadas del contenido en
  `data/tokenized/runs/`, junto al manifiesto autoritativo
  `data/tokenized/manifest.json`.

El entrenamiento de la Etapa 3 también está implementado. Esta etapa:

- construye datasets autorregresivos estrictos de entrenamiento y validación
  únicamente desde las entradas del manifiesto autoritativo de la Etapa 2,
  verifica sus hashes y nunca trunca secuencias silenciosamente;
- excluye de la pérdida la decisión técnica posterior a `Duration` en
  secuencias `UNLABELED`, de modo que los archivos antiguos enseñen notas y
  ritmo sin convertirse en falsos negativos; los ejemplos `COMPLETE` sí se
  entrenan íntegramente;
- desplaza cada secuencia para formar pares entrada/token siguiente y rellena
  cada batch solo hasta su miembro más largo, usando el token `PAD` reservado;
- entrena la GRU configurada de dos capas con AdamW, cross-entropy ponderada por
  token que ignora el padding, gradient clipping, semillas deterministas,
  selección CPU/CUDA y precisión mixta opcional en CUDA;
- evalúa el split de validación después de cada época y registra loss,
  perplexity, cantidad de tokens, norma del gradiente, learning rate y duración;
- escribe eventos de TensorBoard, un reporte JSON y checkpoints atómicos
  `best.pt`, `latest.pt` y periódicos dentro de una carpeta aislada por corrida;
  y
- reanuda únicamente un checkpoint compatible, restaurando el modelo,
  optimizador, scaler, data loader y los generadores aleatorios antes de seguir.

La Etapa 4 ya incluye una primera generación incondicional. Esta etapa:

- carga `best.pt` primero en CPU, comprueba su manifiesto, tokenizer,
  vocabulario, arquitectura, tensores e identidades SHA-256 exactas, y solo
  después mueve el modelo al dispositivo CPU o CUDA solicitado;
- muestrea de forma reproducible desde `BOS`, sin MIDI semilla, usando
  temperature, top-k, top-p, penalización de repetición, límite de tokens e
  intentos acotados;
- restringe tipos y valores a continuaciones GuitarREMI válidas, limita las
  notas simultáneas de guitarra, rechaza notas superpuestas del mismo pitch y
  valida cada secuencia decodificada antes de publicarla;
- exporta MIDI editable e incorpora el RPN explícito y constante de +/-6
  semitonos cuando la muestra contiene pitch bends;
- escribe JSON de tokens y procedencia, un sidecar no ingerible de técnicas
  generadas y un piano-roll PNG opcional con bends y etiquetas; y
- publica cada corrida completa de forma atómica en
  `outputs/generated/<run_id>/`, con hashes, sin sobrescribir muestras previas.

La continuación condicionada por un MIDI semilla **todavía no está
implementada**. El checkpoint actual es una prueba de una sola época: demuestra
el flujo completo, pero aún no representa un modelo musicalmente maduro.

## Arquitectura y decisiones principales

El repositorio usa un paquete Python con layout `src` llamado
`midi_idea_generator`. Los scripts de línea de comandos son delgados; la
configuración, validación, entrada/salida MIDI y transformaciones viven en
módulos separados y comprobables. YAML almacena los parámetros, los errores se
reportan por archivo, las semillas hacen reproducibles los splits y los
artefactos generados quedan fuera del control de versiones.

El manifiesto de la Etapa 1 define el límite entre preprocesamiento y
tokenización: registra configuración normalizada, versiones de implementación
y dependencias, SHA-256 de cada fuente, split, índice de instrumento,
duración/resolución y cada fragmento vigente. La Etapa 2 lee exactamente sus
entradas `fragments` y nunca recorre todo `data/processed`, que puede contener
corridas inmutables antiguas. Las dos primeras etapas derivan el run ID de sus
entradas y configuración normalizada: un trabajo idéntico reutiliza la corrida
existente; cualquier cambio crea una corrida aislada sin borrar datos previos.

El manifiesto de la Etapa 2 también es el contrato de entrada de la Etapa 3. El
data loader lee exactamente las rutas registradas y verifica los hashes del
manifiesto, tokenizer, configuración y secuencias. Los checkpoints registran
estas identidades junto a la configuración del modelo, datos y optimizador, la
implementación, la versión de PyTorch y la semilla; la reanudación se rechaza
si el experimento vigente es incompatible. El split de prueba no se evalúa
durante el ajuste del modelo.

Los parámetros REMI de `configs/tokenize.yaml` se eligieron a partir del corpus
actual de riffs, no copiando la configuración de un dataset genérico grande.
La grilla activa usa 24 posiciones por pulso para duraciones entre 0 y 4 pulsos,
lo que preserva subdivisiones binarias y tresillos, y 4 posiciones por pulso
para duraciones de hasta 16 pulsos, evitando aumentar el vocabulario sin
necesidad. Los tokens de tempo usan un rango configurado de 40–250 BPM, que
cubre los aproximadamente 92–230 BPM observados. Los tokens de velocity están
deshabilitados porque las exportaciones
inspeccionadas tienen elecciones de velocity por fuente, pero no dinámicas
útiles dentro del riff; conservarlos alargaría cada nota sin aportar variación
relevante. El programa de guitarra se guarda junto a los IDs como metadata, no
dentro del vocabulario REMI, de acuerdo con el límite de una sola pista de la
Etapa 1.

El límite actual asume una pista instrumental principal, compás 4/4, tempo
constante por fragmento, pitch bends con sensibilidad fuente explícita y
constante cuya excursión real no supere +/-6 semitonos, y material monofónico
o con polifonía sencilla. Las técnicas
que un MIDI común no puede distinguir usan el contrato opcional y estricto de
[`docs/technique-sidecars.md`](docs/technique-sidecars.md). La unidad para
dividir el dataset es el archivo fuente, no la frase derivada. La transposición
nunca cambia el split de origen.

El campo `track_number` se conserva en el manifiesto para cumplir el contrato
del proyecto, pero en esta etapa significa el índice base cero dentro de
`pretty_midi.instruments`. También se publica como `instrument_index`; no es el
número de track crudo del Standard MIDI File, porque un track puede contener
varios canales o programas.

El modelo implementado es una GRU autorregresiva pequeña de dos capas, no un
Transformer. El baseline actual usa embedding de 64 dimensiones, estado oculto
de 128, dropout 0.2 y batch 4. `data.max_sequence_length: 512` es un límite de
seguridad, no un objetivo de truncado: el loader produce un error si una
secuencia guardada lo supera.

## Estructura del proyecto

```text
configs/                    configs ejecutables de Etapas 1–4
data/raw/                   MIDI del usuario (ignorado por Git)
data/processed/runs/        corridas inmutables, cada una separada por split
data/tokenized/runs/        corridas REMI inmutables separadas por split
data/tokenized/manifest.json manifiesto autoritativo de tokenización actual
data/splits/                manifiesto autoritativo de preprocesamiento
outputs/                    resultados, gráficos y logs (ignorados)
checkpoints/                corridas y checkpoints de Etapa 3 (ignorados)
scripts/                    comandos de las Etapas 1–4
src/midi_idea_generator/    código reutilizable del paquete
tests/                      tests automáticos
```

## Tecnologías y compatibilidad

El objetivo es Python 3.12. `pyproject.toml` declara PyTorch, MidiTok 3.x,
pretty_midi 0.2.11, Mido, NumPy, PyYAML, Matplotlib, TensorBoard y pytest.

La compatibilidad entre Python, PyTorch y CUDA depende de los wheels y de la
plataforma. El baseline conservador declarado usa PyTorch 2.12.1,
MidiTok 3.0.6.post1, pretty_midi 0.2.11 y NumPy 1.26.4. Versiones antiguas
pueden no soportar Python 3.12 y las versiones mayores de MidiTok tienen APIs
incompatibles. En particular, MidiTok 3 recibe rutas MIDI u objetos
`symusic.Score`, no objetos `pretty_midi.PrettyMIDI`. Por eso la Etapa 2
mantiene un límite por archivo MIDI entre el preprocesamiento con pretty_midi y
la tokenización con MidiTok, en vez de depender de ejemplos antiguos de la API
2.x.

El flujo de preprocesamiento/tokenización y el corpus actual se ejercitaron con
Python 3.12.13, MidiTok 3.0.6.post1, Symusic 0.5.9, pretty_midi 0.2.11, Mido
1.3.3, PyYAML 6.0.3 y NumPy 1.26.4. Los tests automáticos usan fixtures MIDI
sintéticos en directorios temporales; no se versiona ningún MIDI del dataset.

En Linux, el wheel PyTorch predeterminado de PyPI puede apuntar a una versión
CUDA incompatible con una GPU antigua de 4 GB. Antes de instalar el proyecto,
elige un wheel CPU o CUDA que corresponda al driver y GPU local mediante el
[selector oficial de PyTorch](https://pytorch.org/get-started/locally/). Las
Etapas 1 y 2 no necesitan GPU y la Etapa 3 también admite entrenamiento en CPU,
aunque es más lento.

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

Los MIDIs antiguos sin sidecar siguen siendo válidos y quedan registrados como
`UNLABELED`: enseñan notas y ritmo, pero no aportan objetivos de técnicas. Para
material revisado, agrega el sidecar hermano descrito en
[`docs/technique-sidecars.md`](docs/technique-sidecars.md). Un sidecar
`COMPLETE` presente y vacío significa explícitamente «revisado y sin técnicas
soportadas». Conserva además los originales Guitar Pro o MusicXML: el MIDI
estándar no permite distinguir de forma fiable un slide de un bend ni recuperar
todas las articulaciones.

El rango predeterminado es MIDI 21–108. Los eventos de compás ausentes pueden
interpretarse como 4/4, pero se rechazan eventos explícitos distintos de 4/4.
La cuantización está desactivada por defecto; al activarla,
`subdivisions_per_beat: 4` usa una grilla de semicorcheas. Las proporciones de
los splits deben sumar 1.0.

Cada manifiesto exitoso apunta a una sola corrida inmutable. Las corridas
anteriores se conservan por seguridad y solo deben borrarse manualmente después
de comprobar que su run ID no está referenciado por el manifiesto actual.

Antes de tokenizar, revisa `configs/tokenize.yaml`. La Etapa 2 considera
autoritativo `data/splits/manifest.json` y verifica cada fragmento referenciado
y su hash. No descubre MIDIs adicionales colocados dentro de una corrida
procesada antigua.

## Comandos de las Etapas 1–4

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

Tokeniza con REMI el manifiesto vigente de la Etapa 1:

```bash
python scripts/tokenize_midis.py --config configs/tokenize.yaml
```

El comando escribe una corrida inmutable que conserva los splits:

```text
data/tokenized/runs/<run_id>/
├── tokenizer.json
├── manifest.json
├── train/*.json
├── validation/*.json
└── test/*.json
data/tokenized/manifest.json       manifiesto autoritativo de la corrida actual
```

Los manifiestos autoritativos informan el tamaño vivo del corpus, fuentes
aceptadas/rechazadas, splits, cobertura de técnicas, tokens de bends y longitudes
de secuencia. Esos valores no se fijan aquí porque `data/raw` es una colección
privada en crecimiento. La Etapa 2 nunca trunca ni rellena los archivos
guardados; la Etapa 3 aplica padding dinámico y rechaza una secuencia que exceda
el máximo configurado, en vez de acortarla silenciosamente.

El vocabulario y schema GuitarREMI son distintos del baseline anterior de solo
notas. Los checkpoints entrenados con un manifiesto de tokenización antiguo son
deliberadamente incompatibles: después de esta migración hay que preprocesar,
tokenizar e iniciar un entrenamiento nuevo.

Entrena la GRU desde el manifiesto autoritativo de tokens:

```bash
python scripts/train.py --config configs/train.yaml
```

Para una prueba corta de punta a punta, sobrescribe el total configurado de
épocas:

```bash
python scripts/train.py --config configs/train.yaml --epochs 1
```

Cada entrenamiento nuevo escribe artefactos en:

```text
checkpoints/<training_run_id>/
├── best.pt
├── latest.pt
├── epoch-0001.pt
├── ...
└── training_report.json
outputs/logs/training/<training_run_id>/   eventos de TensorBoard
```

Para continuar una corrida, asigna a `paths.resume_from` en
`configs/train.yaml` su `latest.pt` y ejecuta el mismo comando. Los checkpoints
históricos y `best.pt` se rechazan deliberadamente para no crear una rama
sobre la misma carpeta de corrida. `training.epochs` y `--epochs`
indican el número **total** de épocas deseado, no cuántas épocas adicionales
ejecutar. El checkpoint debe seguir coincidiendo con el corpus tokenizado y
todos los ajustes críticos de compatibilidad.

Inspecciona las métricas guardadas y en vivo con:

```bash
tensorboard --logdir outputs/logs/training
```

Genera cuatro ideas incondicionales con el checkpoint y el manifiesto de
tokens indicados explícitamente en `configs/generate.yaml`:

```bash
python scripts/generate.py --config configs/generate.yaml
```

El checkpoint debe ser un `best.pt` compatible con el manifiesto de la Etapa
2 configurado. La generación rechaza tokenizers antiguos, vocabularios
distintos, rutas inseguras, checkpoints malformados, streams inválidos, scores
vacíos y notas ambiguas superpuestas. La salida predeterminada es:

```text
outputs/generated/<generation_run_id>/
├── manifest.json
├── sample-001.mid
├── sample-001.tokens.json
├── sample-001.techniques.generated.json
├── sample-001.piano-roll.png
└── ...
```

El JSON de técnicas generadas no usa deliberadamente el sufijo ingerible
`*.mid.techniques.json`. Conserva la semántica exacta `{type, note_index}` de
los tokens; GuitarREMI v1 conoce la dirección de un slide, pero no su pitch
destino original. `generation.max_simultaneous_notes` vale 3 por defecto para
resultados tipo riff y admite valores entre 1 y el límite físico de seis notas.

Ejecuta los tests:

```bash
python -m pytest
```

La suite escribe todos los fixtures MIDI sintéticos y artefactos de
entrenamiento en directorios temporales; no se versiona ningún MIDI del
dataset.

## Entrenamiento y generación

El entrenamiento está disponible mediante `scripts/train.py`, como se describe
arriba. La configuración predeterminada de 50 épocas incluye early stopping
basado en validación después de 8 épocas sin mejora suficiente y guarda el
estado necesario para una continuación exacta y compatible.

La generación incondicional está disponible mediante `scripts/generate.py`.
Admite semilla reproducible, límites mínimo/máximo de tokens, temperature,
top-k, top-p, cantidad de muestras, penalización de repetición, límite de notas
simultáneas e intentos acotados. Nunca publica un MIDI decodificado inválido.
La continuación condicionada desde un MIDI semilla sigue siendo trabajo futuro.

## Limitaciones actuales

- Se conserva una sola pista instrumental por ejemplo fuente.
- Solo se admite compás 4/4 y el tempo debe ser constante.
- Los pitch bends requieren una sensibilidad fuente explícita y constante y
  una excursión no mayor que el rango canónico de +/-6 semitonos. Una
  sensibilidad ausente o cambiante y las excursiones reales más amplias se
  rechazan: no se adivinan ni recortan.
- El MIDI estándar no identifica de forma fiable notas muertas, palm mute,
  semántica de slide o vibrato; requieren un sidecar `COMPLETE`. La versión 1
  aprende la dirección del slide, pero no tokeniza su pitch destino exacto.
- Letras, audio, cuerdas/trastes de tablatura, dirección de púa y otras
  digitaciones siguen fuera del lenguaje actual.
- El aumento rítmico no está implementado.
- Se acepta polifonía sencilla, pero no se conserva la semántica de arreglos
  multipista complejos.
- Los unísonos duplicados exactos se colapsan. Se siguen rechazando note-on no
  idénticos superpuestos del mismo pitch/canal y notas colgantes, porque su
  interpretación en Standard MIDI File es ambigua.
- Los control changes se rechazan en vez de descartarlos silenciosamente, salvo
  el RPN validado de sensibilidad de pitch bend; aún no se renderiza el pedal
  de sustain.
- La completitud de la última frase usa la duración End-of-Track de la fuente y
  los fragmentos preservan silencios finales hasta el límite nominal de 2/4/8
  compases.
- La Etapa 2 descarta intencionalmente las variaciones de velocity, cuantiza el
  timing a la grilla REMI configurada y conserva el único programa de guitarra
  fuera del stream, como metadata de la secuencia.
- El preprocesamiento no vuelve legalmente seguro un dataset inadecuado o mal
  etiquetado.
- El condicionamiento con MIDI semilla, beam search y una interfaz interactiva
  de audición todavía no están implementados.
- El corpus actual contiene pitch bends, pero ningún sidecar de técnicas
  `COMPLETE`. Los tokens y el exportador ya existen, pero el modelo no puede
  aprender el uso musical de esas técnicas hasta recibir ejemplos supervisados.

## Roadmap

1. **Hecho:** estructura, configuración, inspección, preprocesamiento,
   manifiesto, splits por fuente sin leakage, transposición y tests críticos de
   la Etapa 1.
2. **Hecho:** tokenización REMI basada en los datos con MidiTok, `PAD`/`BOS`/
   `EOS` explícitos, pitch bends nativos, el primer lenguaje de tokens de
   técnicas de guitarra, validación de ida y vuelta, corridas inmutables de
   tokens y tests de la Etapa 2.
3. **Hecho:** datasets autorregresivos estrictos y padding dinámico.
4. **Hecho:** GRU compacta, entrenamiento CPU/CUDA, validación, gradient
   clipping, reanudación exacta y compatible, checkpoints atómicos, early
   stopping, reportes y métricas en TensorBoard.
5. **Hecho:** muestreo incondicional reproducible con temperature/top-k/top-p,
   restricciones GuitarREMI, procedencia estricta del checkpoint, exportación
   MIDI/bends, metadata de técnicas generadas, piano-roll y manifiestos
   inmutables en `outputs/generated/`.
6. Agregar condicionamiento por MIDI semilla, evaluación de calidad y un flujo
   de audición/curación después de entrenar con el corpus ampliado.

## Licencia

El código fuente está bajo licencia MIT. La licencia cubre el código del
repositorio, no el dataset MIDI aportado por el usuario ni la situación legal
de los resultados generados. La responsabilidad sobre los derechos y usos
permitidos de los datos corresponde al usuario.
