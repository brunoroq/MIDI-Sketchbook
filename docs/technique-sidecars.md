# Sidecars de técnicas de guitarra

Este documento describe el contrato v1 implementado en
`midi_idea_generator.techniques`. Un sidecar permite anotar técnicas que no se
pueden recuperar de forma fiable desde un MIDI convencional sin modificar el
archivo musical original.

## Nombre y estado de etiquetado

El sidecar es opcional y debe estar junto al MIDI. Su nombre conserva la
extensión completa del archivo fuente:

```text
riff.mid
riff.mid.techniques.json

otra-frase.midi
otra-frase.midi.techniques.json
```

Estos dos casos tienen significados distintos:

- **Sidecar ausente:** la muestra está **UNLABELED**. No existe una afirmación
  sobre si contiene o no técnicas. La API devuelve `None`.
- **Sidecar presente, `coverage: "COMPLETE"` y listas vacías:** se revisó la
  muestra y se confirma que no contiene ninguna de las técnicas admitidas por
  este contrato.

Por tanto, no se debe interpretar automáticamente un MIDI sin sidecar como un
ejemplo negativo de todas las técnicas. Si una muestra antigua fue revisada y
se sabe que no contiene ninguna, conviene crearle un sidecar vacío para dejar
esa decisión registrada explícitamente:

Durante el entrenamiento, las secuencias `UNLABELED` conservan el aprendizaje
de notas, ritmo, tempo y bends, pero la transición posterior a cada token
`Duration_*` se excluye de la pérdida. Así no penalizan artificialmente los
tokens de técnicas. En una secuencia `COMPLETE` esa transición sí participa en
la pérdida, incluso cuando el sidecar está vacío.

```json
{
  "schema_version": 1,
  "source_midi": "riff.mid",
  "source_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "ticks_per_quarter": 480,
  "instrument_index": 0,
  "coverage": "COMPLETE",
  "note_techniques": [],
  "palm_mute_ranges": []
}
```

El valor de `source_sha256` anterior es ilustrativo. En un sidecar real debe
ser el SHA-256 exacto de los bytes del MIDI correspondiente.

## Ejemplo completo

El siguiente ejemplo es válido para un archivo llamado `riff.mid`, con
resolución de 480 ticks por negra, cuyo instrumento seleccionado es el índice
cero. Las referencias de nota y el SHA-256 deben coincidir con el MIDI real.

```json
{
  "schema_version": 1,
  "source_midi": "riff.mid",
  "source_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "ticks_per_quarter": 480,
  "instrument_index": 0,
  "coverage": "COMPLETE",
  "note_techniques": [
    {
      "note": {
        "onset_tick": 0,
        "end_tick": 240,
        "pitch": 52,
        "velocity": 96
      },
      "techniques": [
        {
          "type": "DEAD_NOTE"
        }
      ]
    },
    {
      "note": {
        "onset_tick": 480,
        "end_tick": 720,
        "pitch": 55,
        "velocity": 94
      },
      "techniques": [
        {
          "type": "SLIDE",
          "direction": "UP",
          "target_pitch": 60
        },
        {
          "type": "VIBRATO"
        }
      ]
    }
  ],
  "palm_mute_ranges": [
    {
      "start_tick": 0,
      "end_tick": 720
    }
  ]
}
```

El rango de palm mute del ejemplo afecta a las notas que comienzan en los
ticks 0 y 480. No afectaría a una nota que comenzara exactamente en el tick
720 porque el extremo final es exclusivo.

## Campos de raíz

El objeto raíz debe contener exactamente estas ocho claves; no se admiten
campos adicionales ni faltantes:

| Campo | Contrato |
|---|---|
| `schema_version` | Entero `1`. Los booleanos no cuentan como enteros. |
| `source_midi` | Nombre exacto del MIDI hermano, sin directorios. |
| `source_sha256` | SHA-256 del MIDI: 64 caracteres hexadecimales minúsculos. |
| `ticks_per_quarter` | Entero positivo igual a `midi.resolution`. |
| `instrument_index` | Índice no negativo del instrumento seleccionado. |
| `coverage` | Debe ser exactamente `"COMPLETE"`. |
| `note_techniques` | Array de anotaciones ligadas a notas exactas. |
| `palm_mute_ranges` | Array de intervalos temporales half-open. |

`instrument_index` es el índice de `pretty_midi.instruments`, no el número de
track crudo del Standard MIDI File. Además de estar dentro del MIDI, debe
coincidir con el índice que el llamador entrega a
`load_technique_sidecar(...)`.

## Identidad exacta de nota: `NoteRef`

Cada entrada de `note_techniques` contiene exactamente `note` y `techniques`.
`note` contiene exactamente:

```json
{
  "onset_tick": 480,
  "end_tick": 720,
  "pitch": 55,
  "velocity": 94
}
```

Las reglas son:

- `onset_tick` es un entero mayor o igual que cero.
- `end_tick` es un entero estrictamente mayor que `onset_tick`.
- `pitch` está entre 0 y 127.
- `velocity` está entre 1 y 127.
- La tupla completa debe coincidir con una nota del instrumento seleccionado.
- Una misma `NoteRef` no puede aparecer dos veces en `note_techniques`.

Los ticks son posiciones musicales absolutas desde el origen del MIDI. No son
segundos, ticks delta, posiciones relativas al compás ni posiciones relativas
a un fragmento. Esta representación evita desalinear anotaciones al trabajar
con diferentes tempos.

La implementación obtiene las referencias canónicas mediante
`midi.time_to_tick(note.start)` y `midi.time_to_tick(note.end)`. Si el MIDI
contiene duplicados exactamente iguales, estos comparten una sola identidad
semántica para el sidecar.

## Técnicas ligadas a una nota

El array `techniques` de cada nota debe ser no vacío. El orden de entrada no es
significativo; el cargador lo normaliza.

### Nota muerta

```json
{
  "type": "DEAD_NOTE"
}
```

No admite campos adicionales. `DEAD_NOTE` no puede coexistir en la misma nota
con `SLIDE` ni `VIBRATO`. Sí puede pertenecer a un rango de palm mute.

### Vibrato

```json
{
  "type": "VIBRATO"
}
```

No admite intensidad, curva ni otros parámetros en la versión 1.

### Slide

```json
{
  "type": "SLIDE",
  "direction": "UP",
  "target_pitch": 60
}
```

Un slide contiene exactamente `type`, `direction` y `target_pitch`:

- `direction` solo puede ser `"UP"` o `"DOWN"`.
- `target_pitch` es un pitch MIDI absoluto entre 0 y 127.
- El destino debe estar entre 1 y 24 semitonos de la nota de origen.
- Para `UP`, el destino debe ser mayor que el pitch de origen.
- Para `DOWN`, el destino debe ser menor que el pitch de origen.
- Una nota puede contener como máximo un slide.
- `SLIDE` puede coexistir con `VIBRATO`, pero no con `DEAD_NOTE`.

Al cargar, `SLIDE` se normaliza a uno de los tipos de dominio finales
`SLIDE_UP` o `SLIDE_DOWN`, conservando `target_pitch`.

## Rangos de palm mute

Cada elemento de `palm_mute_ranges` contiene exactamente:

```json
{
  "start_tick": 0,
  "end_tick": 960
}
```

El intervalo es **half-open**: una nota se considera palm-muted cuando se
cumple:

```text
start_tick <= note.onset_tick < end_tick
```

La duración de la nota no interviene en esta decisión; se etiqueta su ataque.
Por ejemplo, una nota sostenida cuyo onset ocurrió antes de `start_tick` no se
incorpora al rango aunque siga sonando dentro de él.

Restricciones:

- `start_tick` es un entero no negativo.
- `end_tick` es estrictamente mayor que `start_tick`.
- `end_tick` no puede superar la duración estructural del MIDI.
- Cada rango debe afectar al menos a un onset del instrumento seleccionado.
- Los rangos no pueden solaparse.
- Los rangos adyacentes tampoco se aceptan; deben fusionarse en uno solo.

El cargador conserva los rangos ordenados y, además, los expande a la técnica
de dominio `PALM_MUTE` para cada `NoteRef` afectada. No se debe escribir
`{"type": "PALM_MUTE"}` dentro de `note_techniques`: en JSON, palm mute se
representa exclusivamente mediante `palm_mute_ranges`.

## Resultado canónico de la API

La API pública es:

```python
from midi_idea_generator.techniques import load_technique_sidecar

sidecar = load_technique_sidecar(
    "data/raw/riff.mid",
    source_sha256=source_sha256,
    midi=midi,
    instrument_index=0,
)
```

Si el archivo no existe, devuelve `None`. Si está presente, cualquier error
produce `TechniqueSidecarError`; nunca se descartan silenciosamente
anotaciones inválidas.

Un `TechniqueSidecar` válido expone:

- `path`: ruta absoluta resuelta del sidecar;
- `fingerprint` y su alias `sha256`: SHA-256 de los bytes del sidecar;
- `size_bytes`: tamaño exacto del sidecar;
- los datos de enlace con el MIDI;
- `palm_mute_ranges` en orden canónico;
- `note_techniques`, incluyendo el palm mute expandido;
- `annotations_by_note`, un mapping inmutable;
- `techniques_for(note)`, que devuelve una tupla vacía si la nota no está
  anotada.

Los tipos de dominio finales son:

```text
DEAD_NOTE
PALM_MUTE
SLIDE_UP
SLIDE_DOWN
VIBRATO
```

Las notas se ordenan por onset, pitch, fin y velocity. Dentro de una nota, las
técnicas se ordenan como en la lista anterior. Los rangos se ordenan por inicio
y fin.

## Validación de seguridad e integridad

El cargador aplica una frontera estricta:

- JSON UTF-8 válido, sin claves duplicadas, `NaN` ni infinitos;
- objetos con claves exactas y valores de tipos exactos;
- máximo 1 MiB por sidecar;
- máximo 10.000 anotaciones, incluyendo técnicas explícitas y rangos; la
  expansión por nota también está limitada;
- MIDI y sidecar deben ser archivos regulares y no symlinks;
- el SHA-256 suministrado, el declarado y el calculado desde el MIDI deben
  coincidir;
- `source_midi`, PPQ e instrumento deben coincidir con el MIDI recibido;
- todas las referencias y rangos deben resolver sin ambigüedad;
- el MIDI y el sidecar se vuelven a comprobar al terminar para detectar
  cambios mientras se cargaban.

Editar el MIDI invalida deliberadamente su sidecar, aunque el cambio parezca
pequeño. Después de cualquier edición o nueva exportación se deben regenerar
el SHA-256 y todas las `NoteRef`.

## Flujo recomendado al recopilar muestras

1. Guarda cada MIDI autorizado dentro de `data/raw` usando un nombre estable.
2. Decide el estado real de la muestra:
   - si todavía no fue revisada, déjala sin sidecar y regístrala como
     **UNLABELED**;
   - si fue revisada y no contiene técnicas, crea un sidecar `COMPLETE` vacío;
   - si contiene técnicas, etiqueta todas las técnicas v1 y usa igualmente
     `coverage: "COMPLETE"`.
3. Calcula el SHA-256 después de la exportación definitiva del MIDI, no antes.
4. Usa el mismo `instrument_index` que seleccionará el preprocesamiento.
5. Extrae onset, fin, pitch y velocity desde el MIDI analizado; no estimes los
   ticks convirtiendo segundos manualmente.
6. Marca las notas muertas, slides y vibratos por `NoteRef`; agrupa secuencias
   continuas de palm mute en rangos no solapados ni adyacentes.
7. Carga el sidecar con `load_technique_sidecar` y corrige cualquier error antes
   de incorporar la muestra al corpus.
8. Si vuelves a editar o exportar el MIDI, considera obsoleto el sidecar y
   recréalo.

Para inspeccionar las identidades exactas que espera el contrato se puede usar
la misma conversión que el cargador:

```python
from hashlib import sha256
from pathlib import Path

from midi_idea_generator.midi_io import read_midi

path = Path("data/raw/riff.mid")
midi = read_midi(path)
instrument_index = 0

print("source_sha256:", sha256(path.read_bytes()).hexdigest())
print("ticks_per_quarter:", midi.resolution)
for note in midi.instruments[instrument_index].notes:
    print(
        {
            "onset_tick": int(midi.time_to_tick(note.start)),
            "end_tick": int(midi.time_to_tick(note.end)),
            "pitch": int(note.pitch),
            "velocity": int(note.velocity),
        }
    )
```

Este contrato no intenta reconstruir digitación, cuerda, traste, intensidad de
vibrato ni forma exacta de una curva. Su finalidad es conservar un lenguaje
simbólico pequeño, validable y transformable para el entrenamiento.
