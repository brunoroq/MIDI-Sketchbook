# Condicionamiento por tónica y modo

El modelo usa dos tokens de contexto al comienzo de cada secuencia:

```text
BOS Tonic_E Mode_PHRYGIAN Bar Position_0 Tempo_... ... EOS
```

Estos tokens indican el centro tonal y el modo esperado, pero no prohíben notas
cromáticas. El objetivo es que el modelo aprenda cuándo una nota exterior tiene
sentido a partir del corpus, no imponer una máscara rígida de escala.

## Flujo predeterminado

Con `missing_sidecar_policy: infer_source`, la Etapa 1 estima una sola tónica y
un modo a partir de las notas del instrumento normalizado completo. No confía en
el evento `key_signature` del MIDI, porque muchos exportadores escriben un valor
predeterminado que no describe el riff real.

La estimación queda registrada en el manifiesto con su método y dos valores de
confianza. Es una heurística reproducible, no un análisis musical infalible. Los
casos que conozcas con certeza se pueden reemplazar mediante un sidecar manual.

Al aumentar datos, una transposición mueve la tónica por la misma cantidad de
semitonos y conserva el modo. Por ejemplo, `E PHRYGIAN` transpuesto dos
semitonos se etiqueta `F_SHARP PHRYGIAN`. La tonalidad nunca se vuelve a inferir
desde cada variante aumentada.

## Corrección manual opcional

El sidecar debe ser hermano del MIDI y conservar el nombre completo de la
fuente:

```text
riff.mid
riff.mid.tonality.json
```

Su objeto raíz contiene exactamente:

```json
{
  "schema_version": 1,
  "source_midi": "riff.mid",
  "source_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "instrument_index": 0,
  "tonic": "E",
  "mode": "PHRYGIAN"
}
```

Calcula el hash después de la exportación definitiva:

```bash
sha256sum data/raw/riff.mid
```

`instrument_index` es el índice base cero dentro de
`pretty_midi.instruments`, el mismo que selecciona la Etapa 1. Editar o volver
a exportar el MIDI invalida deliberadamente el sidecar y exige actualizar su
hash.

Las tónicas canónicas son:

```text
C C_SHARP D D_SHARP E F F_SHARP G G_SHARP A A_SHARP B UNKNOWN
```

Los modos canónicos son:

```text
MAJOR MINOR DORIAN PHRYGIAN LYDIAN MIXOLYDIAN LOCRIAN
HARMONIC_MINOR PHRYGIAN_DOMINANT BLUES UNKNOWN
```

Usa `UNKNOWN/UNKNOWN` únicamente cuando no quieras afirmar una tonalidad. Una
tónica conocida también puede usar `mode: "UNKNOWN"`; una tónica `UNKNOWN` no
puede acompañarse de un modo conocido. Un sidecar presente pero inválido se
reporta como error de esa fuente: nunca se ignora silenciosamente para volver a
la inferencia automática.

## Revisar las estimaciones

Después de preprocesar, el manifiesto autoritativo contiene la etiqueta de cada
fuente y de cada fragmento. El reporte incluido ordena primero las estimaciones
automáticas con menor confianza de modo:

```bash
python scripts/report_tonalities.py
```

También puede ordenarse por nombre de fuente:

```bash
python scripts/report_tonalities.py --sort source
```

Conviene escuchar y corregir primero las estimaciones de confianza baja y las
muestras cuyo centro tonal conozcas. No es necesario crear sidecars para cada
archivo antes de probar el sistema. Una confianza de modo `0.0000` significa
que al menos dos plantillas compatibles empataron con las notas disponibles;
la etiqueta sigue siendo determinista, pero es una candidata prioritaria para
revisión auditiva.

## Entrenamiento y generación

La migración cambia el vocabulario y los schemas de datos. Hay que ejecutar de
nuevo preprocesamiento y tokenización e iniciar un entrenamiento limpio; un
checkpoint anterior no es compatible con el nuevo manifiesto.

Durante el entrenamiento, `Tonic` y `Mode` permanecen en la entrada recurrente
pero no se cuentan como targets de la loss: son datos proporcionados, no algo
que la GRU deba adivinar desde `BOS`. El primer target optimizado del cuerpo es
`Bar`, y los gradientes musicales posteriores siguen actualizando los
embeddings y el estado asociados al prefijo.

La generación selecciona el contexto explícitamente:

```yaml
conditioning:
  tonic: E
  mode: PHRYGIAN
```

El prefijo se fuerza y se entrega completo a la GRU antes de muestrear el primer
`Bar`. El MIDI exportado no recibe una firma tonal ficticia: la condición exacta
permanece en los tokens y en el manifiesto de generación.
