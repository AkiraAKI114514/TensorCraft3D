"""The promo's soundtrack, synthesized on the cut's own tempo grid: python score.py <timeline.json> <out.wav>

compose.py runs it with the project Python (it needs numpy) and puts every cut on a beat of the
timeline's `bpm`, so the music and the picture share one grid:

- cold open: an open A minor pad, a low drone and a soft bell on each cut, no drums;
- title: an impact, then F, C and G with the bass on the roots;
- features: Am-F-C-G a bar each with a groove (kick, clap on 2 and 4, hats), a bass on eighths and
  plucked arpeggios; the operator parade doubles the arpeggio and hats and rings a rising bell on
  each operator; the attention anatomy drops to a breakdown without kick or clap;
- recap: G under a snare roll and a riser, a kick on each recap shot;
- end card: an impact, F then C, ringing out.

Instruments are wavetable oscillators (band-limited by construction), noise shaped with FFT
filters, a kick-driven duck on pad, bass and arpeggio, and a three-band hall reverb. Nothing is
sampled.
"""
import json, sys, wave
import numpy as np

SR = 48000
rng = np.random.default_rng(11)
timeline = json.loads(open(sys.argv[1], encoding='utf-8').read())
TOTAL = timeline['total'] + 0.1
N = int(TOTAL * SR)
BEAT = 60 / timeline.get('bpm', 120)
BAR = BEAT * 4
segments = timeline['segments']
clips = [s['clip'] for s in segments]
cut = [s.get('cut', s['start']) for s in segments]

HERO, GROOVE = clips.index('hero'), next(i for i, c in enumerate(clips) if c.startswith('ui_'))
PARADE = clips.index('wide_sequence')
PARADE_END = next(i for i in range(PARADE + 1, len(clips)) if not clips[i].startswith('parade_'))
BREAK, BREAK_END = clips.index('attn_gqa'), clips.index('ui_import')
RECAP, OUTRO = len(segments) - 7, len(segments) - 1
SECTION_STARTS = {'ui_error', 'wide_sequence', 'wide_signal', 'attn_gqa', 'ui_import', 'ui_train', 'ui_inspect', 'ui_export'}
T = {name: cut[index] for name, index in dict(hero=HERO, groove=GROOVE, parade=PARADE, parade_end=PARADE_END, breakdown=BREAK, breakdown_end=BREAK_END, recap=RECAP, outro=OUTRO).items()}

STEMS = ('pad', 'bass', 'arp', 'bell', 'kick', 'drums', 'fx')
stem = {name: np.zeros((N, 2)) for name in STEMS}


def place(name, signal, at, gain=1.0, pan=0.0):
    """Adds a mono (panned, constant power) or stereo signal to a stem at `at` seconds."""
    i = int(round(at * SR))
    if i >= N or gain == 0:
        return
    if signal.ndim == 1:
        signal = np.stack([signal * np.cos((pan + 1) * np.pi / 4), signal * np.sin((pan + 1) * np.pi / 4)], 1) * np.sqrt(2)
    if i < 0:
        signal, i = signal[-i:], 0
    end = min(N, i + len(signal))
    stem[name][i:end] += signal[:end - i] * gain


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def filtered(x, low=None, high=None, order=2):
    """Zero-phase Butterworth-shaped band limits by FFT (mono or stereo)."""
    n = len(x)
    size = 1 << int(np.ceil(np.log2(n * 2)))
    f = np.fft.rfftfreq(size, 1 / SR)
    h = np.ones_like(f)
    if high:
        h /= np.sqrt(1 + (f / high) ** (2 * order))
    if low:
        h /= np.sqrt(1 + (low / np.maximum(f, 1e-3)) ** (2 * order))
    spectrum = np.fft.rfft(x, size, axis=0)
    return np.fft.irfft(spectrum * (h[:, None] if x.ndim == 2 else h), size, axis=0)[:n]


def table_for(f0, cutoff, shape='saw'):
    """One band-limited cycle: saw (all harmonics) or square-ish (odd ones), rolled off at `cutoff`."""
    count = int(min(64, (SR * 0.45) // f0))
    k = np.arange(1, count + 1)
    amp = (1 / k) / np.sqrt(1 + (k * f0 / cutoff) ** 4)
    if shape == 'square':
        amp[1::2] = 0
    x = np.arange(4096) / 4096
    return (amp[:, None] * np.sin(2 * np.pi * k[:, None] * x[None, :] + rng.uniform(0, 2 * np.pi, count)[:, None])).sum(0)


def oscillate(table, f0, n, vibrato=0.0):
    t = np.arange(n) / SR
    phase = f0 * t + rng.uniform()
    if vibrato:
        phase += vibrato * np.sin(2 * np.pi * 4.8 * t) / (2 * np.pi * 4.8) * f0 * 0.002
    index = (phase % 1) * 4096
    i0 = index.astype(int)
    frac = index - i0
    return table[i0] * (1 - frac) + table[(i0 + 1) % 4096] * frac


def envelope(n, attack, release, sustain=1.0, decay=0.0):
    t = np.arange(n) / SR
    dur = n / SR
    rise = np.sin(np.clip(t / max(attack, 1e-4), 0, 1) * np.pi / 2) ** 2
    body = sustain + (1 - sustain) * np.exp(-t / decay) if decay else np.ones(n)
    fall = np.sin(np.clip((dur - t) / max(release, 1e-4), 0, 1) * np.pi / 2) ** 2
    return rise * body * fall


# ---- instruments ------------------------------------------------------------------------------

def pad_chord(notes, at, dur, gain):
    n = int((dur + 1.4) * SR)
    for j, m in enumerate(notes):
        table = table_for(hz(m), 950)
        for detune, pan in ((-7, -0.55), (0, 0.0), (7, 0.55)):
            voice = oscillate(table, hz(m) * 2 ** (detune / 1200), n) * envelope(n, 0.9, 1.4)
            place('pad', voice, at, gain / len(notes) / 3, pan * (0.4 + 0.6 * j / max(1, len(notes) - 1)))


def bass_note(m, at, dur, gain):
    n = int(dur * SR)
    tone = oscillate(table_for(hz(m), 380), hz(m), n) * 0.7 + np.sin(2 * np.pi * hz(m) * np.arange(n) / SR) * 0.6
    place('bass', tone * envelope(n, 0.006, 0.04, sustain=0.65, decay=0.18), at, gain)


def pluck(m, at, gain, pan=0.0, length=0.45):
    n = int(length * SR)
    t = np.arange(n) / SR
    bright, dark = oscillate(table_for(hz(m), 3200), hz(m), n), oscillate(table_for(hz(m), 700), hz(m), n)
    sweep = np.exp(-t * 14)
    place('arp', (bright * sweep + dark * (1 - sweep)) * envelope(n, 0.003, 0.08, sustain=0.0, decay=0.16), at, gain, pan)


def bell(m, at, gain, pan=0.0, length=2.4):
    n = int(length * SR)
    t = np.arange(n) / SR
    f = hz(m)
    tone = np.sin(2 * np.pi * f * t) + 0.45 * np.sin(2 * np.pi * f * 2.0 * t) * np.exp(-t * 3) + 0.2 * np.sin(2 * np.pi * f * 3.01 * t) * np.exp(-t * 6)
    place('bell', tone * envelope(n, 0.004, 0.3, sustain=0.0, decay=0.9), at, gain, pan)


KICKS = []


def kick(at, gain=1.0):
    if gain <= 0:
        return
    KICKS.append(at)
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    body = np.sin(2 * np.pi * np.cumsum(46 + 95 * np.exp(-t * 32)) / SR) * np.exp(-t * 6.5)
    click = filtered(rng.standard_normal(n) * np.exp(-t * 900), low=2500)
    place('kick', body + click * 0.25, at, gain)


def noise_hit(name, at, gain, low=None, high=None, decay=0.1, length=0.4, pan=0.0, bursts=1):
    n = int(length * SR)
    t = np.arange(n) / SR
    noise = filtered(rng.standard_normal(n), low=low, high=high)
    shape = np.exp(-t / decay)
    if bursts > 1:  # a clap: a few quick bursts before the tail
        shape = np.maximum(shape * 0.8, sum(np.exp(-np.clip(t - b * 0.011, 0, None) / 0.004) * (t >= b * 0.011) for b in range(bursts)))
    place(name, noise * shape, at, gain, pan)


def clap(at, gain=0.5):
    noise_hit('drums', at, gain, low=900, high=5200, decay=0.11, length=0.45, bursts=3)


def hat(at, gain=0.12, open_=False):
    noise_hit('drums', at, gain, low=7000, decay=0.18 if open_ else 0.03, length=0.4 if open_ else 0.08, pan=rng.uniform(-0.3, 0.3))


def snare(at, gain=0.4):
    noise_hit('drums', at, gain, low=1400, high=7500, decay=0.09, length=0.3)
    n = int(0.15 * SR)
    t = np.arange(n) / SR
    place('drums', np.sin(2 * np.pi * 190 * t) * np.exp(-t * 30), at, gain * 0.5)


def crash(at, gain=0.35):
    n = int(2.6 * SR)
    t = np.arange(n) / SR
    noise = filtered(rng.standard_normal((n, 2)), low=3500, high=13000)
    place('fx', noise * (np.exp(-t / 0.9) * envelope(n, 0.002, 0.4))[:, None], at, gain)


def impact(at, gain=1.0):
    n = int(2.8 * SR)
    t = np.arange(n) / SR
    sub = np.sin(2 * np.pi * np.cumsum(30 + 30 * np.exp(-t * 5)) / SR) * np.exp(-t * 1.6)
    thump = filtered(rng.standard_normal(n), high=380) * np.exp(-t * 9)
    place('fx', sub * 0.9 + thump * 0.6, at, gain)


def riser(end, dur, gain=0.3):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    steps = 6
    for s in range(steps):  # noise bands opening upward, crossfaded
        band = filtered(rng.standard_normal(n), low=300 * 2 ** s, high=900 * 2 ** s)
        weight = np.clip(1 - np.abs(t / dur * (steps - 1) - s), 0, 1)
        out += band * weight
    place('fx', out * (t / dur) ** 2.2, end - dur, gain)


# ---- harmony ----------------------------------------------------------------------------------

AM_OPEN = [45, 52, 57, 60, 64, 71]
CHORDS = {'Am': [45, 52, 60, 64, 69], 'F': [41, 48, 57, 60, 65], 'C': [48, 55, 60, 64, 67], 'G': [43, 50, 59, 62, 67],
          'Fadd9': [41, 48, 57, 60, 64, 67], 'Cend': [36, 48, 55, 60, 64, 67], 'Gsus': [43, 50, 60, 62, 67]}
ROOTS = {'Am': 33, 'F': 29, 'C': 36, 'G': 31, 'Fadd9': 29, 'Cend': 36, 'Gsus': 31}
LOOP = ['Am', 'F', 'C', 'G']

plan = [('Am-open', 0.0, T['hero'])]
title_beats = round((T['groove'] - T['hero']) / BEAT)
at = T['hero']
for name, beats in (('F', 4), ('C', 4), ('G', max(1, title_beats - 8))):
    plan.append((name, at, at + beats * BEAT)); at += beats * BEAT
bar = 0
at = T['groove']
while at < T['recap'] - 1e-6:
    end = min(at + BAR, T['recap'])
    plan.append((LOOP[bar % 4], at, end)); at, bar = end, bar + 1
plan += [('Gsus', T['recap'], T['recap'] + BEAT * 2), ('G', T['recap'] + BEAT * 2, T['outro']), ('Fadd9', T['outro'], T['outro'] + BAR), ('Cend', T['outro'] + BAR, TOTAL)]


def chord_at(t):
    for name, start, end in plan:
        if start <= t < end:
            return name
    return plan[-1][0]


for name, start, end in plan:
    notes = AM_OPEN if name == 'Am-open' else CHORDS[name]
    level = 0.45 if name == 'Am-open' else 0.85 if start >= T['outro'] else 0.62
    pad_chord(notes, start - 0.05, end - start, level)
    if name == 'Am-open':
        place('pad', np.sin(2 * np.pi * hz(33) * np.arange(int((end + 1) * SR)) / SR) * envelope(int((end + 1) * SR), 2.0, 1.5), 0.0, 0.18)

# ---- rhythm and parts, beat by beat -------------------------------------------------------------

def section(t):
    if t < T['hero']: return 'open'
    if t < T['groove']: return 'title'
    if T['parade'] <= t < T['breakdown']: return 'parade'
    if T['breakdown'] <= t < T['breakdown_end']: return 'breakdown'
    if t < T['recap']: return 'groove'
    if t < T['outro']: return 'recap'
    return 'end'


beat_count = int(TOTAL / BEAT) + 1
for b in range(beat_count):
    t = b * BEAT  # every cut sits on this grid
    if t >= TOTAL:
        continue
    where, name = section(t), chord_at(t)
    in_bar = int(round((t - T['groove']) / BEAT)) % 4
    chord = CHORDS.get(name, AM_OPEN)
    tones = sorted(m for m in chord if m >= 57) + [m + 12 for m in sorted(m for m in chord if m >= 57)][:2]
    if where in ('groove', 'parade'):
        kick(t, 0.9 if in_bar in (0, 2) else (0.75 if where == 'parade' else 0))
        if in_bar in (1, 3):
            clap(t, 0.42)
        for half in (0.0, 0.5) if where == 'groove' else (0.0, 0.25, 0.5, 0.75):
            hat(t + half * BEAT, 0.07 if half == 0.0 else 0.11, open_=(where == 'groove' and half == 0.5 and in_bar == 3))
        for half in (0.0, 0.5):
            bass_note(ROOTS[name] + (12 if half and in_bar % 2 else 0), t + half * BEAT, BEAT * 0.48, 0.55)
        steps = 4 if where == 'parade' else 2
        for s in range(steps):
            pluck(tones[(in_bar * steps + s) % len(tones)], t + s * BEAT / steps, 0.30 if where == 'parade' else 0.26, pan=(-0.35, 0.35)[s % 2])
    elif where == 'breakdown':
        for half in (0.0, 0.5):
            hat(t + half * BEAT, 0.05)
            pluck(tones[(in_bar * 2 + int(half * 2)) % len(tones)], t + half * BEAT, 0.22, pan=(-0.4, 0.4)[int(half * 2)])
        if in_bar == 0:
            bass_note(ROOTS[name], t, BAR * 0.95, 0.45)
    elif where == 'title':
        if in_bar == 0:
            bass_note(ROOTS[name], t, BAR * 0.95, 0.5)
            kick(t, 0.55)
        if in_bar == 2:
            kick(t, 0.35)
    elif where == 'recap':
        kick(t, 0.85)
        bass_note(ROOTS['G'], t, BEAT * 0.9, 0.5)

# A short pickup into the groove, and drums back after the breakdown on a fill.
for i in range(4):
    snare(T['groove'] - BEAT + i * BEAT / 4, 0.12 + 0.07 * i)
for i in range(8):
    snare(T['breakdown_end'] - BEAT * 2 + i * BEAT / 4, 0.08 + 0.05 * i)
riser(T['breakdown_end'], BAR, 0.18)

# Bells: one on each cold-open statement, a rising one on each operator, and the end card.
OPEN_BELLS = [76, 79, 83]
for k, i in enumerate(range(0, HERO)):
    bell(OPEN_BELLS[k % 3], cut[i] + 0.02, 0.32, pan=(-0.3, 0.3, 0.0)[k % 3])
parade_cuts = [cut[i] for i in range(PARADE + 1, PARADE_END)]
RISING = [64, 67, 69, 72, 74, 76, 79, 81, 84, 86, 88, 91]  # A minor pentatonic, up an octave and a half
for k, at_ in enumerate(parade_cuts):
    bell(RISING[min(k, len(RISING) - 1)], at_, 0.22, pan=(-0.25, 0.25)[k % 2], length=1.4)
for m, delay in ((72, 0.0), (76, 0.5), (79, 1.0), (84, 1.5)):
    bell(m, T['outro'] + BAR + delay, 0.24, length=3.0)

# Accents on the edit: impacts and crashes where sections change, a roll and a riser into the end card.
impact(T['hero'], 1.0); crash(T['hero'], 0.4); riser(T['hero'], 1.8, 0.22)
crash(T['groove'], 0.3)
for i in range(1, len(segments)):
    kind = (segments[i - 1].get('trans') or ['fade', 0])[0]
    if i in (HERO, GROOVE, OUTRO) or RECAP < i < OUTRO or PARADE < i < PARADE_END:
        continue
    # A section's first shot gets a cymbal and a soft low hit; dips to black too.
    if kind == 'fadeblack' or clips[i] in SECTION_STARTS:
        crash(cut[i], 0.2); impact(cut[i], 0.28)
roll = int((T['outro'] - T['recap']) / (BEAT / 4))
for i in range(roll):
    snare(T['recap'] + i * BEAT / 4, 0.06 + 0.32 * (i / max(1, roll - 1)) ** 1.6)
riser(T['outro'], T['outro'] - T['recap'], 0.32)
impact(T['outro'], 1.1); crash(T['outro'], 0.45)

# ---- mix ---------------------------------------------------------------------------------------

# The kick ducks the pad, bass and arpeggio, so the groove breathes.
duck = np.ones(N)
length = int(0.4 * SR)
curve = 1 - 0.42 * np.exp(-np.arange(length) / SR / 0.11)
for at_ in KICKS:
    i = int(round(at_ * SR))
    end = min(N, i + length)
    duck[i:end] = np.minimum(duck[i:end], curve[:end - i])
for name in ('pad', 'bass', 'arp'):
    stem[name] *= duck[:, None]

GAINS = {'pad': 0.55, 'bass': 0.5, 'arp': 0.30, 'bell': 0.42, 'kick': 0.85, 'drums': 0.75, 'fx': 0.8}
dry = sum(stem[name] * GAINS[name] for name in STEMS)
send = stem['pad'] * 0.35 + stem['arp'] * 0.55 + stem['bell'] * 0.7 + stem['drums'] * 0.25 + stem['fx'] * 0.3


def hall(seconds=2.6):
    """A hall's impulse: early reflections, then a tail that dies faster in the highs than the lows."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    tail = np.zeros((n, 2))
    for low, high, decay in ((None, 300, 0.75), (300, 3000, 0.62), (3000, None, 0.28)):
        tail += filtered(rng.standard_normal((n, 2)), low=low, high=high) * np.exp(-t / decay)[:, None]
    tail *= np.clip(t / 0.03, 0, 1)[:, None]  # pre-delay
    for k, (delay, level) in enumerate(((0.011, 0.5), (0.019, 0.35), (0.027, 0.3), (0.041, 0.22))):
        tail[int(delay * SR)] += level * np.array([1.0, 0.8] if k % 2 else [0.8, 1.0])
    return tail / np.sqrt(np.sum(tail ** 2) / 2)


ir = hall()
size = 1 << int(np.ceil(np.log2(N + len(ir))))
wet = np.stack([np.fft.irfft(np.fft.rfft(send[:, c], size) * np.fft.rfft(ir[:, c], size), size)[:N] for c in range(2)], 1)
mix = filtered(dry + wet * 0.9, low=30)

# Gentle block limiter: no transient gets more than ~10 dB above the body before loudness normalisation.
block = 256
blocks = int(np.ceil(N / block))
padded = np.pad(np.abs(mix).max(1), (0, blocks * block - N))
peaks = padded.reshape(blocks, block).max(1)
target = np.percentile(peaks[peaks > 1e-4], 90) * 1.6 if np.any(peaks > 1e-4) else 1.0
gain = np.minimum(1.0, target / np.maximum(peaks, 1e-9))
smooth = np.empty_like(gain)
level = 1.0
for i, g in enumerate(gain):
    level = g if g < level else level + (g - level) * 0.02  # instant attack, slow release
    smooth[i] = level
mix *= np.interp(np.arange(N), np.arange(blocks) * block + block / 2, smooth)[:, None]

fade_in, fade_out = int(0.3 * SR), int(1.6 * SR)
mix[:fade_in] *= np.linspace(0, 1, fade_in)[:, None]
mix[-fade_out:] *= (np.linspace(1, 0, fade_out) ** 2)[:, None]
mix *= 0.89 / (np.max(np.abs(mix)) + 1e-9)
rms = 20 * np.log10(np.sqrt(np.mean(mix ** 2)) + 1e-12)
with wave.open(sys.argv[2], 'wb') as out:
    out.setnchannels(2); out.setsampwidth(2); out.setframerate(SR)
    out.writeframes((mix * 32767).astype('<i2').tobytes())
print(f'{sys.argv[2]}: {TOTAL:.2f} s at {60 / BEAT:.0f} BPM, RMS {rms:.1f} dBFS before normalisation')
