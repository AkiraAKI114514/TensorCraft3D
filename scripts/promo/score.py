"""Synthesized score timed to a promo cut: python score.py <timeline.json> <out.wav> (compose.py runs it; needs numpy)

Pad chords per section, a low pulse under the feature montage, impacts on flash cuts, ticks on the
operator cuts, risers into the title and the end card, and a resolving chord at the end. Everything is
synthesized here; nothing is sampled.
"""
import json, sys, wave
import numpy as np

SR = 48000
timeline = json.loads(open(sys.argv[1], encoding='utf-8').read())
TOTAL = timeline['total'] + 0.05
N = int(TOTAL * SR)
rng = np.random.default_rng(7)
L = np.zeros(N); R = np.zeros(N)
segs = timeline['segments']
start = {i: s['start'] for i, s in enumerate(segs)}
clip_at = [s['clip'] for s in segs]


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def env(n, attack, release, sustain_end=None):
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-3), 0, 1)
    end = n / SR
    r = np.clip((end - t) / max(release, 1e-3), 0, 1)
    return np.sin(a * np.pi / 2) ** 2 * np.sin(r * np.pi / 2) ** 2


def add(sig, at, gain=1.0, pan=0.0):
    i = int(at * SR)
    if i >= N:
        return
    sig = sig[: N - i]
    left, right = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    L[i:i + len(sig)] += sig * gain * left
    R[i:i + len(sig)] += sig * gain * right


def pad_note(midi, dur, detune=0.0):
    n = int(dur * SR); t = np.arange(n) / SR
    f = hz(midi) * 2 ** (detune / 1200)
    wobble = 1 + 0.0015 * np.sin(2 * np.pi * 0.23 * t + midi)
    out = np.zeros(n)
    for k in range(1, 9):  # soft saw: harmonics rolled off hard, like a low-passed pad
        out += np.sin(2 * np.pi * f * k * wobble * t + k * 0.7) * (1 / k) * np.exp(-0.42 * (k - 1))
    return out * env(n, 1.6, 2.2)


def chord(notes, at, dur, gain):
    for j, m in enumerate(notes):
        pan = (j / max(1, len(notes) - 1)) * 1.2 - 0.6
        add(pad_note(m, dur, -5), at, gain / len(notes), pan)
        add(pad_note(m, dur, +5), at, gain / len(notes), -pan)


def impact(at, gain=1.0):
    n = int(1.4 * SR); t = np.arange(n) / SR
    f = 34 + 70 * np.exp(-t * 9)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 3.2)
    noise = rng.standard_normal(n) * np.exp(-t * 18)
    noise = np.convolve(noise, np.ones(24) / 24, mode='same')  # dull the burst
    add(body * 0.9 + noise * 0.35, at, gain)


def tick(at, gain=0.25, pitch=1760):
    n = int(0.09 * SR); t = np.arange(n) / SR
    s = np.sin(2 * np.pi * pitch * t) * np.exp(-t * 60) + np.sin(2 * np.pi * pitch * 1.5 * t) * np.exp(-t * 80) * 0.4
    add(s, at, gain, pan=rng.uniform(-0.4, 0.4))


def riser(end, dur, gain=0.35):
    n = int(dur * SR); t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    out = np.zeros(n); y = 0.0
    cutoff = 0.004 + 0.12 * (t / dur) ** 2  # one-pole low-pass sweeping open
    for i in range(n):
        y += cutoff[i] * (noise[i] - y); out[i] = y
    out *= (t / dur) ** 1.6 * 3.0
    add(out, end - dur, gain)


def pulse(at, gain=0.5):
    n = int(0.42 * SR); t = np.arange(n) / SR
    f = 48 + 40 * np.exp(-t * 30)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7)
    add(s, at, gain)


# Sections by segment index (see compose.py's EDL).
first_ui = next(i for i, c in enumerate(clip_at) if c.startswith('ui_'))
hero = clip_at.index('hero')
recap = len(segs) - 7
outro = len(segs) - 1
A3, C4, D4, E4, F3, G3 = 57, 60, 62, 64, 53, 55
progression = [  # (from segment, notes)
    (0, [45, 52, 57, 59, 64]),                          # A minor add9, low and open
    (hero, [41, 48, 57, 60, 64]),                       # F maj7: the title lifts
    (first_ui, [48, 55, 60, 64, 67]),                   # C (build)
    (clip_at.index('wide_sequence'), [43, 50, 59, 62, 67]),  # G (operators)
    (clip_at.index('attn_gqa'), [45, 52, 60, 64, 69]),  # Am (anatomy)
    (clip_at.index('ui_import'), [41, 48, 57, 60, 65]), # F (import, train)
    (clip_at.index('ui_inspect'), [48, 55, 64, 67, 72]),  # C (observe, export)
    (recap, [43, 50, 59, 62, 67]),                      # G (recap tension)
    (outro, [41, 48, 57, 60, 64, 69]),                  # F add9 under the end card
]
for k, (seg, notes) in enumerate(progression):
    at = start[seg]
    end = start[progression[k + 1][0]] if k + 1 < len(progression) else TOTAL
    chord(notes, max(0, at - 0.25), end - at + 2.0 if k + 1 < len(progression) else TOTAL - at, 0.2 if seg < hero else (0.62 if seg in (hero, outro) else 0.42))
add(pad_note(29, TOTAL - start[outro]) * 0.3, start[outro])  # low F root under the end card

# Risers into the title and the end card; impacts on flash and dip cuts; ticks on operator cuts.
riser(start[hero] + 0.05, 1.6, 0.3)
riser(start[outro] + 0.05, 1.4, 0.3)
impact(start[hero], 1.0)
impact(start[outro], 0.9)
for i in range(1, len(segs)):
    kind, d = segs[i - 1]['trans'] or ('fade', 0)
    at = segs[i]['start'] + (d or 0) / 2
    if clip_at[i].startswith('parade_') or (recap <= i < outro):
        tick(at, 0.22, 1760 if i % 2 else 1318)
        if recap <= i < outro:
            impact(at, 0.45)
    elif kind in ('fadewhite', 'fadeblack') and i not in (hero, outro):
        impact(at, 0.22 if i < hero else 0.55)

# A driving pulse at 120 BPM from the first feature to the recap.
t = start[first_ui]
while t < start[recap] - 0.3:
    pulse(t, 0.38)
    t += 0.5

# Space: a short synthetic reverb (decaying noise impulse) on the whole mix.
def reverb(x, seconds=2.2, mix=0.22):
    n = int(seconds * SR); t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-t * 3.0); ir[0] = 0; ir /= np.sqrt(np.sum(ir ** 2))
    size = 1 << int(np.ceil(np.log2(len(x) + n)))
    wet = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[: len(x)]
    return x * (1 - mix) + wet * mix

L, R = reverb(L), reverb(R)
mix = np.stack([L, R], axis=1)
fade = int(1.5 * SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None] ** 2
fade_in = int(0.4 * SR); mix[:fade_in] *= np.linspace(0, 1, fade_in)[:, None]
mix *= 0.89 / (np.max(np.abs(mix)) + 1e-9)  # headroom only; loudness is set by a linear two-pass loudnorm at mux time
rms = 20 * np.log10(np.sqrt(np.mean(mix ** 2)) + 1e-12)
pcm = (mix * 32767).astype('<i2')
with wave.open(sys.argv[2], 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print(f'{sys.argv[2]}: {TOTAL:.2f} s, RMS {rms:.1f} dBFS before normalisation')
