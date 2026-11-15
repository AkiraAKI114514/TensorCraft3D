r"""Composes the promo video from the recorded shots and title cards (see docs/workbench-guide.md).

    .\.venv\Scripts\python.exe scripts/promo/compose.py <zh|en>

Segments are rendered one by one (trim, speed, framing, per-segment cards), then joined with xfade
transitions; global cards are laid over the joined cut. score.py then synthesizes the soundtrack for
that timeline (it needs numpy, hence the project's Python), a two-pass linear loudnorm brings it to
-16 LUFS, and it is muxed in. Writes artifacts/promo/TensorCraft3D-promo-<lang>-1080p.mp4 and a 720p
preview; intermediate files stay in artifacts/promo/build.
"""
import json, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE.parent.parent / 'artifacts' / 'promo'
BUILD = OUT / 'build'
LANG = sys.argv[1] if len(sys.argv) > 1 else ''
if LANG not in ('zh', 'en'):
    raise SystemExit('usage: python scripts/promo/compose.py <zh|en>')
FPS = 30
CLIPS = BUILD / 'clips' / LANG
TITLES = BUILD / 'titles' / LANG
WORK = BUILD / 'work' / LANG
WORK.mkdir(parents=True, exist_ok=True)
UI_W, UI_H, UI_X, UI_Y = 1440, 810, 240, 222


def ff(args, quiet=True):
    command = ['ffmpeg', '-y', '-hide_banner'] + (['-loglevel', 'error'] if quiet else []) + args
    result = subprocess.run(command)
    if result.returncode:
        raise SystemExit(f'ffmpeg failed: {" ".join(command)[:600]}')


def length(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(path)], capture_output=True, text=True).stdout
    return float(out.strip())


def S(clip, start, dur=None, *, frame='full', speed=1.0, cards=(), trans=('fade', 1 / FPS), tail=0.15):
    """One segment: `dur` seconds of output (None: to the clip's end), starting `start` s into the clip."""
    return dict(clip=clip, start=start, dur=dur, frame=frame, speed=speed, cards=list(cards), trans=trans, tail=tail)


FLASH = ('fadewhite', 0.2)
QUICK = ('fadewhite', 0.12)
DIP = ('fadeblack', 0.6)
SOFT = ('fade', 0.35)

EDL = [
    # Cold open: three macro shots, one statement each, then the title.
    S('open_pulses', 0.4, 3.1, cards=[('k1', 0.15)], trans=FLASH),
    S('open_mla', 0.6, 3.0, cards=[('k2', 0.1)], trans=FLASH),
    S('open_moe', 0.6, 2.9, cards=[('k3', 0.1)], trans=DIP),
    S('hero', 0.3, 5.9, cards=[('title', 0.5)], trans=FLASH),
    # 01 Build
    S('ui_build', 0.35, 7.0, frame='ui', speed=1.4, cards=[('f1', 0.15)], trans=SOFT),
    S('ui_error', 0.35, None, frame='ui', speed=1.2, trans=FLASH),
    # 02 Operators: the counter over a wide shot, then one fly-in per operator.
    S('wide_sequence', 0.4, 3.5, cards=[('ops', 0.15)], trans=QUICK),
    S('parade_vision_c0_0', 0.5, 1.05, cards=[('op_conv2d', 0.0)], trans=QUICK),
    S('parade_vision_c0_3', 0.5, 1.0, cards=[('op_maxpool', 0.0)], trans=QUICK),
    S('parade_vision_c0_10', 0.5, 1.0, cards=[('op_dropout2d', 0.0)], trans=QUICK),
    S('parade_sequence_c0_1', 0.5, 1.0, cards=[('op_transformer', 0.0)], trans=QUICK),
    S('parade_sequence_c0_4', 0.5, 1.0, cards=[('op_mla', 0.0)], trans=QUICK),
    S('parade_sequence_c0_5', 0.5, 1.0, cards=[('op_moe', 0.0)], trans=QUICK),
    S('parade_attention_scores', 0.5, 1.0, cards=[('op_einsum', 0.0)], trans=QUICK),
    S('parade_attention_scale', 0.5, 1.0, cards=[('op_scale', 0.0)], trans=QUICK),
    S('parade_attention_mask', 0.5, 1.0, cards=[('op_mask', 0.0)], trans=QUICK),
    S('parade_attention_softmax', 0.5, 1.0, cards=[('op_softmax', 0.0)], trans=QUICK),
    S('parade_signal_alpha', 0.5, 1.0, cards=[('op_alpha', 0.0)], trans=QUICK),
    S('parade_vision_c0_19', 0.5, 1.05, cards=[('op_logsoftmax', 0.0)], trans=FLASH),
    S('wide_signal', 0.6, 2.6, trans=FLASH),
    # 03 Anatomy
    S('attn_gqa', 0.4, 3.6, cards=[('f3', 0.2)], trans=SOFT),
    S('attn_by_hand', 0.5, 3.4, trans=SOFT),
    S('attn_cross', 0.6, 2.2, trans=FLASH),
    # 04 Import, 05 Train
    S('ui_import', 0.35, None, frame='ui', speed=1.35, cards=[('f4', 0.15)], trans=FLASH),
    S('ui_train', 0.35, None, frame='ui', speed=1.3, cards=[('f5', 0.15)], trans=FLASH),
    # 06 Observe
    S('ui_inspect', 0.35, None, frame='ui', speed=1.4, cards=[('f6', 0.15)], trans=SOFT),
    S('rings', 0.6, 4.0, trans=FLASH),
    S('ui_attention', 0.35, None, frame='ui', speed=1.25, cards=[('f6b', 0.15)], trans=FLASH),
    # 07 Export
    S('ui_export', 0.35, None, frame='ui', speed=1.3, cards=[('f7', 0.15)], trans=DIP),
    # Flashback recap, then the end card.
    S('open_mla', 3.6, 0.5, trans=QUICK),
    S('parade_sequence_c0_5', 1.2, 0.45, trans=QUICK),
    S('rings', 2.4, 0.5, trans=QUICK),
    S('attn_gqa', 3.0, 0.5, trans=QUICK),
    S('parade_attention_softmax', 1.2, 0.5, trans=QUICK),
    S('open_pulses', 2.6, 0.5, trans=FLASH),
    S('outro', 0.3, 6.2, cards=[('end', 0.7)], trans=None),
]
# (card, segment index, offset into it): the recap's statement starts with its six flashbacks, before the end card.
GLOBAL_CARDS = [('k4', len(EDL) - 7, 0.05)]


def card_inputs(cards):
    args, chains = [], []
    for index, (card, offset) in enumerate(cards):
        args += ['-framerate', str(FPS), '-i', str(TITLES / card / '%04d.png')]
        chains.append((index, offset))
    return args, chains


def render_segment(i, seg, background):
    clip = CLIPS / f"{seg['clip']}.mp4"
    available = (length(clip) - seg['start'] - seg['tail']) / seg['speed']
    dur = round(min(seg['dur'], available) if seg['dur'] else available, 3)
    out = WORK / f'seg_{i:02d}.mp4'
    inputs = ['-ss', f"{seg['start']:.3f}", '-i', str(clip)]
    graph = [f"[0:v]setpts=(PTS-STARTPTS)/{seg['speed']},fps={FPS},format=rgba[base0]"]
    base = 'base0'
    extra = 1
    if seg['frame'] == 'ui':
        inputs += ['-stream_loop', '-1', '-i', str(background), '-loop', '1', '-i', str(TITLES / 'mask' / '0001.png'), '-loop', '1', '-i', str(TITLES / 'shadow' / '0001.png')]
        graph = [
            f"[0:v]setpts=(PTS-STARTPTS)/{seg['speed']},fps={FPS},scale={UI_W}:{UI_H}:flags=lanczos,format=rgba[ui]",
            f"[2:v]crop={UI_W}:{UI_H}:0:0,format=rgba,alphaextract[mask]", "[ui][mask]alphamerge[uim]",
            f"[1:v]setpts=PTS-STARTPTS,fps={FPS},format=rgba[bg]", "[bg][3:v]overlay=0:0:format=auto[bgs]",
            f"[bgs][uim]overlay={UI_X}:{UI_Y}:format=auto[base0]",
        ]
        extra = 4
    card_args, chains = card_inputs(seg['cards'])
    inputs += card_args
    for n, (index, offset) in enumerate(chains):
        graph.append(f"[{extra + index}:v]setpts=PTS-STARTPTS+{offset}/TB[c{n}]")
        graph.append(f"[{base}][c{n}]overlay=eof_action=pass:format=auto[base{n + 1}]")
        base = f'base{n + 1}'
    graph.append(f'[{base}]format=yuv420p[out]')
    ff(inputs + ['-filter_complex', ';'.join(graph), '-map', '[out]', '-t', f'{dur:.3f}', '-r', str(FPS), '-c:v', 'libx264', '-preset', 'medium', '-crf', '14', '-an', str(out)])
    return out, dur


def background_loop():
    """The UI segments float over the hero shot, blurred and darkened."""
    out = WORK / 'ui-background.mp4'
    if not out.exists():
        ff(['-i', str(CLIPS / 'open_mla.mp4'), '-vf', 'scale=960:540,gblur=sigma=22,scale=1920:1080:flags=bicubic,eq=brightness=-0.02:saturation=1.35,vignette=PI/5',
            '-c:v', 'libx264', '-crf', '16', '-preset', 'medium', '-an', str(out)])
    return out


def main():
    background = background_loop()
    segments = []
    for i, seg in enumerate(EDL):
        path, dur = render_segment(i, seg, background)
        segments.append((path, dur, seg['trans']))
        print(f'seg {i:02d} {seg["clip"]:<28} {dur:6.2f}s')
    # Join with xfade; record where each segment starts in the cut.
    inputs, graph, starts = [], [], []
    for path, _, _ in segments:
        inputs += ['-i', str(path)]
    total = segments[0][1]
    starts.append(0.0)
    label = '0:v'
    for i in range(1, len(segments)):
        kind, tdur = segments[i - 1][2]
        offset = total - tdur
        starts.append(offset)
        graph.append(f'[{label}][{i}:v]xfade=transition={kind}:duration={tdur:.3f}:offset={offset:.3f}[x{i}]')
        label = f'x{i}'
        total = offset + segments[i][1]
    card_args = []
    for n, (card, index, offset) in enumerate(GLOBAL_CARDS):
        card_args += ['-framerate', str(FPS), '-i', str(TITLES / card / '%04d.png')]
        at = starts[index] + offset
        graph.append(f'[{len(segments) + n}:v]setpts=PTS-STARTPTS+{at:.3f}/TB[g{n}]')
        graph.append(f'[{label}][g{n}]overlay=eof_action=pass:format=auto[go{n}]')
        label = f'go{n}'
    graph.append(f'[{label}]fade=t=in:st=0:d=0.6,fade=t=out:st={total - 1.2:.3f}:d=1.2,format=yuv420p[v]')
    silent = WORK / 'promo-silent.mp4'
    ff(inputs + card_args + ['-filter_complex', ';'.join(graph), '-map', '[v]', '-r', str(FPS), '-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(silent)])
    timeline = {'total': total, 'segments': [{'clip': EDL[i]['clip'], 'start': starts[i], 'dur': segments[i][1], 'trans': segments[i][2]} for i in range(len(segments))]}
    timeline_path = WORK / 'timeline.json'
    timeline_path.write_text(json.dumps(timeline, indent=1), encoding='utf-8')
    print(f'{silent}: {total:.2f} s')
    master(silent, timeline_path)


def master(silent, timeline_path):
    """Score for this timeline, two-pass linear loudnorm to -16 LUFS / -1.5 dBTP, mux, and a 720p preview."""
    wav = WORK / 'score.wav'
    if subprocess.run([sys.executable, str(HERE / 'score.py'), str(timeline_path), str(wav)]).returncode:
        raise SystemExit('score.py failed (it needs numpy: run compose.py with the project Python)')
    target = 'I=-16:TP=-1.5:LRA=11'
    probe = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-i', str(wav), '-af', f'loudnorm={target}:print_format=json', '-f', 'null', '-'], capture_output=True, text=True)
    # loudnorm prints its measurement as a JSON block; ffmpeg's summary lines may follow it.
    blocks = re.findall(r'\{[^{}]*"input_i"[^{}]*\}', probe.stderr)
    if not blocks:
        raise SystemExit('loudnorm printed no measurement:\n' + probe.stderr[-2000:])
    measured = json.loads(blocks[-1])
    loudnorm = (f"loudnorm={target}:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:measured_LRA={measured['input_lra']}"
                f":measured_thresh={measured['input_thresh']}:offset={measured['target_offset']}:linear=true")
    final = OUT / f'TensorCraft3D-promo-{LANG}-1080p.mp4'
    ff(['-i', str(silent), '-i', str(wav), '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-af', f'{loudnorm},aresample=48000',
        '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', str(final)])
    preview = OUT / f'TensorCraft3D-promo-{LANG}-720p-preview.mp4'
    ff(['-i', str(final), '-vf', 'scale=1280:720:flags=lanczos', '-c:v', 'libx264', '-preset', 'slow', '-crf', '25', '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', str(preview)])
    print(f'{final}\n{preview}')


if __name__ == '__main__':
    main()
