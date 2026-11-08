# SSTV Decoder

Decodes slow-scan TV (SSTV) pictures from whatever is playing on this computer: a WebSDR in the
browser, an SDR program, a radio on line-in. Nothing has to be routed: it records the output device's
monitor through PipeWire, so you still hear the audio as normal. Pictures are saved as PNG files in
`~/Pictures/SSTV`.

The companion of [Morse Decoder](../morse-decoder), built the same way.

## Modes

Robot 36 and 72, Martin M1 and M2, Scottie S1, S2 and DX, PD 50, 90, 120, 160, 180, 240 and 290,
Wraase SC2-180.

## Install

Needs Ubuntu with PipeWire (`pw-record`) and `python3-numpy python3-scipy python3-pyqt6`.

```bash
scripts/install.sh
```

Then open **SSTV Decoder** from the app menu. `scripts/install.sh --uninstall` removes it (your
pictures stay). From a checkout without installing: `PYTHONPATH=src python3 -m sstv_decoder`.

## Use

1. Tune the WebSDR to the signal and play it: **FM** for VHF/UHF (the ISS on 145.800 MHz), **USB**
   on HF (14.230 MHz is the usual 20 m frequency).
2. The app starts listening to the default output. Pick another output under **Listen to** if the
   audio goes somewhere else.
3. A picture starts on its VIS code (the two-tone header at the start), or, if you tuned in part
   way through, after about 3–12 seconds of its line sync pulses. It builds up line by line. It is
   saved when it's complete or when the signal goes (12 lines without a sync pulse). **End
   picture** saves what has come in so far.

The status line shows the mode, the row, and two measurements:

- **tuning**: how far the leader tone was from 1900 Hz. On SSB this is your tuning error. It's
  corrected for automatically up to ±250 Hz. The marks on the spectrum move with it.
- **clock**: how fast or slow the sender's (or the WebSDR stream's) sample clock runs, in parts per
  million. It's measured from the sync pulses and corrected, so the picture isn't slanted.

Click a picture in the list to look at it, double-click to open it; **Back to live** returns to the
one coming in. **Mode** can be set to one mode, which ignores the others (and picks up pictures of
that mode from sync pulses only).

## How it decodes

1. Audio at 12 kHz mono from `pw-record` (the output's monitor).
2. Mixed down around 1900 Hz and low-pass filtered. The phase change from sample to sample gives the
   instantaneous frequency.
3. VIS: a steady leader (its frequency gives the tuning error), then a 1200 Hz start bit, 7 data bits
   (1100 Hz = 1, 1300 Hz = 0), even parity and a stop bit, all judged relative to the leader. The
   start bit's edge fixes where line 0 begins.
4. Each line's sync pulse is found near where it's expected. A straight line fitted through the
   last 40 pulses gives the line period, so a clock error doesn't slant the picture, and the scan
   timing inside the line is scaled to match. The first lines are redrawn once 10 pulses are in. If
   pulses go missing (an audio dropout), the whole line is searched and decoding carries on from
   the pulse it finds.
5. Each pixel is the mean frequency over its time slot, 1500 Hz black to 2300 Hz white. Robot and PD
   modes send luminance and colour difference (YCbCr). Robot 36 sends one colour difference per
   line (its separator tone says which) and borrows the other from the next line.
6. Without a VIS: pulses of one mode's sync length, spaced by its line period (allowing ±0.5% clock
   error), in the last 12 seconds.

## Tests

```bash
python3 -m pytest tests                               # encode test pictures, decode them back
QT_QPA_PLATFORM=offscreen python3 tests/gui_smoke.py  # the window, fed two pictures
PYTHONPATH=src python3 -m sstv_decoder.synth --mode "PD 120" -o test.wav [picture.png]
```

The tests cover every mode, 0.3% clock error with an 80 Hz mistuning, noise, picking up a picture
30% in without its VIS, an 80 ms audio dropout, two pictures back to back, noise and voice that
must give no picture, and live audio arriving in pieces of any size.

`tests/crosscheck_pysstv.py` decodes audio from [PySSTV](https://github.com/dnet/pySSTV), an
independent encoder, in the 13 modes it shares (see the script for its venv). PySSTV differs from
the specification in two places, which the check allows for: it sends Martin M2 160 pixels wide,
and it leaves out Scottie's start sync.

## Problem reports

**Report a problem…** sends your note, the app's log (`~/.local/state/sstv-decoder/app.log`) and its
settings (mode setting, whether the default output is used, what is being received) to the
developer's server. Crashes and decoder errors are sent the same way. The pictures are never sent.
