# Capture protocol

The application and a capture node talk over one TCP connection. The Pi capture node of
`DESIGN.md` D13 and the stand-in node of D17 speak the same protocol (D18).

## Messages

Each message is a header followed by its payload. Integers are little-endian.

| Field | Type | Contents |
|---|---|---|
| magic | 4 bytes | `SLS1`, which also versions the protocol |
| type | u8 | message type |
| length | u32 | payload length in bytes |

| Type | Name | Direction | Payload |
|---|---|---|---|
| 1 | HELLO | node to application | JSON: node name, projector size and refresh rate, cameras with size and format |
| 2 | SEQUENCE | application to node | JSON: set (`stereo`, `stereo-calibration`, `scan` or `calibration`), frame pairs, trigger rate and light-off spacing for a stereo set, phase pitch, phase steps |
| 3 | FRAME | node to application | frame header, then pixels row by row |
| 4 | END | node to application | JSON: sequence number and frame count |
| 5 | ERROR | node to application | JSON: message |

Frame header, 23 bytes:

| Field | Type | Contents |
|---|---|---|
| sequence | u32 | sequence number on this connection |
| index | u32 | pattern index in projection order |
| exposure | u16 | exposure index |
| timestamp | i64 | exposure start in nanoseconds on the node clock |
| width, height | u16 | image size in pixels |
| format | u8 | 0 for 8-bit monochrome |

## Session

1. On connection, the node sends HELLO.
2. For each SEQUENCE from the application, the node projects every pattern of the set
   in order, captures it, and sends one FRAME per pattern, then END.
3. Further SEQUENCE messages may follow on the same connection, one per board pose
   during calibration.

Clients wait for HELLO before they start a sequence, retrying the connection until it
arrives or their wait runs out. An open port alone does not mean a node is ready: a
container's port forward accepts connections before the node inside listens.

A `stereo` set is a sequence of frame pairs, its count in the request's `frames` field
(one when absent). Each pair is the left then the right frame of one trigger, both with
the trigger's timestamp, saved as `frame_000/left.png` and `frame_000/right.png` onward.
A `stereo-calibration` set is one board pair, saved as `left.png` and `right.png`.

A stereo request may carry `rate_hz`, the trigger rate in pairs per second, with 0 or
absent for as fast as the node can. A node drops a pair whose trigger time passed more
than one period before it can send it, so a client that falls behind sees missing
indices. The stand-in plays a short pre-rendered sweep in a loop at that rate
(`DESIGN.md` D46).

A stereo request may set `dark_every`, with 0 or absent for none. Every that many-th
pair from the first is taken with the pattern light off and saved as `dark_000/left.png`
and `dark_000/right.png` in place of `frame_` (D77).

Pattern order follows the capture layouts of `docs/pipeline.md`. A scan set is the
column Gray codes with their inverses, then the phase shifts (`gray_000` onward, then
`phase_000` onward). A calibration set is white, then column and row Gray codes
(`white`, `columns_000` onward, `rows_000` onward).

## Stand-in node

    slscan node-sim --scene plate                 # serves synthetic frames on port 5600
    slscan capture --set scan --pitch 16 --wait 30 --out data/view
    slscan capture --set calibration --out data/calibration/pose_000
    slscan capture --set stereo-calibration --out data/stereo-calibration/pose_000
    slscan capture --set stereo --frames 25 --out data/sweep
    app/target/debug/sls-scan capture --set scan --pitch 16 --wait 30

The stand-in renders a sweep around a part on a mat for stereo sets, about 2° per frame
at 30 fps, and the next board pose for each calibration set. With light-off pairs its mat
has reference dots, and those pairs are rendered under room light. For phase-shift scan sets it
renders the chosen scene with timestamps at 60 Hz. Captures land in the layouts that
`slscan scan` and `slscan calibrate` read.

The `Dockerfile` at the repository root packages the stand-in, so it runs on its own
address the way the Pi does:

    docker build -t sls-scan-node .
    docker run --rm -p 5600:5600 sls-scan-node --scene sphere

`node.do` builds that image and runs a calibration capture and a plate scan through it
over the protocol, then calibrates, scans and measures the result.
