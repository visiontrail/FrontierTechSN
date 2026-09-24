# Local render media requests

HyperFrames 0.6.99's bundled media servers ignore HTTP Range and send large
videos/audio as chunked 200 responses. Chromium can suspend those transfers
while keeping its per-origin HTTP/1 connections occupied. Extracted-frame
images then remain queued and `img.decode()` never completes. The visible
failure is `Runtime.callFunctionOn timed out`; trailing `ERR_ABORTED` media
messages can hide that cause.

Task `20260924-072036-83ae4b` reproduced this with 14 video elements and a
78 MB program WAV. One worker stalled during image readiness, while the
other stalled at an extracted frame in scene 06. A separate HTTP request
read that image in 4 ms; a media Range request returned 200 instead of 206.

A second, independent wait occurred in pooled Chrome 154 workers on macOS:
the hidden worker page had five completely downloaded images, but every
`img.decode()` promise hung. Bringing that page to the front immediately
resolved decoding. The render subprocess therefore sets
`PRODUCER_ENABLE_BROWSER_POOL=false` so workers own separate browsers. Worker
count, frame rate and output resolution are preserved.

`hyperframe/scripts/patch-media-server.mjs` patches both bundled file servers
to use `media-response.mjs` for audio/video only. Responses stream the requested
bytes with Content-Length, Accept-Ranges and Content-Range; unsatisfiable
ranges return 416. HTML, image injection, rendering quality and review gates
are unchanged. The original installed package version remains in use.

The patch runs after `npm install` in `hyperframe/` and on service startup.
It is idempotent and rejects an unrecognized bundle instead of silently
skipping the repair. Reassess/remove this compatibility patch when upgrading
HyperFrames to a version with equivalent media serving behavior.

Run `npm --prefix hyperframe run test:media-server` for the response tests.
Validate an actual video-heavy render after renderer upgrades as well.
