---
name: Camera or phone support
about: Footage from a device Reelsift doesn't handle well yet
labels: device
---

**Device**

Make, model and firmware, and the recording settings (resolution, frame rate, HDR or log, GPS on or off).

**What goes wrong**

For example: long recordings aren't joined, times are off, proxies aren't used, colors look wrong.

**File details**

For one file, the output of:

```
ffprobe -v error -show_format -show_streams FILE
```

and the names of the files the device writes next to it.

**Sample footage**

Can you share a few seconds of untouched footage you shot yourself, to be published in the test suite under the project's license? Say so here, and we'll agree on how to send it. Please don't attach it to the issue.
