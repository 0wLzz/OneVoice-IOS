# OneVoiceAI for iPhone

OneVoiceAI translates Vietnamese and Korean speech into text using the bundled models. It runs offline from the first launch and supports iOS 17 or later.

## Run on your iPhone

1. Open `OneVoiceAI.xcodeproj` in Xcode.
2. Select the **OneVoiceAI** target. Under **Signing & Capabilities**, select your Apple development team. If Xcode reports a bundle ID conflict, change `com.onevoiceai.offline` to a unique identifier.
3. Select your connected iPhone as the run destination and press **Run**. The scheme runs a Release build.
4. Choose a direction and tap **Try Sample** to check the bundled pipeline. Tap **Record** to use the microphone. The app stops recording after 30 seconds.

The prepared assets take about 3.4 GiB. Xcode and the iPhone need space for the built app. The app runs one inference stage at a time on the CPU.

After a translation, **Model performance** shows the elapsed time and peak sampled RAM for speech recognition, MiniLM, and M2M. Times include model loading. RAM is the app's physical memory footprint during each stage, sampled every 20 ms.

## Prepare the assets again

From the repository root on an Apple silicon Mac with Xcode and [uv](https://docs.astral.sh/uv/getting-started/installation/) installed:

```sh
./iOS/prepare.sh
```

The first run downloads pinned build dependencies. The model weights come from the local repository. The command verifies the original pipeline files against its manifest, exports Whisper and MiniLM to FP32 ONNX, and stages the existing Zipformer, M2M v8 ONNX INT8, and sample WAVs. It verifies the copied files byte-for-byte and records checksums in `ModelAssets/checksums.json`. Generated files in `ModelAssets` and `Vendor` are already present in this workspace and are excluded from Git.

## Verify the bundled samples

Run **Product → Test** in Xcode on an Apple silicon iPhone simulator. The tests compare the transcript, context tag, and translation for each bundled sample with the Python reference. Both use ONNX Runtime 1.27.0 with graph rewrites disabled for M2M so the INT8 decoding follows the same execution path. To print the reference results after preparing the assets:

```sh
iOS/.prepare/venv/bin/python -B iOS/Tools/reference_samples.py
```
