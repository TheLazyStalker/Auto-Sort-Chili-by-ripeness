# Tích hợp Android — chili-ripeness-ai (MobileNetV2, 3 lớp ớt)

> **Lưu ý tương thích MediaPipe:** file `.tflite` thô (export từ Keras) **không tự tương thích** MediaPipe Tasks Vision Image Classifier. MediaPipe yêu cầu metadata (labels + tensor associates) nhúng trong file qua **Model Metadata Writer** (công cụ Task Library). Hướng dẫn: `https://ai.google.dev/mediapipe/solutions/vision/image_classifier/android` và `https://ai.google.dev/mediapipe/tasks-vision` (metadata writer: `org.tensorflow:tensorflow-lite-metadata`). Nếu chưa nhúng metadata, hãy dùng TensorFlow Lite Task Library (`ImageClassifier`) hoặc Interpreter trực tiếp — vẫn chạy được mà không cần MediaPipe.

## 1. Model được khuyến nghị

| Model | Khi nào dùng | Dung lượng (ước tính từ MobileNetV2 ~2.26M params) |
|---|---|---|
| `chili_ripeness.tflite` (Float32) | Fallback an toàn, mọi thiết bị, kiểm tra lần đầu | ~8.6 MB |
| `chili_ripeness_float16.tflite` (Float16) | Thiết bị có GPU delegate (MediaPipe GPU) — nhanh hơn, nhẹ hơn | ~4.5 MB |
| `chili_ripeness_int8.tflite` (INT8) | Thiết bị yếu, cần tốc độ nhất — **bắt buộc calibration bằng ảnh train thật** | ~2.3 MB |

**Khuyến nghị mặc định:** bắt đầu với **Float32** để kiểm tra correctness trên máy thật (đầu ra khớp Keras), sau đó chuyển Float16 trên thiết bị có GPU. INT8 chỉ dùng khi đã đo độ lệch không đáng kể trên tập test độc lập.

## 2. Input / Output

- **Input:** RGB, kích thước **224×224**, 3 kênh. Tiền xử lý: `Resize(224×224)` giữ pixel **uint8 RGB [0,255]**; model MobileNetV2 **tự** `preprocess_input` map `[0,255] → [-1,1]` bên trong. **KHÔNG được `Rescaling(1/255)`** (sẽ double-scaling, làm sai inference). Với INT8, input là uint8 `[0,255]`.
- **Output:** softmax 3 giá trị, `labels.txt`:

  ```
  unripe
  medium_ripe
  ripe
  ```

  Thứ tự cố định: `0=unripe, 1=medium_ripe, 2=ripe`. Confidence = max(output).
- **Metadata:** `model_metadata.json` mô tả input/output (tạo tự động khi export). Luôn kiểm tra `export_report.json` trong cùng thư mục: `input_shape` phải là `[1,224,224,3]`, `output_is_3_classes=true`.

## 3. Dependency Android (Kotlin, TFLite Task Library)

```kotlin
// app/build.gradle.kts
dependencies {
    implementation("org.tensorflow:tensorflow-lite:2.17.0")           // Interpreter fallback
    implementation("com.google.android.gms:play-services-tflite-java:21.0.0") // optional
    // MediaPipe Tasks Vision (cần metadata, xem lưu ý trên):
    // implementation("com.google.mediapipe:tasks-vision:0.10.x")  // phiên bản mới nhất
}
```

Đặt file model vào `app/src/main/assets/` (vd. `chili_ripeness.tflite`). Đặt `labels.txt` vào assets để load labels.

## 4. Ví dụ inference (TFLite Interpreter — không phụ thuộc MediaPipe)

```kotlin
import android.graphics.Bitmap
import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.channels.FileChannel

// Load model
fun loadModel(context: Context, asset: String = "chili_ripeness.tflite"): Interpreter {
    val fd = context.assets.openFd(asset)
    val channel: FileChannel = fd.createInputStream().channel
    val mapped = channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    return Interpreter(mapped)
}

// Tiền xử lý khớp pipeline train: resize 224x224, giữ float32 [0,255]
// (model tự preprocess_input [0,255]->[-1,1]; KHÔNG chia 255).
fun preprocess(input: Bitmap): Array<Array<Array<FloatArray>>> {
    val bmp = Bitmap.createScaledBitmap(input, 224, 224, true)
    val out = Array(1) { Array(224) { Array(224) { FloatArray(3) } } }
    for (y in 0 until 224) {
        for (x in 0 until 224) {
            val px = bmp.getPixel(x, y)
            out[0][y][x][0] = android.graphics.Color.red(px).toFloat()      // R in [0,255]
            out[0][y][x][1] = android.graphics.Color.green(px).toFloat()    // G in [0,255]
            out[0][y][x][2] = android.graphics.Color.blue(px).toFloat()     // B in [0,255]
        }
    }
    return out
}

fun predict(interpreter: Interpreter, input: Bitmap): Pair<String, Map<String, Float>> {
    val labels = listOf("unripe", "medium_ripe", "ripe")
    val inputArr = preprocess(input)
    val output = Array(1) { FloatArray(3) }
    interpreter.run(inputArr, output)
    val bestIdx = output[0].indices.maxByOrNull { output[0][it] } ?: 0
    val probs = labels.associateWith { output[0][labels.indexOf(it)] }
    return labels[bestIdx] to probs
}

// Xử lý confidence thấp
val (label, probs) = predict(interpreter, bitmap)
val confidence = probs[label] ?: 0f
if (confidence < 0.6f) {
    // Gửi trạng thái "uncertain" cho ESP32-S3 thay vì hành động theo label chưa chắc
    // vd. gửi '{"status":"uncertain","confidence":0.5}' qua UART/BLE
} else {
    // Gửi label/confidence cho ESP32
}
```

MediaPipe Tasks Vision (khi đã có metadata):
```kotlin
// Pseudo-code qua BaseOptions + ImageClassifier:
// val options = ImageClassifier.ImageClassifierOptions.builder()
//     .setMaxResults(3).setScoreThreshold(0.6f).build()
// val classifier = ImageClassifier.createFromOptions(context, baseOptions, options)
// classifier.classify(mpImage) → Category list (label + score)
```

## 5. Xử lý confidence thấp

- Ngưỡng mặc định trong `config.yaml`: `inference.confidence_threshold = 0.6`. Dưới ngưỡng → trạng thái `uncertain` (predict.py in `"status": "uncertain"`). Android nên **không** phát hành động cho ESP32-S3 dựa trên kết quả uncertain; thay vào đó chụp lại / yêu cầu người dùng xác nhận.
- Log `class_id`, `label`, `confidence`, `probabilities` (cả 3 lớp) cho debugging trên thiết bị.

## 6. Giao tiếp ESP32-S3

Chuỗi JSON qua UART/BLE: `{"label":"ripe","class_id":2,"confidence":0.93}` hoặc `{"status":"uncertain"}`. Firmware ESP32-S3 diễn giải thêm phần cứng (đèn, motor thu hoạch…) theo quyết định app, không phụ thuộc vào model.

## 7. Kiểm tra tương thích thực tế

- Chưa có model đã huấn luyện (dataset trống) → chưa thể đo accuracy/latency trên thiết bị. Sau khi huấn luyện và export, chạy các bước trong `export_report.json` (`validate` + `compare.max_abs_diff`, `same_label`) để xác nhận khớp Keras, rồi đo inference time trên thiết bị thật trước khi kết luận Float16/INT8.
- Không tuyên bố MediaPipe tương thích nếu chưa nhúng metadata — xem lưu ý ở đầu file.
