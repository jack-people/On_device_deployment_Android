package com.agenew.clip

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

class ClipPredictor(private val context: Context) {

    private var imageInterpreter: Interpreter? = null
    private var textInterpreter: Interpreter? = null

    // CLIP 标准参数
    private val inputImageSize = 224
    private val textContextLength = 77
    private val embeddingSize = 512 // 根据你的模型，可能是 512, 768 等

    // CLIP 的归一化参数 (OpenAI 原版参数)
//    private val mean = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
//    private val std = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)

    private val mean = floatArrayOf(122.77f, 116.75f, 104.09f)
    private val std = floatArrayOf(68.50f, 66.63f, 70.32f)

    init {
        initModels()
    }

    private fun initModels() {
        // 1. 准备图像模型的 Options (尝试 GPU)
        val imageOptions = Interpreter.Options().apply {
            val compatList = CompatibilityList()
            if (compatList.isDelegateSupportedOnThisDevice) {
                val delegateOptions = compatList.bestOptionsForThisDevice
                addDelegate(GpuDelegate(delegateOptions))
            } else {
                setNumThreads(4)
            }
            setUseXNNPACK(true) // 即使是 CPU 模式也加速
        }

        // 2. 准备文本模型的 Options (建议优先 CPU，因为 CLIP Text 兼容性差)
        val textOptions = Interpreter.Options().apply {
            setNumThreads(4)
            setUseXNNPACK(true)
        }

        try {
            // 加载图像模型
            imageInterpreter = Interpreter(
                FileUtil.loadMappedFile(context, "clip_image_encoder_float32_new.tflite"),
                imageOptions
            )
            Log.d("TFLite", "图像模型加载成功")
        } catch (e: Exception) {
            Log.e("TFLite", "图像模型 GPU 初始化失败，尝试 CPU 降级: ${e.message}")
            val fallbackOptions = Interpreter.Options().setNumThreads(4)
            imageInterpreter = Interpreter(
                FileUtil.loadMappedFile(context, "clip_image_encoder_float32_new.tflite"),
                fallbackOptions
            )
        }

        try {
            // 加载文本模型 (直接用 textOptions，避免之前报错的 GPU 问题)
            textInterpreter = Interpreter(
                FileUtil.loadMappedFile(context, "clip_text_encoder_float32_new.tflite"),
                textOptions
            )
            Log.d("TFLite", "文本模型加载成功")
        } catch (e: Exception) {
            Log.e("TFLite", "文本模型加载失败: ${e.message}")
        }

        // --- 🕵️‍♀️ 侦探代码开始 ---
        val outputCount = imageInterpreter?.outputTensorCount ?: 0
        Log.e("ZJJ_DEBUG", "--------------------------------------------------")
        Log.e("ZJJ_DEBUG", "模型共有 $outputCount 个输出端")

        for (i in 0 until outputCount) {
            val tensor = imageInterpreter?.getOutputTensor(i)
            val shape = tensor?.shape()?.contentToString()
            val bytes = tensor?.numBytes()

            Log.e("ZJJ_DEBUG", "Output Index [$i]: Shape=$shape, Bytes=$bytes")

            if (bytes == 2048) { // 512 * 4
                Log.e("ZJJ_DEBUG", "✅ 找到目标！真正的 Embedding 在 Index [$i]")
            } else if (bytes == 153600) {
                Log.e("ZJJ_DEBUG", "❌ 发现原始层！这是导致报错的元凶 (Index [$i])")
            }
        }
        Log.e("ZJJ_DEBUG", "--------------------------------------------------")
        // --- 侦探代码结束 ---

        // --- 🕵️‍♀️ 侦探代码开始 ---
        val outputCountText = textInterpreter?.outputTensorCount ?: 0
        Log.e("ZJJ_DEBUG", "--------------------------------------------------")
        Log.e("ZJJ_DEBUG", "模型共有 $outputCountText 个输出端")

        for (i in 0 until outputCount) {
            val tensor = textInterpreter?.getOutputTensor(i)
            val shape = tensor?.shape()?.contentToString()
            val bytes = tensor?.numBytes()

            Log.e("ZJJ_DEBUG", "Output Index [$i]: Shape=$shape, Bytes=$bytes")

            if (bytes == 2048) { // 512 * 4
                Log.e("ZJJ_DEBUG", "✅ 找到目标！真正的 Embedding 在 Index [$i]")
            } else if (bytes == 157696) {
                Log.e("ZJJ_DEBUG", "❌ 发现原始层！这是导致报错的元凶 (Index [$i])")
            }
        }
        Log.e("ZJJ_DEBUG", "--------------------------------------------------")
        // --- 侦探代码结束 ---

    }

    /**
     * 1. 图像编码 (修复版：指定读取 Index 1)
     */
    fun encodeImage(bitmap: Bitmap): FloatArray {
        // 1. 预处理图片 (保持不变)
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(inputImageSize, inputImageSize,
                ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(mean, std))
            .build()

        var tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        tensorImage = imageProcessor.process(tensorImage)

        // 2. 准备输入 (必须包装成数组，因为我们要用 runForMultipleInputsOutputs)
        val inputs = arrayOf(tensorImage.buffer)

        // 3. 准备输出 Map
        // 关键点：我们要把 Buffer 绑定到 Index 1，而不是默认的 Index 0
        val outputs = HashMap<Int, Any>()

        val outputBuffer = ByteBuffer.allocateDirect(embeddingSize * 4) // 512 * 4 = 2048
        outputBuffer.order(ByteOrder.nativeOrder())

        // 🔥【核心修改】这里填 1，对应 Log 中的 "✅ 找到目标 Index [1]"
        outputs[1] = outputBuffer

        // 4. 运行推理 (使用多输入输出 API)
        imageInterpreter?.runForMultipleInputsOutputs(inputs, outputs)

        // 5. 获取结果并归一化
        outputBuffer.rewind()
        val embedding = FloatArray(embeddingSize)
        outputBuffer.asFloatBuffer().get(embedding)
        return normalizeVector(embedding)
    }

    /**
     * 2. 文本编码
     * @param tokenIds: 必须是 Tokenizer 处理后的 ID 数组 (例如 int[77])
     */
//    fun encodeText(tokenIds: IntArray): FloatArray {
//        // 确保输入 Shape 符合模型要求，通常是 [1, 77]
//        // 这里需要将 IntArray 转换为 ByteBuffer 或者直接传入多维数组
//        val input = Array(1) { tokenIds }
//        val outputBuffer = ByteBuffer.allocateDirect(embeddingSize * 4)
//        outputBuffer.order(ByteOrder.nativeOrder())
//
//        textInterpreter?.run(input, outputBuffer)
//
//        outputBuffer.rewind()
//        val embedding = FloatArray(embeddingSize)
//        outputBuffer.asFloatBuffer().get(embedding)
//        return normalizeVector(embedding)
//    }

    /**
     * 文本编码 (适配 PyTorch 导出的双输入模型)
     */
//    fun encodeText(tokenIds: IntArray): FloatArray {
//        // 1. 准备 Input IDs
//        // PyTorch 模型通常需要 Long (Int64) 或 Int32，视导出时的配置而定
//        // 如果报错类型不匹配，把 IntBuffer 换成 LongBuffer
//        val inputIdsBuffer = ByteBuffer.allocateDirect(1 * 77 * 4) // 假设是 Int32
//        inputIdsBuffer.order(ByteOrder.nativeOrder())
//        inputIdsBuffer.asIntBuffer().put(tokenIds)
//
//        // 2. 准备 Attention Mask
//        // 对于推理，Mask 通常全是 1 (关注所有 Token)
//        // 除非你有 Padding (补0) 的部分，补0的地方 Mask 应该是 0
//        val maskArray = IntArray(77) { index ->
//            if (tokenIds[index] != 0) 1 else 0 // 简单逻辑：非0即有效
//        }
//        val maskBuffer = ByteBuffer.allocateDirect(1 * 77 * 4)
//        maskBuffer.order(ByteOrder.nativeOrder())
//        maskBuffer.asIntBuffer().put(maskArray)
//
//        // 3. 构建输入数组 (顺序必须和 ONNX 导出的 input_names 顺序一致)
//        // 通常是 [input_ids, attention_mask]
//        val inputs = arrayOf(inputIdsBuffer, maskBuffer)
//
//        // 4. 准备输出 Map
//        val outputMap = HashMap<Int, Any>()
//        val outputBuffer = ByteBuffer.allocateDirect(512 * 4) // Float output
//        outputBuffer.order(ByteOrder.nativeOrder())
//        outputMap[0] = outputBuffer
//
//        // 5. 运行推理 (使用 runForMultipleInputsOutputs)
//        textInterpreter?.runForMultipleInputsOutputs(inputs, outputMap)
//
//        // 6. 获取结果
//        outputBuffer.rewind()
//        val embedding = FloatArray(512)
//        outputBuffer.asFloatBuffer().get(embedding)
//        return normalizeVector(embedding)
//    }


    /**
     * 文本编码 (最终修复版：Int64输入 + 指定输出Index 1)
     */
    fun encodeText(tokenIds: IntArray): FloatArray {
        // --- 1. 准备 Input IDs (Int64/Long) ---
        // 之前报错 input_ids mismatch，必须用 8 字节的 Long
        val inputIdsBuffer = ByteBuffer.allocateDirect(1 * 77 * 8)
        inputIdsBuffer.order(ByteOrder.nativeOrder())
        for (id in tokenIds) {
            inputIdsBuffer.putLong(id.toLong())
        }

        // --- 2. 准备 Attention Mask (Int64/Long) ---
        val maskBuffer = ByteBuffer.allocateDirect(1 * 77 * 8)
        maskBuffer.order(ByteOrder.nativeOrder())
        for (id in tokenIds) {
            val maskVal = if (id != 0) 1L else 0L
            maskBuffer.putLong(maskVal)
        }

        // 重置 Buffer 指针
        inputIdsBuffer.rewind()
        maskBuffer.rewind()

        // 构造输入数组 [input_ids, attention_mask]
        val inputs = arrayOf(inputIdsBuffer, maskBuffer)

        // --- 3. 准备输出 (修复点在这里！！！) ---
        val outputs = HashMap<Int, Any>()
        val outputBuffer = ByteBuffer.allocateDirect(embeddingSize * 4) // 512 * 4 = 2048
        outputBuffer.order(ByteOrder.nativeOrder())

        // ❌ 之前是 outputs[0] = outputBuffer
        // ✅ 根据 Log，正确的 Embedding 在 Index 1
        outputs[1] = outputBuffer

        // --- 4. 运行推理 ---
        textInterpreter?.runForMultipleInputsOutputs(inputs, outputs)

        // --- 5. 获取结果 ---
        outputBuffer.rewind()
        val embedding = FloatArray(embeddingSize)
        outputBuffer.asFloatBuffer().get(embedding)

        return normalizeVector(embedding)
    }

    /**
     * 计算两个向量的余弦相似度
     */
    fun calculateSimilarity(imageEmb: FloatArray, textEmb: FloatArray): Float {
        var dotProduct = 0.0f
        for (i in imageEmb.indices) {
            dotProduct += imageEmb[i] * textEmb[i]
        }
        // 因为我们之前已经做了 L2 归一化，所以点积就是余弦相似度
        return dotProduct
    }

    // L2 归一化向量
    private fun normalizeVector(v: FloatArray): FloatArray {
        var sum = 0.0f
        for (x in v) sum += x * x
        val magnitude = sqrt(sum)
        if (magnitude > 0) {
            for (i in v.indices) v[i] /= magnitude
        }
        return v
    }

    fun close() {
        imageInterpreter?.close()
        textInterpreter?.close()
    }
}