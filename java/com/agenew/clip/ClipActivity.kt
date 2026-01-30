package com.agenew.clip

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import android.content.Context
import androidx.activity.result.contract.ActivityResultContracts
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

class ClipActivity : AppCompatActivity(), View.OnClickListener {

    private val TAG = "ClipActivity"

    // 定义 UI 控件
    private lateinit var etInputText: EditText
    private lateinit var etInputImage: View // 假设这是点击触发加载图片的 View
    private lateinit var tvResult: TextView // 用于显示结果

    // CLIP 预测器实例
    private var predictor: ClipPredictor? = null

    // 当前选中的图片（实际项目中应从相册获取）
    private var currentBitmap: Bitmap? = null
    private val embeddingSize = 512
    private val predefinedTokens = HashMap<String, IntArray>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 1. 初始化 UI
        etInputText = findViewById(R.id.input_text)
        etInputImage = findViewById(R.id.input_image)
        tvResult = findViewById(R.id.tv_result) // 确保你的 layout 里有这个 TextView

        etInputText.setOnClickListener(this)
        etInputImage.setOnClickListener(this)

        // 2. 异步初始化模型 (避免阻塞主线程启动)
        initClipModel()
        initTokenMap()
    }

    private fun initClipModel() {
        lifecycleScope.launch(Dispatchers.Default) {
            try {
                Log.d(TAG, "ZJJ_DEBUG ClipActivity 正在加载模型...")
                // 初始化我们之前写的 ClipPredictor 类
                predictor = ClipPredictor(this@ClipActivity)

                // 预加载一张默认图片用于测试 (实际开发中应去掉)
                currentBitmap = BitmapFactory.decodeResource(resources, R.drawable.img_cat)

                withContext(Dispatchers.Main) {
                    Toast.makeText(this@ClipActivity, "CLIP 模型加载完成", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                Log.e(TAG, "ZJJ_DEBUG ClipActivity 模型加载失败", e)
            }
        }
    }

    override fun onClick(v: View?) {
        when (v?.id) {
            R.id.input_text -> {
                Log.d(TAG, "ZJJ_DEBUG ClipActivity 点击了文本输入框 - 触发推理")
                // 获取用户输入的文本
                val textInput = etInputText.text.toString()
                if (textInput.isNotEmpty() && currentBitmap != null) {
                    runClipInference(textInput, currentBitmap!!)
                } else {
                    Toast.makeText(this, "请输入文本并确保图片已加载", Toast.LENGTH_SHORT).show()
                }
            }
            R.id.input_image -> {
                Log.d(TAG, "ZJJ_DEBUG ClipActivity 点击了图片区域 - (此处应实现打开相册逻辑)")
                Toast.makeText(this, "这里应该打开相册选择图片", Toast.LENGTH_SHORT).show()
                // 模拟：为了测试，我们这里什么都不做，因为 onCreate 里已经加载了默认图
                Log.d(TAG, "ZJJ_DEBUG 打开相册")
                // 2. 启动相册
                pickImageLauncher.launch("image/*")
            }
        }
    }

    /**
     * 核心推理逻辑
     */
//    private fun runClipInference(text: String, bitmap: Bitmap) {
//        if (predictor == null) {
//            Log.e(TAG, "ZJJ_DEBUG ClipActivity 预测器尚未初始化")
//            return
//        }
//
//        // 使用协程在 IO 线程运行，避免卡顿 UI
//        lifecycleScope.launch(Dispatchers.IO) {
//            val startTime = System.currentTimeMillis()
//
//            try {
//                // 1. 编码图片
//                val imageEmbedding = predictor!!.encodeImage(bitmap)
//
//                // 2. 编码文本
//                // 注意：这里需要你实现真实的 Tokenizer
//                val tokens = getTokensFor(text)
//                val textEmbedding = predictor!!.encodeText(tokens)
//
//                // 3. 计算相似度
//                val score = predictor!!.calculateSimilarity(imageEmbedding, textEmbedding)
//
//                // 4. 回到主线程更新 UI
//                withContext(Dispatchers.Main) {
//                    val timeCost = System.currentTimeMillis() - startTime
//                    val resultStr = "文本: \"$text\"\n相似度: ${String.format("%.4f", score)}\n耗时: ${timeCost}ms"
//                    tvResult.text = resultStr
//                    Log.d(TAG, "ZJJ_DEBUG ClipActivity resultStr =" + resultStr)
//                }
//
//            } catch (e: Exception) {
//                Log.e(TAG, "ZJJ_DEBUG ClipActivity 推理出错", e)
//                withContext(Dispatchers.Main) {
//                    tvResult.text = "推理出错: ${e.message}"
//                }
//            }
//        }
//    }

    /**
     * 核心推理逻辑：支持多标签对比
     */
    private fun runClipInference(inputText: String, bitmap: Bitmap) {
        if (predictor == null) return

        lifecycleScope.launch(Dispatchers.IO) {
            val startTime = System.currentTimeMillis()

            try {
                // 1. 编码图片 (只做一次)
                val imageEmbedding = predictor!!.encodeImage(bitmap)

                // 2. 定义你要对比的标签列表
                // 实际场景中，这些可以来自用户输入，用逗号分隔，或者硬编码
                val labels = listOf("a photo of a cat", "a photo of a dog", "a photo of a car")

                // 存储原始分数
                val rawScores = FloatArray(labels.size)

                // 3. 循环编码文本并计算相似度
                // 因为 TFLite 模型输入 Batch=1，所以我们得循环跑
                for (i in labels.indices) {
                    val label = labels[i]
                    // 查表获取 Token
                    val tokens = getTokensFor(label)
                    // 编码文本
                    val textEmbedding = predictor!!.encodeText(tokens)
                    // 计算原始相似度 (Dot Product)
                    rawScores[i] = predictor!!.calculateSimilarity(imageEmbedding, textEmbedding)
                }

                // 4. 计算 Softmax 概率 (关键步骤！让分数变成百分比)
                val probabilities = softmax(rawScores)

                // 5. 格式化结果
                val sb = StringBuilder()
                sb.append("耗时: ${System.currentTimeMillis() - startTime}ms\n\n")

                // 找出最大概率的索引
                var maxIndex = 0
                for (i in labels.indices) {
                    val isBest = if (probabilities[i] == probabilities.maxOrNull()) "🏆 " else ""
                    sb.append("$isBest${labels[i]}\n")
                    sb.append("原始分: ${String.format("%.4f", rawScores[i])} -> 概率: ${String.format("%.1f", probabilities[i] * 100)}%\n\n")

                    if (probabilities[i] > probabilities[maxIndex]) maxIndex = i
                }

                // 6. 回到主线程更新 UI
                withContext(Dispatchers.Main) {
                    tvResult.text = sb.toString()
                    Log.d(TAG, "推理结果:\n$sb")
                }

            } catch (e: Exception) {
                Log.e(TAG, "推理出错", e)
                withContext(Dispatchers.Main) { tvResult.text = "出错: ${e.message}" }
            }
        }
    }

    /**
     * Softmax 函数：将原始分数转换为概率分布
     * CLIP 通常使用 100.0 作为缩放因子 (logit_scale)
     */
    private fun softmax(scores: FloatArray): FloatArray {
        val scale = 100.0f // CLIP 的魔法数字，让差距拉大
        val expScores = FloatArray(scores.size)
        var sumExp = 0.0f

        // 1. 计算 exp(x * scale)
        for (i in scores.indices) {
            expScores[i] = kotlin.math.exp(scores[i] * scale)
            sumExp += expScores[i]
        }

        // 2. 归一化
        val probs = FloatArray(scores.size)
        for (i in scores.indices) {
            probs[i] = expScores[i] / sumExp
        }
        return probs
    }

    override fun onDestroy() {
        super.onDestroy()
        // 释放 TFLite 资源，防止内存泄漏
        predictor?.close()
    }

    // 3. 填入你从 Python 跑出来的结果
    private fun initTokenMap() {
        // 注意：CLIP 的 context length 必须是 77

        // "a photo of a cat"
        predefinedTokens["a photo of a cat"] = intArrayOf(49406, 320, 1125, 539, 320, 2368, 49407,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        // "a photo of a dog"
        predefinedTokens["a photo of a dog"] = intArrayOf(49406, 320, 1125, 539, 320, 1929, 49407,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        // "a photo of a car"
        predefinedTokens["a photo of a car"] = intArrayOf(49406, 320, 1125, 539, 320, 1615, 49407,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    }

    /**
     * 4. 实现核心转换方法
     */
    private fun getTokensFor(text: String): IntArray {
        // 简单的数据清洗：去除首尾空格，转小写（假设你的 Key 都是小写）
        val cleanText = text.trim().lowercase()

        // 查表
        val tokens = predefinedTokens[cleanText]

        if (tokens != null) {
            return tokens
        } else {
            // 如果输入的文本不在我们的 Map 里
            Log.e(TAG, "ZJJ_DEBUG ClipActivity 找不到文本 '$text' 对应的 Token，请先在 Python 中生成！")

            // 返回一个空的 Token 数组 (或者全是 0)，但这会导致推理结果无意义
            // 建议：返回一个 "Unkown" 的通用 embedding 占位，或者直接 Toast 提示用户

            // 为了防止 Crash，返回一个全 0 数组 (CLIP 中 0 通常是 Padding，不起作用)
            return IntArray(77) { 0 }
        }
    }

    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let {
            try {
                // 将 URI 转为 Bitmap
                val inputStream = contentResolver.openInputStream(it)
                val bitmap = BitmapFactory.decodeStream(inputStream)
                inputStream?.close()

                if (bitmap != null) {
                    currentBitmap = bitmap
                    // 可以在这里更新 UI 显示选中的图片，比如 findViewById<ImageView>(...).setImageBitmap(bitmap)
                    Toast.makeText(this, "图片已更新", Toast.LENGTH_SHORT).show()
                    Log.d(TAG, "ZJJ_DEBUG 图片已更新: ${bitmap.width}x${bitmap.height}")
                }
            } catch (e: Exception) {
                Log.e(TAG, "读取图片失败", e)
            }
        }
    }

}