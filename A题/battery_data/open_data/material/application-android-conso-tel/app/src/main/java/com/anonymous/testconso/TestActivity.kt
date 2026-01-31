package com.anonymous.testconso

import android.graphics.Color
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.anonymous.testconso.databinding.ActivityTestBinding
import kotlinx.coroutines.*
import okhttp3.OkHttpClient
import okhttp3.Request
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.sin
import java.io.*
import android.provider.Settings
import android.widget.Toast
import android.content.Intent
import android.net.Uri
import kotlin.random.Random

class TestActivity : AppCompatActivity() {

    private lateinit var binding: ActivityTestBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityTestBinding.inflate(layoutInflater)
        setContentView(binding.root)
        processIntentData()
    }
    // Request code for WRITE_SETTINGS permission
    private val PERMISSION_REQUEST_WRITE_SETTINGS = 123

    private fun processIntentData() {
        with(intent) {
            val red = getIntExtra("RED", 0)
            val green = getIntExtra("GREEN", 0)
            val blue = getIntExtra("BLUE", 0)
            val brightness = getIntExtra("BRIGHTNESS", 0)
            val cpuLoad = getStringExtra("CPU_LOAD") ?: "0%"
            val animations = getBooleanExtra("ANIMATIONS", false)
            val downloadMode = getStringExtra("DOWNLOAD_MODE") ?: "Off"
            val io = getStringExtra("IO") ?: "Off"
            val nbAnimation = getIntExtra("NB_ANIM", 0)
            val bandwidth = getIntExtra("BANDWIDTH", 0)
            setBackgroundColor(red, green, blue)
            setBrightness(brightness)
            performCpuLoad(cpuLoad)
            handleAnimations(animations, red, green, blue,nbAnimation)
            performDownload(downloadMode, bandwidth)
            performIO(io)
        }
    }

    private fun setBackgroundColor(red: Int, green: Int, blue: Int) {
        binding.testActivityLayout.setBackgroundColor(Color.rgb(red, green, blue))
    }

    private fun handleAnimations(animations: Boolean, red: Int, green: Int, blue: Int, nbAnimation: Int) {
        if (animations) {
            var animationCount = nbAnimation
            if (animationCount == 0) {
                animationCount = Random.nextInt(1, 30000)
            }
            binding.gpuIntensiveView.apply {
                startAnimation(animationCount)
                setParticleColor(red, green, blue)
            }
        }
    }

    private fun performCpuLoad(cpuLoad: String) {
        lifecycleScope.launch {
            try {
                withContext(Dispatchers.Default) {
                    when (cpuLoad) {
                        "Light" -> performLightCpuLoad()
                        "Medium" -> performMediumCpuLoad()
                        "Heavy" -> performHeavyCpuLoad()
                        "Max" -> performMaxCpuLoad()
                    }
                }
            } catch (e: CancellationException) {
                // Log.d("TestActivity", "CPU load task canceled")
            }
        }
    }

    private suspend fun performLightCpuLoad() = withContext(Dispatchers.Default) {
        val cpuTask1 = async { performCpuIntensiveTask(100) }
        val cpuTask2 = async { performCpuIntensiveTask(100) }
        cpuTask1.await()
        cpuTask2.await()
    }

    private suspend fun performMediumCpuLoad() = withContext(Dispatchers.Default) {
        val cpuTask1 = async { performCpuIntensiveTask(20) }
        val cpuTask2 = async { performCpuIntensiveTask(20) }
        val cpuTask3 = async { performCpuIntensiveTask(20) }
        val cpuTask4 = async { performCpuIntensiveTask(20) }
        val cpuTask5 = async { performCpuIntensiveTask(20) }
        val cpuTask6 = async { performCpuIntensiveTask(20) }
        val cpuTask7 = async { performCpuIntensiveTask(20) }
        val cpuTask8 = async { performCpuIntensiveTask(20) }
        cpuTask1.await()
        cpuTask2.await()
        cpuTask3.await()
        cpuTask4.await()
        cpuTask5.await()
        cpuTask6.await()
        cpuTask7.await()
        cpuTask8.await()
    }

    private suspend fun performHeavyCpuLoad() = withContext(Dispatchers.Default) {
        val matrixTask1 = async { performMatrixMultiplication(20) }
        val matrixTask2 = async { performMatrixMultiplication(20) }
        matrixTask1.await()
        matrixTask2.await()
    }

    private suspend fun performMaxCpuLoad() = withContext(Dispatchers.Default) {
        val cpuTask1 = async { performCpuIntensiveTask(2000) }
        val cpuTask2 = async { performCpuIntensiveTask(2000) }
        val cpuTask3 = async { performCpuIntensiveTask(2000) }
        val cpuTask4 = async { performCpuIntensiveTask(2000) }
        val matrixTask1 = async { performMatrixMultiplication(1000) }
        val matrixTask2 = async { performMatrixMultiplication(1000) }
        val matrixTask3 = async { performMatrixMultiplication(1000) }
        val matrixTask4 = async { performMatrixMultiplication(1000) }
        cpuTask1.await()
        cpuTask2.await()
        cpuTask3.await()
        cpuTask4.await()
        matrixTask1.await()
        matrixTask2.await()
        matrixTask3.await()
        matrixTask4.await()
    }

    private suspend fun performCpuIntensiveTask(restDuration: Long) {
        withContext(Dispatchers.Default) {
            try {
                while (isActive) {
                    repeat(10000) {
                        exp(sin(it.toDouble())) + exp(cos(it.toDouble()))
                        // Log.d("TestActivity", "CPU load")
                    }
                    delay(restDuration)
                }
            } catch (e: CancellationException) {
                // Log.d("TestActivity", "CPU intensive task canceled")
            }
        }
    }

    private fun performMatrixMultiplication(size: Int) {
        lifecycleScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    while (isActive) {
                        val matrix1 = Array(size) { DoubleArray(size) { Math.random() } }
                        val matrix2 = Array(size) { DoubleArray(size) { Math.random() } }
                        multiplyMatrices(matrix1, matrix2)
                        // Log.d("TestActivity", "CPU load")
                    }
                }
            } catch (e: CancellationException) {
                // Log.d("TestActivity", "CPU task canceled")
            }
        }
    }

    private fun multiplyMatrices(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> {
        val result = Array(a.size) { DoubleArray(b[0].size) }
        for (i in a.indices) {
            for (j in b[0].indices) {
                for (k in b.indices) {
                    result[i][j] += a[i][k] * b[k][j]
                }
            }
        }
        return result
    }
    private fun performDownload(downloadMode: String, bandwidth: Int) {
        lifecycleScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    val lowFrequencies = listOf<Long>( 7000, 7400, 7800, 8200, 8600, 9000, 9400, 9800, 10200, 10400, 10800, 11200,11600, 12000, 12400)
                    val highFrequencies = listOf<Long>(1, 200, 400, 600, 800, 1000)
                    val frequency = if (bandwidth != 0) bandwidth.toLong() else when (downloadMode) {
                        "Low" -> lowFrequencies.random()
                        "High" -> highFrequencies.random()
                        "Off" -> 0
                        else -> 0
                    }
                    when (downloadMode) {
                        "Low", "High" -> download("https://service-pages.greenspector.com/testBP/formatImage/img/drapeau.jpg", frequency)
                        "Off" -> {}
                        else -> {}
                    }
                }
            } catch (e: CancellationException) {
                Log.d("TestActivity", "Download task canceled")
            }
        }
    }
    private fun download(url: String, frequency: Long) {
        lifecycleScope.launch {
            while (isActive) {
                withContext(Dispatchers.IO) { // Exécuter sur un thread d'E/S
                    try {
                        val client = OkHttpClient()
                        val request = Request.Builder()
                            .url(url)
                            .build()

                        val response = client.newCall(request).execute()
                        if (response.isSuccessful) {
                            // Log.d("Download", "Succès: ${response.code}")
                        } else {
                            Log.e("Download", "Erreur: ${response.code}")
                        }
                    } catch (e: IOException) {
                        Log.e("Download", "Erreur: ${e.message}")
                    }
                }
                delay(frequency)
            }
        }
    }
    private fun performIO(io: String) {
        lifecycleScope.launch {
            when (io) {
                "Read" -> performRead()
                "Write" -> performWrite()
                "Off" -> {}
                else -> {}
            }
        }
    }
    private suspend fun performRead() {
        withContext(Dispatchers.IO) {
            val file = File(getExternalFilesDir(null), "test_read.txt")


            if (!file.exists()) {
                if (file.createNewFile()) {
                    println("Fichier créé !")
                    // Log.d("performRead", "Créé") // Ajout du log

                } else {
                    println("Erreur lors de la création du fichier.")
                    Log.d("performRead", "Erreur création") // Ajout du log

                }
            }
            val endTime = System.currentTimeMillis() + 35000 // 35 secondes

            while (System.currentTimeMillis() < endTime) {
                // Log.d("performRead", "Lecture en route") // Ajout du log
                try {
                    BufferedReader(FileReader(file)).use { reader ->
                        Log.d("performRead", "Lu: Ligne") // Ajout du log
                        reader.forEachLine { line ->
                            println("Lu: $line")
                        }
                    }
                } catch (e: IOException) {
                    println("Erreur de lecture: ${e.message}")
                    Log.d("performRead", "Erreur lecture") // Ajout du log

                }
                delay(100)
            }
        }
    }
    private suspend fun performWrite() {
        withContext(Dispatchers.IO) {
            try {
                val file = File(getExternalFilesDir(null), "test_write.txt")
                var counter = 0
                while (isActive) { // Run as long as the screen is displayed (isActive)
                    BufferedWriter(FileWriter(file, true)).use { writer ->
                        writer.write("Ligne ${++counter}\n")
                        // Log.d("PerformWrite", "Ecris: Ligne $counter")
                    }
                    delay(100) // Introduce a delay between writes
                }
            } catch (e: IOException) {
                println("Erreur d'écriture: ${e.message}")
            } catch (e: CancellationException) {
                // Log.d("TestActivity", "Write task canceled")
            }
        }
    }
    private fun setBrightness(brightness: Int) {
        if (checkWriteSettingsPermission()) {
            Settings.System.putInt(contentResolver, Settings.System.SCREEN_BRIGHTNESS, brightness)
//            val layoutParams = window.attributes
//            layoutParams.screenBrightness = brightness / 255.0f
//            window.attributes = layoutParams
        }
    }

    // Vérifie si la permission WRITE_SETTINGS est accordée
    private fun checkWriteSettingsPermission(): Boolean {
        return if (Settings.System.canWrite(this)) {
            true
        } else {
            // Demande à l'utilisateur d'activer la permission
            val intent = Intent(Settings.ACTION_MANAGE_WRITE_SETTINGS)
            intent.data = Uri.parse("package:$packageName")
            startActivityForResult(intent, PERMISSION_REQUEST_WRITE_SETTINGS)
            false
        }
    }

    // Gère le retour de l'activité de gestion des permissions
    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == PERMISSION_REQUEST_WRITE_SETTINGS) {
            if (Settings.System.canWrite(this)) {
                Toast.makeText(this, "Permission accordée", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Permission refusée", Toast.LENGTH_SHORT).show()
            }
        }
    }
}