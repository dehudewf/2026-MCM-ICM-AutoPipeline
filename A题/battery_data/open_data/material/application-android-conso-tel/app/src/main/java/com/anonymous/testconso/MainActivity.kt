package com.anonymous.testconso

import android.annotation.SuppressLint
import android.os.Bundle
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import android.content.Intent
import kotlin.random.Random

class MainActivity : AppCompatActivity() {

    private lateinit var btnRandom: Button
    private lateinit var etRed: EditText
    private lateinit var etGreen: EditText
    private lateinit var etBlue: EditText
    private lateinit var brightness: EditText
    private lateinit var spinnerCpu: Spinner
    private lateinit var cbAnimations: CheckBox
    private lateinit var nbAnimation: EditText
    private lateinit var rgDownload: RadioGroup
    private lateinit var rgReadWrite: RadioGroup
    private lateinit var btnStartTest: Button
    private lateinit var bandwidth: EditText

    @SuppressLint("WrongViewCast")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        initializeViews()
        setupSpinner()
        setupButton()
        rgDownload.setOnCheckedChangeListener { _, checkedId ->
            bandwidth.isEnabled = checkedId == R.id.rbDownloadLow || checkedId == R.id.rbDownloadHigh
            if (checkedId == R.id.rbDownloadNone) {
                bandwidth.text.clear()
            }
        }
        cbAnimations.setOnCheckedChangeListener { _, isChecked ->
            nbAnimation.isEnabled = isChecked
            if (!isChecked) {
                nbAnimation.text.clear()
            }
        }
    }

    private fun initializeViews() {
        etRed = findViewById(R.id.etRed)
        etGreen = findViewById(R.id.etGreen)
        etBlue = findViewById(R.id.etBlue)
        brightness = findViewById(R.id.brightness)
        spinnerCpu = findViewById(R.id.spinnerCpu)
        cbAnimations = findViewById(R.id.cbAnimations)
        rgDownload = findViewById(R.id.rgDownload)
        btnStartTest = findViewById(R.id.btnStartTest)
        rgReadWrite = findViewById(R.id.rgIO)
        btnRandom = findViewById(R.id.btnRandom)
        nbAnimation = findViewById(R.id.nbAnimation)
        nbAnimation.isEnabled = false
        bandwidth = findViewById(R.id.bandwith)
        bandwidth.isEnabled = false
        btnRandom.setOnClickListener {
            generateRandomScenario()
        }
    }

    private fun setupSpinner() {
        ArrayAdapter.createFromResource(
            this,
            R.array.cpu_loads,
            android.R.layout.simple_spinner_item
        ).also { adapter ->
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            spinnerCpu.adapter = adapter
        }
    }

    private fun setupButton() {
        btnStartTest.setOnClickListener { startTest() }
    }

    @SuppressLint("SetTextI18n")
    private fun generateRandomScenario() {
        // Générer des valeurs RGB aléatoires
        etRed.setText(Random.nextInt(0, 256).toString())
        etGreen.setText(Random.nextInt(0, 256).toString())
        etBlue.setText(Random.nextInt(0, 256).toString())

        // Générer une luminosité aléatoire
        brightness.setText(Random.nextInt(0, 101).toString())

        // Sélectionner une charge CPU aléatoire
        spinnerCpu.setSelection(Random.nextInt(spinnerCpu.count))

        // Activer ou désactiver les animations aléatoirement
        cbAnimations.isChecked = true

        // Sélectionner un mode de téléchargement aléatoire
        val downloadOptions = listOf(R.id.rbDownloadHigh)
        rgDownload.check(downloadOptions.random())

        // Sélectionner un mode Lecture/Ecriture aléatoire
        val ioOptions = listOf(R.id.rbIONone)
        rgReadWrite.check(ioOptions.random())
    }

    private fun startTest() {
        val red = getColorValue(etRed)
        val green = getColorValue(etGreen)
        val blue = getColorValue(etBlue)
        val brightness = brightness.text.toString().toIntOrNull() ?: 0
        if (!validateRGB(red, green, blue)) {
            Toast.makeText(this, "Veuillez entrer des valeurs RGB valides (0-255)", Toast.LENGTH_SHORT).show()
            return
        }

        val cpuLoad = spinnerCpu.selectedItem.toString()
        val animations = cbAnimations.isChecked

        val readWrite = when (rgReadWrite.checkedRadioButtonId) {
            R.id.rbRead -> "Read"
            R.id.rbWrite -> "Write"
            R.id.rbIONone -> "Off"
            else -> "Off"
        }

        val downloadMode = when (rgDownload.checkedRadioButtonId) {
            R.id.rbDownloadLow -> "Low"
            R.id.rbDownloadHigh -> "High"
            R.id.rbDownloadNone -> "Off"
            else -> "Off"
        }

        // Passer les données à TestActivity
        val intent = Intent(this, TestActivity::class.java).apply {
            putExtra("RED", red)
            putExtra("GREEN", green)
            putExtra("BLUE", blue)
            putExtra("BRIGHTNESS", brightness)
            putExtra("CPU_LOAD", cpuLoad)
            putExtra("ANIMATIONS", animations)
            putExtra("DOWNLOAD_MODE", downloadMode)
            putExtra("IO", readWrite)
            putExtra("NB_ANIM", nbAnimation.text.toString().toIntOrNull() ?: 0)
            putExtra("BANDWIDTH", bandwidth.text.toString().toIntOrNull() ?: 0)
        }
        startActivity(intent)
    }

    private fun getColorValue(editText: EditText): Int {
        return editText.text.toString().toIntOrNull()?.coerceIn(0, 255) ?: 0
    }

    private fun validateRGB(red: Int, green: Int, blue: Int): Boolean {
        return red in 0..255 && green in 0..255 && blue in 0..255
    }
}
