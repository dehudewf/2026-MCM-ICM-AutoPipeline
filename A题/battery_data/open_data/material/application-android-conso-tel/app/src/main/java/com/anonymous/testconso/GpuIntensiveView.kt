package com.anonymous.testconso

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import java.util.concurrent.atomic.AtomicBoolean

class GpuIntensiveView(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private var particles: Array<Particle>? = null
    private val isAnimating = AtomicBoolean(false)
    private var particleColor: Int = Color.WHITE // Couleur par défaut

    init {
        paint.style = Paint.Style.FILL
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        particles?.forEach { it.resetPosition(w, h) }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (isAnimating.get()) {
            // Utiliser la couleur définie pour les particules
            paint.color = particleColor

            for (particle in particles!!) {
                particle.update(width, height)
                canvas.drawCircle(particle.x, particle.y, particle.size, paint)
            }

            invalidate()
        }
    }

    fun startAnimation(numParticles: Int) {
        if (particles == null || particles!!.size != numParticles) {
            particles = Array(numParticles) { Particle() }
        }
        isAnimating.set(true)
        invalidate()
    }

    fun stopAnimation() {
        isAnimating.set(false)
    }

    fun setParticleColor(red: Int, green: Int, blue: Int) {
        particleColor = Color.rgb(red, green, blue)
        invalidate() // Redessiner avec la nouvelle couleur
    }

    private class Particle {
        var x: Float = 0f
        var y: Float = 0f
        var size: Float = 0f
        var speedX: Float = 0f
        var speedY: Float = 0f

        fun resetPosition(width: Int, height: Int) {
            x = (Math.random() * width).toFloat()
            y = (Math.random() * height).toFloat()
            size = (Math.random() * 10 + 5).toFloat()
            speedX = (Math.random() * 8 - 4).toFloat()
            speedY = (Math.random() * 8 - 4).toFloat()
        }

        fun update(width: Int, height: Int) {
            x += speedX
            y += speedY

            if (x < 0 || x > width) {
                speedX *= -1
                x = x.coerceIn(0f, width.toFloat())
            }
            if (y < 0 || y > height) {
                speedY *= -1
                y = y.coerceIn(0f, height.toFloat())
            }
        }
    }
}
