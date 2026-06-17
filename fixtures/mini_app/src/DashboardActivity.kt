package com.example.miniapp

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

/** Home screen shown after a successful login API response. */
class DashboardActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_dashboard)
    }
}
