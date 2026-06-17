package com.example.miniapp

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    private val navHelper = NavigationHelper()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        loginButton.setOnClickListener {
            startActivity(Intent(this, AuthActivity::class.java))
        }
        settingsButton.setOnClickListener {
            navHelper.openSettings(this)
        }
    }
}
