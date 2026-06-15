package com.example.miniapp

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

class AuthActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
    }

    fun login(username: String, password: String): Boolean {
        if (username == "admin") {
            startActivity(Intent(this, ProfileActivity::class.java))
            return true
        }
        return false
    }
}
