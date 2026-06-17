package com.example.miniapp

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

/**
 * Shown when the login API succeeds for a standard user (type == 1)
 * but the session check reports passwordExpired == true.
 */
class PasswordExpiredActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_password_expired)
    }
}
