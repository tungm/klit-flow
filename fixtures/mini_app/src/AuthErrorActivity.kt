package com.example.miniapp

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

/** Error screen shown when the login API returns an authentication failure. */
class AuthErrorActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_auth_error)
    }
}
