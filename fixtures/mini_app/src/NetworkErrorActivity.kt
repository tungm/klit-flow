package com.example.miniapp

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

/** Error screen shown when the login API call fails due to a network error. */
class NetworkErrorActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_network_error)
    }
}
