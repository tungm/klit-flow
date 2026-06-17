package com.example.miniapp

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

/**
 * Admin home screen shown when the login API succeeds and user.type == 2.
 */
class AdminDashboardActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_admin_dashboard)
    }
}
