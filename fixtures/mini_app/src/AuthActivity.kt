package com.example.miniapp

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.example.miniapp.api.UserApiService
import com.example.miniapp.data.LoginResult

/**
 * Auth screen.
 *
 * Navigation conditions are annotated with `// klit:condition:` so the
 * klit-flow extractor can capture the full nested decision path for each edge.
 * Conditions are comma-separated; each part represents one decision level.
 */
class AuthActivity : AppCompatActivity() {

    private val api = UserApiService()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_auth)

        loginButton.setOnClickListener {
            api.login(
                username = usernameField.text.toString(),
                password = passwordField.text.toString()
            ) { loginResult ->
                when (loginResult) {
                    is LoginResult.Success -> handleLoginSuccess(loginResult)
                    is LoginResult.AuthError -> {
                        // klit:condition: loginResult is AuthError
                        startActivity(Intent(this, AuthErrorActivity::class.java))
                    }
                    is LoginResult.NetworkError -> {
                        // klit:condition: loginResult is NetworkError
                        startActivity(Intent(this, NetworkErrorActivity::class.java))
                    }
                }
            }
        }
    }

    /** Called when /api/auth/login succeeds; fetches the user profile next. */
    private fun handleLoginSuccess(result: LoginResult.Success) {
        api.getUser(result.userId) { user ->
            when (user.type) {
                2 -> {
                    // klit:condition: loginResult is Success, user.type == 2 (admin)
                    startActivity(Intent(this, AdminDashboardActivity::class.java))
                }
                else -> checkSession(result.token)
            }
        }
    }

    /**
     * Called for standard users (type == 1).
     * Checks /api/sessions/check to see whether the password has expired.
     */
    private fun checkSession(token: String) {
        api.checkSession(token) { session ->
            if (session.passwordExpired) {
                // klit:condition: loginResult is Success, user.type == 1, session.passwordExpired == true
                startActivity(Intent(this, PasswordExpiredActivity::class.java))
            } else {
                // klit:condition: loginResult is Success, user.type == 1, session.passwordExpired == false
                startActivity(Intent(this, DashboardActivity::class.java))
            }
        }
    }

    // Legacy direct login used by Repository.authenticate()
    fun login(username: String, password: String): Boolean {
        if (username == "admin") {
            startActivity(Intent(this, ProfileActivity::class.java))
            return true
        }
        return false
    }
}
