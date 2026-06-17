package com.example.miniapp.data

import com.example.miniapp.AuthActivity

sealed class LoginResult {
    data class Success(val userId: String, val token: String) : LoginResult()
    object AuthError : LoginResult()
    object NetworkError : LoginResult()
}

class Repository {
    /** Legacy: used by AuthActivity.login() directly. */
    fun authenticate(activity: AuthActivity): Boolean {
        return activity.login("admin", "pass")
    }
}
