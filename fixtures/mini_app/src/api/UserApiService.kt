package com.example.miniapp.api

import com.example.miniapp.data.LoginResult

/** User profile returned by GET /api/users/{userId} */
data class User(val id: String, val type: Int, val name: String)

/** Session status returned by GET /api/sessions/check */
data class Session(val token: String, val passwordExpired: Boolean)

/**
 * HTTP API client for the mini-app backend.
 *
 * Each method models one REST endpoint:
 *   POST /api/auth/login         → login()
 *   GET  /api/users/{userId}     → getUser()
 *   GET  /api/sessions/check     → checkSession()
 *
 * Callbacks are invoked on the main thread (simulated here synchronously).
 */
class UserApiService {

    /** POST /api/auth/login — authenticate with username + password. */
    fun login(username: String, password: String, callback: (LoginResult) -> Unit) {
        try {
            if (username.isNotBlank() && password == "secret") {
                callback(LoginResult.Success(userId = "u_${username.hashCode()}", token = "tok_abc"))
            } else {
                callback(LoginResult.AuthError)
            }
        } catch (e: Exception) {
            callback(LoginResult.NetworkError)
        }
    }

    /** GET /api/users/{userId} — fetch the authenticated user's profile. */
    fun getUser(userId: String, callback: (User) -> Unit) {
        callback(User(id = userId, type = 1, name = "Demo User"))
    }

    /** GET /api/sessions/check — check whether the current session is still valid. */
    fun checkSession(token: String, callback: (Session) -> Unit) {
        callback(Session(token = token, passwordExpired = false))
    }
}
