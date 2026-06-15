package com.example.miniapp

fun formatDate(timestamp: Long): String {
    return timestamp.toString()
}

fun isValidEmail(email: String): Boolean {
    return email.contains("@")
}
