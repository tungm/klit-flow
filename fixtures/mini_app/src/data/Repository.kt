package com.example.miniapp.data

import com.example.miniapp.AuthActivity

class Repository {
    fun authenticate(activity: AuthActivity): Boolean {
        return activity.login("admin", "pass")
    }
}
