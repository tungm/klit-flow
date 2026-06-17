package com.example.miniapp

import android.app.Activity
import android.content.Intent

/**
 * Utility class that navigates to SettingsActivity from any calling screen.
 * The klit-flow extractor should trace this call back to the originating screen.
 */
class NavigationHelper {
    fun openSettings(activity: Activity) {
        if (activity.hasPermission("settings")) {
            startActivity(Intent(activity, SettingsActivity::class.java))
        }
    }

    fun openDashboardIfLoggedIn(activity: Activity, isLoggedIn: Boolean) {
        if (isLoggedIn) {
            startActivity(Intent(activity, DashboardActivity::class.java))
        }
    }
}
