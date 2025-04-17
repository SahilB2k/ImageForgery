import { initializeApp } from "https://www.gstatic.com/firebasejs/9.6.1/firebase-app.js";
import { getAnalytics } from "https://www.gstatic.com/firebasejs/9.6.1/firebase-analytics.js";

const firebaseConfig = {
    apiKey: "AIzaSyBX3FViKDITFrEapNYJSr9rZ0mPWefOxhU",
    authDomain: "imageforgery-ff2f3.firebaseapp.com",
    projectId: "imageforgery-ff2f3",
    storageBucket: "imageforgery-ff2f3.firebasestorage.app",
    messagingSenderId: "412293593917",
    appId: "1:412293593917:web:5105787312da9dbb82a354",
    measurementId: "G-8E9RW3772R"
};

// Initialize Firebase
try {
    firebase.initializeApp(firebaseConfig);
    console.log("Firebase initialized successfully");
} catch (error) {
    console.error("Firebase initialization error:", error);
}

firebase.auth().setPersistence(firebase.auth.Auth.Persistence.LOCAL);
const auth = firebase.auth();