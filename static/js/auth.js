// Firebase configuration
const firebaseConfig = {
    apiKey: process.env.REACT_APP_FIREBASE_API_KEY,
    authDomain: process.env.REACT_APP_FIREBASE_AUTH_DOMAIN,
    projectId: process.env.REACT_APP_FIREBASE_PROJECT_ID,
    storageBucket: process.env.REACT_APP_FIREBASE_STORAGE_BUCKET,
    messagingSenderId: process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID,
    appId: process.env.REACT_APP_FIREBASE_APP_ID,
    measurementId: process.env.REACT_APP_FIREBASE_MEASUREMENT_ID
  };
  
  export default firebaseConfig;

// Initialize Firebase
firebase.initializeApp(firebaseConfig);

// Google Auth Provider
const provider = new firebase.auth.GoogleAuthProvider();
provider.addScope('profile');
provider.addScope('email');

// Handle Google Sign In
function signInWithGoogle() {
    const loading = document.getElementById('loading');
    const error = document.getElementById('error');
    
    if (loading) loading.style.display = 'block';
    if (error) error.textContent = '';

    firebase.auth().signInWithPopup(provider)
        .then((result) => {
            return result.user.getIdToken();
        })
        .then((idToken) => {
            return fetch('/verify-token', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                credentials: 'include',
                body: JSON.stringify({ 
                    idToken: idToken,
                    timestamp: new Date().toISOString()
                })
            });
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                window.location.href = '/';
            } else {
                throw new Error(data.error || 'Authentication failed');
            }
        })
        .catch((error) => {
            console.error('Error:', error);
            if (error.code === 'auth/popup-blocked') {
                return firebase.auth().signInWithRedirect(provider);
            }
            if (error) error.textContent = error.message;
        })
        .finally(() => {
            if (loading) loading.style.display = 'none';
        });
}

// Listen for auth state changes
firebase.auth().onAuthStateChanged((user) => {
    const userStatus = document.getElementById('userStatus');
    if (user) {
        if (userStatus) userStatus.textContent = `Signed in as ${user.email}`;
    } else {
        if (userStatus) userStatus.textContent = 'Not signed in';
    }
});