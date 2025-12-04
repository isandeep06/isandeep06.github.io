console.log("JavaScript code is running!");

// Firebase configuration for sandeep-maurya-portfolio project
const firebaseConfig = {
    apiKey: "AIzaSyCP6YvJDtnmxDOFv6zWqVYfIrK62reUGAE",
    authDomain: "sandeep-maurya-portfolio.firebaseapp.com",
    databaseURL: "https://sandeep-maurya-portfolio-default-rtdb.firebaseio.com",
    projectId: "sandeep-maurya-portfolio",
    storageBucket: "sandeep-maurya-portfolio.firebasestorage.app",
    messagingSenderId: "434641595792",
    appId: "1:434641595792:web:b59ccba03d28c2831fd715",
    measurementId: "G-XRSSZL8X31"
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);
const db = firebase.database();

// Function to update like count
function updateLikeCount(projectId, count) {
    const likeCountElement = document.querySelector(`.card[data-id="${projectId}"] .like-count`);
    if (likeCountElement) {
        likeCountElement.textContent = `${count} likes`;
    }
}

// Check all buttons when page loads
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.like-button').forEach(button => {
        const project = button.closest('.card');
        const projectId = project.dataset.id;

        if (!projectId) {
            console.warn('Project ID is missing!');
            return;
        }

        const likesRef = db.ref(`likes/${projectId}`);

        // Listen to like data from Firebase
        likesRef.on('value', snapshot => {
            const likeCount = snapshot.val() ? snapshot.val().count : 0;
            updateLikeCount(projectId, likeCount);

            const userLikes = JSON.parse(localStorage.getItem('userLikes') || '{}');
            if (userLikes[projectId]) {
                button.disabled = true;
                button.textContent = 'Liked';
            }
        });

        // Button click handler
        button.addEventListener('click', () => {
            const userLikes = JSON.parse(localStorage.getItem('userLikes') || '{}');

            // Prevent if already liked
            if (userLikes[projectId]) return;

            // Increase like count
            likesRef.once('value', snapshot => {
                const currentCount = snapshot.val() ? snapshot.val().count : 0;
                likesRef.set({ count: currentCount + 1 });

                userLikes[projectId] = true;
                localStorage.setItem('userLikes', JSON.stringify(userLikes));

                button.disabled = true;
                button.textContent = 'Liked';
            });
        });
    });
});
