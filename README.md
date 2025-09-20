# Nomadz
## Multi-Agent Flow:
* Agent 1: Interpret query -> make rules from pdf.
* Agent 2: Check compliance -> change input based on rules
* Agent 3: Retrieve/transform data -> 
* Agent 4: Generate a response
* Agent 5: Last check -> data anonymiser  

Transparency: Show users what privacy measures were applied and why.

<div align="center">
  <img width="561" height="461" alt="Diagram" src="https://github.com/user-attachments/assets/fd6a73eb-6ef8-4d32-9fbd-aec62ee997eb" />
</div>

## Project diagram
WIP

## 🔧 Firebase Setup

Follow these steps to enable Firebase in this project:

---

### 1. Create a Firebase Project
- Go to the [Firebase Console](https://console.firebase.google.com/).  
- Sign in with your Google account.  
- Click **Add Project** and create a new project.  

---

### 2. Enable Authentication & Firestore
- In the Firebase Console, go to **Build → Authentication** and set up a new authentication method (e.g., Email/Password).  
- Still under **Build**, go to **Firestore Database** and create a new database.  

---

### 3. Get Your Firebase Config
- In the Firebase Console, click the **⚙️ Settings icon** (Project Settings) in the left sidebar.  
- Under **Your Apps**, create a new **Web App**.  
- Copy the config values that Firebase gives you.  

---

### 4. Add Environment Variables
- In your project root, create a file called `.env.local`.  
- Paste in the Firebase config values you copied earlier, like this:

```env
NEXT_PUBLIC_FIREBASE_API_KEY=your_api_key
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
NEXT_PUBLIC_FIREBASE_PROJECT_ID=your_project_id
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your_project.appspot.com
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=your_sender_id
NEXT_PUBLIC_FIREBASE_APP_ID=your_app_id
NEXT_PUBLIC_FIREBASE_MEASUREMENT_ID=your_measurement_id
```

### 5. Set Firestore Security Rules

- In the Firebase Console, go to Firestore Database → Rules.

- Replace the default rules with the following:

  ```rules
  rules_version = '2';
  service cloud.firestore {
    match /databases/{database}/documents {
      
      // Each authenticated user can only access their own user document
      match /users/{userId} {
        allow read, write: if request.auth != null && request.auth.uid == userId;
  
        // Chats inside each user document – only accessible by the same user
        match /chats/{chatId} {
          allow read, write: if request.auth != null && request.auth.uid == userId;
        }
      }
    }
  }
  ```
